"""
WMG RL Digital Twin Platform - Backend Main Server
University of Warwick - WMG Automation Systems Group

This module provides the main FastAPI server with integrated WebSocket support
for real-time communication between the reinforcement learning backend and 
the digital twin visualization frontend.
"""

import asyncio
import itertools
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Import custom modules
from rl_agent import DQNAgent, AutomationEnvironment
from websocket_handler import WebSocketConnectionManager
import eae_fbtype_compiler
import iec61499_exporter

# Configure logging. The log folder is created first: a fresh clone has no data/ folder.
os.makedirs("data/training_logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/training_logs/wmg_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# PLC code generation: the trained Q-network exported as an IEC 61499 Basic FB by
# iec61499_exporter.py. Any key can be overridden under "code_generation" in config.json.
CODE_GENERATION_DEFAULTS = {
    "output_dir": "data/generated_code/iec61499",  # one timestamped folder per generation
    "fb_name": "DQN_POLICY",
    "namespace": "Main",
    "state_input": "scalar",                        # or "array": one STATE : ARRAY[0..11] OF REAL
    "verify_states": 2000,                          # replay-buffer states replayed through the FB (0: skip)
    "tie_tolerance": 0.0,                           # > 0: excuse (and count) argmax differences on near-ties
    "eae_check": True,                              # compile with EAE's FBTypeCompiler.exe when installed
}


class WMGDigitalTwinServer:
    """
    Main server class that coordinates all backend services including
    reinforcement learning training, WebSocket communication, and 
    industrial automation code generation.
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="WMG RL Digital Twin Platform",
            description="Industrial Automation with Reinforcement Learning",
            version="1.0.0"
        )
        
        # Core components
        self.connection_manager = WebSocketConnectionManager()
        self.rl_agent = None
        self.automation_environment = None
        self.code_lock = asyncio.Lock()   # one PLC code generation at a time
        self.last_code_export = None      # details of the last generated FB
        
        # Training state
        self.training_active = False
        self.current_episode = 0
        self.max_episodes = 1000
        self.performance_history = []
        
        # System configuration
        self.config = self.load_configuration()
        
        # Initialize server components
        self.setup_middleware()
        self.setup_static_files()
        self.setup_routes()
        self.initialize_rl_components()
        
        logger.info("WMG Digital Twin Server initialized successfully")

    def load_configuration(self) -> Dict:
        """Load system configuration from JSON file."""
        config_path = Path("config/config.json")
        
        default_config = {
            "server": {
                "host": "127.0.0.1",   # this machine only; set a LAN address deliberately
                "port": 8000,
                "reload": False
            },
            "rl_agent": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "memory_size": 10000,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "target_update": 10
            },
            "automation": {
                "cycle_time_target": 30.0,
                "throughput_target": 120.0,
                "safety_threshold": 0.95,
                "energy_efficiency_target": 0.85
            },
            "code_generation": dict(CODE_GENERATION_DEFAULTS)
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration: {e}. Using defaults.")
        
        return default_config

    def own_origins(self, port: int) -> List[str]:
        """Origins of the dashboard this server serves on ``port``: a loopback name or
        the configured host. A fixed list of names, so a DNS-rebound name is refused."""
        hosts = {"127.0.0.1", "localhost", str(self.config["server"].get("host", "127.0.0.1"))}
        return sorted(f"http://{h}:{port}" for h in hosts - {"0.0.0.0", "::", ""})

    def origin_allowed(self, origin: Optional[str], port: int) -> bool:
        """For browser connections to /ws, which CORS does not cover. Clients that send
        no Origin (scripts, tests) are not browsers and are let through."""
        extra = self.config["server"].get("cors_origins", [])
        return origin is None or origin in self.own_origins(port) or origin in extra

    def setup_middleware(self):
        """CORS: the dashboard is served by this server, so it needs none. Only its own
        origins (and any listed under server.cors_origins) may read API responses."""
        server = self.config["server"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.own_origins(server.get("port", 8000)) + list(server.get("cors_origins", [])),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    def setup_static_files(self):
        """Mount the dashboard's static files. data/ is not served: generated code is
        downloaded through /api/code/files/ only."""
        # Mount frontend subdirectories at root level
        self.app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
        self.app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
        self.app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

    def code_generation_settings(self) -> Dict:
        return {**CODE_GENERATION_DEFAULTS, **self.config.get("code_generation", {})}

    def setup_routes(self):
        """Configure HTTP and WebSocket routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_main_page():
            """Serve the main application page."""
            return FileResponse("wmg-digital-twin.html")
        
        @self.app.get("/api/system/status")
        async def get_system_status():
            """Get current system status and health information."""
            return {
                "status": "operational",
                "training_active": self.training_active,
                "current_episode": self.current_episode,
                "max_episodes": self.max_episodes,
                "rl_agent_initialized": self.rl_agent is not None,
                "connected_clients": len(self.connection_manager.active_connections),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/api/training/start")
        async def start_training():
            """Start reinforcement learning training process."""
            if self.training_active:
                raise HTTPException(status_code=400, detail="Training already active")
            
            await self.start_rl_training()
            return {"message": "Training started successfully", "episode": self.current_episode}
        
        @self.app.post("/api/training/stop")
        async def stop_training():
            """Stop reinforcement learning training process."""
            if not self.training_active:
                raise HTTPException(status_code=400, detail="Training not active")
            
            await self.stop_rl_training()
            return {"message": "Training stopped successfully", "episode": self.current_episode}
        
        @self.app.post("/api/training/reset")
        async def reset_training():
            """Reset training state and environment."""
            await self.reset_training_system()
            return {"message": "Training system reset successfully"}
        
        @self.app.post("/api/code/generate")
        async def generate_plc_code():
            """Export the trained Q-network as an IEC 61499 Basic FB that executes it.

            "code" is the .fbt (EcoStruxure Automation Expert dialect). The other
            fields name the files written (.fbt and IEC 61131-3 .st) and their
            download URLs, the network size and cost, the check of the FB against
            the network (Q-values, argmax and setpoints; a failed check is a 500
            and nothing is kept) and the FB type compiler's messages."""
            if not self.rl_agent or not self.rl_agent.is_trained():
                raise HTTPException(status_code=400, detail="Model not trained")

            try:
                code = await self.generate_optimized_code()
            except (RuntimeError, ValueError) as e:
                logger.error(f"PLC code generation failed: {e}")
                raise HTTPException(status_code=500, detail=f"PLC code generation failed: {e}")
            return {"code": code, "generated_at": datetime.now().isoformat(), **self.code_export_details()}

        @self.app.get("/api/code/files/{export_id}/{file_name}")
        async def download_generated_code(export_id: str, file_name: str):
            """A generated .fbt or .st from <code_generation.output_dir>/<export_id>/."""
            base = Path(self.code_generation_settings()["output_dir"]).resolve()
            path = (base / export_id / file_name).resolve()
            if path.parent.parent != base or path.suffix not in (".fbt", ".st") or not path.is_file():
                raise HTTPException(status_code=404, detail="No such generated file")
            media_type = "application/xml" if path.suffix == ".fbt" else "text/plain; charset=utf-8"
            return FileResponse(path, media_type=media_type, filename=path.name)

        @self.app.get("/api/performance/history")
        async def get_performance_history():
            """Get historical performance data."""
            return {
                "history": self.performance_history[-100:],  # Last 100 records
                "total_records": len(self.performance_history)
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint for real-time communication."""
            port = (websocket.scope.get("server") or (None, None))[1]
            if not self.origin_allowed(websocket.headers.get("origin"), port):
                logger.warning(f"Refused WebSocket connection from origin {websocket.headers.get('origin')}")
                await websocket.close(code=1008)  # policy violation; the handshake is rejected
                return
            await self.connection_manager.connect(websocket)
            logger.info(f"Client connected. Total connections: {len(self.connection_manager.active_connections)}")
            
            try:
                # Send initial system state
                await self.send_system_state(websocket)
                
                # Listen for client messages
                while True:
                    data = await websocket.receive_text()
                    await self.handle_websocket_message(websocket, data)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
                logger.info(f"Client disconnected. Total connections: {len(self.connection_manager.active_connections)}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await self.connection_manager.disconnect(websocket)

    def initialize_rl_components(self):
        """Initialize reinforcement learning components."""
        try:
            # Initialize automation environment
            self.automation_environment = AutomationEnvironment(
                config=self.config["automation"]
            )
            
            # Initialize RL agent
            state_size = self.automation_environment.get_state_size()
            action_size = self.automation_environment.get_action_size()
            
            self.rl_agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                config=self.config["rl_agent"]
            )
            
            logger.info("RL components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RL components: {e}")
            raise

    async def handle_websocket_message(self, websocket: WebSocket, message: str):
        """Process incoming WebSocket messages from clients."""
        try:
            data = json.loads(message)
            action = data.get("action")
            
            if action == "start_training":
                await self.start_rl_training()
                
            elif action == "stop_training":
                await self.stop_rl_training()
                
            elif action == "reset_system":
                await self.reset_training_system()
                
            elif action == "generate_code":
                try:
                    code = await self.generate_optimized_code()
                except (RuntimeError, ValueError) as e:  # not trained, or the FB failed its check
                    logger.error(f"PLC code generation failed: {e}")
                    await self.connection_manager.send_personal_message({
                        "type": "plc_code_error",
                        "detail": f"PLC code generation failed: {e}",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                    return
                await self.connection_manager.send_personal_message({
                    "type": "plc_code",
                    "code": code,
                    **self.code_export_details(),
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
            elif action == "update_robot_angles":
                robot_data = data.get("robot_data", {})
                await self.handle_robot_update(robot_data)
                
            elif action == "ping":
                await self.connection_manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
            else:
                logger.warning(f"Unknown WebSocket action: {action}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from WebSocket client")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def start_rl_training(self):
        """Start the reinforcement learning training process."""
        if self.training_active:
            return
        
        self.training_active = True
        logger.info("Starting RL training process")
        
        # Broadcast training start to all clients
        await self.connection_manager.broadcast({
            "type": "training_status",
            "status": "started",
            "episode": self.current_episode,
            "timestamp": datetime.now().isoformat()
        })
        
        # Start training in background task
        asyncio.create_task(self.training_loop())

    async def stop_rl_training(self):
        """Stop the reinforcement learning training process."""
        if not self.training_active:
            return
        
        self.training_active = False
        logger.info("Stopping RL training process")
        
        # Broadcast training stop to all clients
        await self.connection_manager.broadcast({
            "type": "training_status",
            "status": "stopped",
            "episode": self.current_episode,
            "timestamp": datetime.now().isoformat()
        })

    async def reset_training_system(self):
        """Reset the entire training system to initial state."""
        self.training_active = False
        self.current_episode = 0
        self.performance_history.clear()
        
        if self.automation_environment:
            self.automation_environment.reset()
        
        if self.rl_agent:
            self.rl_agent.reset()
        
        logger.info("Training system reset completed")
        
        # Broadcast reset to all clients
        await self.connection_manager.broadcast({
            "type": "system_reset",
            "timestamp": datetime.now().isoformat()
        })

    async def training_loop(self):
        """Main training loop that runs in the background."""
        try:
            while self.training_active and self.current_episode < self.max_episodes:
                # Run one training episode
                episode_result = await self.run_training_episode()
                
                # Update episode counter
                self.current_episode += 1
                
                # Store performance data
                self.performance_history.append(episode_result)
                
                # Broadcast progress to clients
                await self.broadcast_training_progress(episode_result)
                
                # Broadcast performance update
                await self.broadcast_performance_update(episode_result)
                
                # Short delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # Training completed
            if self.current_episode >= self.max_episodes:
                self.training_active = False
                logger.info("Training completed successfully")
                
                await self.connection_manager.broadcast({
                    "type": "training_complete",
                    "final_episode": self.current_episode,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error in training loop: {e}")
            self.training_active = False

    async def run_training_episode(self) -> Dict:
        """Run a single training episode and return performance metrics."""
        # Reset environment for new episode
        state = self.automation_environment.reset()
        total_reward = 0
        steps = 0
        
        while not self.automation_environment.is_done() and steps < 200:
            # Agent selects action
            action = self.rl_agent.act(state)
            
            # Environment executes action
            next_state, reward, done = self.automation_environment.step(action)
            
            # Agent learns from experience
            self.rl_agent.remember(state, action, reward, next_state, done)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Train the agent
        if len(self.rl_agent.memory) > self.config["rl_agent"]["batch_size"]:
            self.rl_agent.replay()
        
        # Calculate performance metrics
        performance_metrics = self.automation_environment.get_performance_metrics()
        
        return {
            "episode": self.current_episode,
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": self.rl_agent.epsilon,
            "cycle_time": performance_metrics["cycle_time"],
            "throughput": performance_metrics["throughput"],
            "safety_score": performance_metrics["safety_score"],
            "energy_efficiency": performance_metrics["energy_efficiency"],
            "oee": performance_metrics["oee"],
            "timestamp": datetime.now().isoformat()
        }

    async def broadcast_training_progress(self, episode_result: Dict):
        """Broadcast training progress to all connected clients."""
        await self.connection_manager.broadcast({
            "type": "training_progress",
            "episode": episode_result["episode"],
            "reward": episode_result["total_reward"],
            "epsilon": episode_result["epsilon"],
            "timestamp": episode_result["timestamp"]
        })

    async def broadcast_performance_update(self, episode_result: Dict):
        """Broadcast performance metrics to all connected clients."""
        await self.connection_manager.broadcast({
            "type": "performance_update",
            "data": {
                "cycle_time": episode_result["cycle_time"],
                "throughput": episode_result["throughput"],
                "safety_score": episode_result["safety_score"],
                "energy_efficiency": episode_result["energy_efficiency"],
                "oee": episode_result["oee"]
            },
            "timestamp": episode_result["timestamp"]
        })

    async def generate_optimized_code(self) -> str:
        """Export the trained Q-network as PLC code that executes it.

        Writes an IEC 61499 Basic FB (<fb_name>.fbt, the form EcoStruxure Automation
        Expert imports) and the same logic as an IEC 61131-3 FUNCTION_BLOCK (.st) to
        <output_dir>/<timestamp>/, replays recent replay-buffer states through the
        generated .fbt to compare it with the network, and returns the .fbt text.
        The other export details are kept in self.last_code_export.
        """
        if not self.rl_agent or not self.rl_agent.is_trained():
            raise ValueError("RL agent not trained")

        settings = self.code_generation_settings()
        async with self.code_lock:
            # Copy the weights, the reference network and the check states here, in the
            # event loop: training keeps updating the live network while the export runs
            # in a worker thread, and building a network draws from torch's RNG.
            optimal_policy = self.rl_agent.get_optimal_policy()
            policy = iec61499_exporter.policy_from_state_dict(optimal_policy["model_state"], {
                "origin": "WMG RL Digital Twin server /api/code/generate",
                "gradient_steps": optimal_policy["training_steps"],
                "episodes_completed": self.current_episode,
                "epsilon": optimal_policy["epsilon"],
            })
            recent = itertools.islice(reversed(self.rl_agent.memory.buffer), int(settings["verify_states"]))
            states = [transition[0] for transition in recent]  # states the environment produced
            reference = iec61499_exporter.reference_network(policy) if states else None
            options = iec61499_exporter.ExportOptions(name=settings["fb_name"], namespace=settings["namespace"],
                                                      state_input=settings["state_input"])
            out_dir = self._new_code_dir(Path(settings["output_dir"]))
            export = await asyncio.to_thread(self._export_policy_code, policy, options, out_dir, states,
                                             reference, bool(settings["eae_check"]),
                                             float(settings["tie_tolerance"]))
            self.last_code_export = export

        check = export["verification"]
        summary = (f"replay-buffer {iec61499_exporter.verification_summary(check)}" if check
                   else "not checked against the network")
        logger.info(f"Generated IEC 61499 FB {export['fb_type']} in {export['files']['fbt']} ({summary})")

        return export["code"]

    def code_export_details(self) -> Dict:
        """The last code export without the code text, for API responses."""
        return {k: v for k, v in (self.last_code_export or {}).items() if k != "code"}

    @staticmethod
    def _new_code_dir(base: Path) -> Path:
        """<base>/<timestamp>, with a numeric suffix if that folder already exists."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out, n = base / stamp, 2
        while out.exists():
            out, n = base / f"{stamp}_{n}", n + 1
        return out

    @staticmethod
    def _export_policy_code(policy, options, out_dir: Path, states, reference, eae_check: bool,
                            tie_tolerance: float = 0.0) -> Dict:
        """Worker-thread part of generate_optimized_code: works on copies only."""
        result = iec61499_exporter.export(policy, out_dir, options)
        verification = None
        if states:
            verification = iec61499_exporter.verify_fbt(result.fbt_path, policy, states, net=reference,
                                                        tie_tolerance=tie_tolerance,
                                                        action_output=options.action_output)
            if not verification["ok"]:
                shutil.rmtree(out_dir, ignore_errors=True)  # do not leave a wrong FB behind
                raise RuntimeError(f"the generated FB failed its check against the Q-network "
                                   f"({iec61499_exporter.verification_summary(verification)}): {verification}")
        eae_compile = None
        compiler = eae_fbtype_compiler.find_fbtype_compiler() if eae_check else None
        if compiler is not None:
            report = eae_fbtype_compiler.compile_fbt(result.fbt_path, out_dir / "eae_build", compiler)
            # ok: the FB type compiler accepted the type; not an EAE solution import, not a runtime test.
            # The message lines are as the compiler printed them, in the three groups of classify().
            eae_compile = {"ok": report.ok, "compiler": str(compiler), "exit_code": report.returncode,
                           "bin_bytes": report.bin_bytes, "errors": report.errors, "warnings": report.warnings,
                           "context_messages": report.context_messages}
            if not report.ok:
                logger.warning(f"EAE FBTypeCompiler rejected {result.fbt_path}: {report.summary()}")
        return {
            "code": result.fbt_path.read_text(encoding="utf-8"),
            "format": "IEC 61499-2 Basic FB type (.fbt), EcoStruxure Automation Expert dialect",
            "fb_type": f"{options.namespace}.{options.name}",
            "files": {"fbt": str(result.fbt_path), "st": str(result.st_path)},
            "downloads": {kind: f"/api/code/files/{out_dir.name}/{path.name}"
                          for kind, path in (("fbt", result.fbt_path), ("st", result.st_path))},
            "network": policy.sizes,
            "parameters": policy.n_params,
            "macs_per_decision": policy.macs,
            "weights_sha256": policy.sha256,
            "verification": verification,
            "eae_compile": eae_compile,
        }

    async def handle_robot_update(self, robot_data: Dict):
        """Handle robot state updates from frontend."""
        # Broadcast robot state to all clients
        await self.connection_manager.broadcast({
            "type": "robot_state",
            "data": robot_data,
            "timestamp": datetime.now().isoformat()
        })

    async def send_system_state(self, websocket: WebSocket):
        """Send current system state to a newly connected client."""
        state_data = {
            "type": "system_state",
            "data": {
                "training_active": self.training_active,
                "current_episode": self.current_episode,
                "max_episodes": self.max_episodes,
                "rl_agent_ready": self.rl_agent is not None,
                "environment_ready": self.automation_environment is not None,
                "performance_history": self.performance_history[-10:] if self.performance_history else []
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await self.connection_manager.send_personal_message(state_data, websocket)

    def get_performance_targets(self) -> Dict:
        """Get current performance targets for code generation."""
        return {
            "cycle_time": self.config["automation"]["cycle_time_target"],
            "throughput": self.config["automation"]["throughput_target"],
            "safety_threshold": self.config["automation"]["safety_threshold"],
            "energy_efficiency": self.config["automation"]["energy_efficiency_target"]
        }

def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    server = WMGDigitalTwinServer()
    return server.app

def main():
    """Main entry point for the server application."""
    # Ensure required directories exist
    os.makedirs("data/training_logs", exist_ok=True)
    os.makedirs("data/generated_code", exist_ok=True)
    
    # Create the application
    server = WMGDigitalTwinServer()
    
    # Get server configuration
    server_config = server.config["server"]
    
    # Run the server
    logger.info(f"Starting WMG Digital Twin Server on {server_config['host']}:{server_config['port']}")
    
    uvicorn.run(
        server.app,
        host=server_config["host"],
        port=server_config["port"],
        reload=server_config["reload"],
        log_level="info"
    )

if __name__ == "__main__":
    main()