"""
WMG RL Digital Twin Platform - Backend Main Server
University of Warwick - WMG Automation Systems Group

This module provides the main FastAPI server with integrated WebSocket support
for real-time communication between the reinforcement learning backend and 
the digital twin visualization frontend.
"""

import asyncio
import json
import logging
import os
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
from plc_generator import PLCCodeGenerator
from websocket_handler import WebSocketConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/training_logs/wmg_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

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
        self.plc_generator = PLCCodeGenerator()
        
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
                "host": "0.0.0.0",
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
            }
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

    def setup_middleware(self):
        """Configure CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_static_files(self):
        """Mount static file directories."""
        # Serve frontend files
        self.app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
        self.app.mount("/data", StaticFiles(directory="data"), name="data")

    def setup_routes(self):
        """Configure HTTP and WebSocket routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_main_page():
            """Serve the main application page."""
            return FileResponse("index.html")
        
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
            """Generate IEC 61499 compliant PLC code from trained model."""
            if not self.rl_agent or not self.rl_agent.is_trained():
                raise HTTPException(status_code=400, detail="Model not trained")
            
            code = await self.generate_optimized_code()
            return {"code": code, "generated_at": datetime.now().isoformat()}
        
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
                code = await self.generate_optimized_code()
                await self.connection_manager.send_personal_message({
                    "type": "plc_code",
                    "code": code,
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
        """Generate optimized PLC code from the trained RL model."""
        if not self.rl_agent or not self.rl_agent.is_trained():
            raise ValueError("RL agent not trained")
        
        # Get optimal policy from trained agent
        optimal_policy = self.rl_agent.get_optimal_policy()
        
        # Generate IEC 61499 compliant code
        plc_code = self.plc_generator.generate_code(
            policy=optimal_policy,
            environment_config=self.automation_environment.get_config(),
            performance_targets=self.get_performance_targets()
        )
        
        # Save generated code
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_file = f"data/generated_code/wmg_optimized_{timestamp}.st"
        
        os.makedirs(os.path.dirname(code_file), exist_ok=True)
        with open(code_file, 'w') as f:
            f.write(plc_code)
        
        logger.info(f"Generated PLC code saved to {code_file}")
        
        return plc_code

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