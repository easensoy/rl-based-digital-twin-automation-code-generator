/**
 * WMG Digital Twin Platform - Main Application Controller
 * University of Warwick - WMG Automation Systems Group
 * 
 * This script manages the complete application lifecycle including initialization,
 * module loading, and user interface management for the digital twin platform.
 */

class WMGDigitalTwinApplication {
    constructor() {
        this.initialized = false;
        this.modules = new Map();
        this.websocketManager = null;
        this.robotController = null;
        this.robotVisualizer = null;
        this.inverseKinematics = null;
        
        // Application state
        this.isTraining = false;
        this.connectionStatus = 'disconnected';
        this.systemUptime = Date.now();
        
        // Performance metrics
        this.performanceMetrics = {
            oee: 0.0,
            efficiency: 0.0,
            cycleTime: 0.0,
            throughput: 0.0,
            safetyScore: 100.0,
            energyEfficiency: 85.0
        };
        
        console.log('WMG Digital Twin Application starting...');
    }

    /**
     * Initialize the complete application
     */
    async initialize() {
        try {
            console.log('Starting application initialization...');
            
            // Initialize event dispatcher first
            this.initializeEventSystem();
            
            // Initialize WebSocket connection
            await this.initializeWebSocket();
            
            // Initialize 3D visualization
            await this.initializeVisualization();
            
            // Initialize robot control systems
            await this.initializeRobotSystems();
            
            // Setup user interface handlers
            this.setupUIHandlers();
            
            // Start application loops
            this.startApplicationLoops();
            
            // Hide loading screen and show application
            this.showApplication();
            
            this.initialized = true;
            console.log('WMG Digital Twin Application initialized successfully');
            
        } catch (error) {
            console.error('Application initialization failed:', error);
            this.showErrorMessage('Failed to initialize application: ' + error.message);
        }
    }

    /**
     * Initialize event system with fallback
     */
    initializeEventSystem() {
        // Create a basic event dispatcher if the module isn't available
        if (typeof window.wmgEventDispatcher === 'undefined') {
            console.warn('Event dispatcher not available, creating fallback');
            window.wmgEventDispatcher = {
                emit: (event, data) => console.log('Event:', event, data),
                on: (event, callback) => console.log('Event listener registered:', event),
                off: () => {},
                startProcessing: () => {}
            };
            window.WMG_EVENTS = {
                SYSTEM_STARTUP: 'SYSTEM_STARTUP',
                TRAINING_STARTED: 'TRAINING_STARTED',
                TRAINING_STOPPED: 'TRAINING_STOPPED',
                WEBSOCKET_CONNECTED: 'WEBSOCKET_CONNECTED',
                WEBSOCKET_DISCONNECTED: 'WEBSOCKET_DISCONNECTED'
            };
        }
    }

    /**
     * Initialize WebSocket connection with fallback
     */
    async initializeWebSocket() {
        try {
            if (typeof WebSocketManager !== 'undefined') {
                this.websocketManager = new WebSocketManager('ws://localhost:8000/ws');
                await this.websocketManager.connect();
                
                // Setup message handlers
                this.websocketManager.onMessage('training_progress', (data) => {
                    this.handleTrainingProgress(data);
                });
                
                this.websocketManager.onMessage('performance_update', (data) => {
                    this.updatePerformanceMetrics(data);
                });
                
                this.connectionStatus = 'connected';
                window.wmgEventDispatcher.emit(window.WMG_EVENTS.WEBSOCKET_CONNECTED);
                
            } else {
                console.warn('WebSocketManager not available, using simulation mode');
                this.connectionStatus = 'simulation';
                this.setupSimulationMode();
            }
        } catch (error) {
            console.error('WebSocket initialization failed:', error);
            this.connectionStatus = 'error';
            this.setupSimulationMode();
        }
        
        this.updateConnectionStatus();
    }

    /**
     * Initialize 3D visualization with fallback
     */
    async initializeVisualization() {
        try {
            // Check if Three.js is available
            if (typeof THREE === 'undefined') {
                throw new Error('Three.js not loaded');
            }
            
            const canvas = document.getElementById('robot-canvas');
            if (!canvas) {
                throw new Error('Robot canvas not found');
            }
            
            // Initialize basic Three.js scene
            await this.setupBasic3DScene(canvas);
            
            // Try to initialize advanced robot visualizer
            if (typeof RobotVisualizer !== 'undefined' && typeof InverseKinematics !== 'undefined') {
                this.inverseKinematics = new InverseKinematics();
                this.robotVisualizer = new RobotVisualizer(this.scene, this.inverseKinematics);
                console.log('Advanced robot visualization initialized');
            } else {
                console.warn('Robot visualizer modules not available, using basic 3D scene');
            }
            
        } catch (error) {
            console.error('Visualization initialization failed:', error);
            this.setupFallbackVisualization();
        }
    }

    /**
     * Setup basic Three.js scene as fallback
     */
    async setupBasic3DScene(canvas) {
        // Create basic scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1d3a);

        // Setup camera
        this.camera = new THREE.PerspectiveCamera(
            75, 
            canvas.clientWidth / canvas.clientHeight, 
            0.1, 
            1000
        );
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);

        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: canvas,
            antialias: true 
        });
        this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        this.renderer.shadowMap.enabled = true;

        // Add basic lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        this.scene.add(directionalLight);

        // Add grid
        const gridHelper = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);

        // Add basic robot representation
        this.createBasicRobot();

        // Start render loop
        this.startRenderLoop();
        
        console.log('Basic 3D scene initialized');
    }

    /**
     * Create basic robot representation
     */
    createBasicRobot() {
        this.robotGroup = new THREE.Group();
        
        // Simple robot base
        const baseGeometry = new THREE.CylinderGeometry(0.5, 0.5, 0.2, 16);
        const baseMaterial = new THREE.MeshPhongMaterial({ color: 0x2d5a87 });
        const base = new THREE.Mesh(baseGeometry, baseMaterial);
        base.position.y = 0.1;
        this.robotGroup.add(base);
        
        // Simple arm
        const armGeometry = new THREE.BoxGeometry(0.1, 1.5, 0.1);
        const armMaterial = new THREE.MeshPhongMaterial({ color: 0x4a7ba7 });
        const arm = new THREE.Mesh(armGeometry, armMaterial);
        arm.position.y = 1.0;
        this.robotGroup.add(arm);
        
        this.scene.add(this.robotGroup);
    }

    /**
     * Initialize robot control systems
     */
    async initializeRobotSystems() {
        try {
            if (typeof RobotController !== 'undefined') {
                this.robotController = new RobotController();
                await this.robotController.initialize(this.robotVisualizer, this.inverseKinematics);
                console.log('Robot controller initialized');
            } else {
                console.warn('Robot controller not available, using simulation');
                this.setupRobotSimulation();
            }
        } catch (error) {
            console.error('Robot system initialization failed:', error);
            this.setupRobotSimulation();
        }
    }

    /**
     * Setup robot simulation mode
     */
    setupRobotSimulation() {
        // Create simple animation for robot group
        if (this.robotGroup) {
            setInterval(() => {
                this.robotGroup.rotation.y += 0.01;
            }, 50);
        }
    }

    /**
     * Setup user interface event handlers
     */
    setupUIHandlers() {
        // Training control buttons
        const startTrainingBtn = document.getElementById('start-training');
        const stopTrainingBtn = document.getElementById('stop-training');
        const resetSystemBtn = document.getElementById('reset-system');

        if (startTrainingBtn) {
            startTrainingBtn.addEventListener('click', () => this.startTraining());
        }
        
        if (stopTrainingBtn) {
            stopTrainingBtn.addEventListener('click', () => this.stopTraining());
        }
        
        if (resetSystemBtn) {
            resetSystemBtn.addEventListener('click', () => this.resetSystem());
        }

        // Code generation button
        const generateCodeBtn = document.getElementById('generate-code');
        if (generateCodeBtn) {
            generateCodeBtn.addEventListener('click', () => this.generateCode());
        }

        // Joint control sliders
        for (let i = 0; i < 6; i++) {
            const slider = document.getElementById(`joint${i}-slider`);
            if (slider) {
                slider.addEventListener('input', (e) => this.handleJointChange(i, e.target.value));
            }
        }

        // Window resize handler
        window.addEventListener('resize', () => this.handleResize());
        
        console.log('UI handlers setup complete');
    }

    /**
     * Start application loops for updates
     */
    startApplicationLoops() {
        // Update system uptime
        setInterval(() => {
            this.updateSystemUptime();
        }, 1000);

        // Update performance metrics display
        setInterval(() => {
            this.updatePerformanceDisplay();
        }, 2000);

        // Update robot coordinates display
        setInterval(() => {
            this.updateRobotCoordinates();
        }, 100);
    }

    /**
     * Start render loop for 3D visualization
     */
    startRenderLoop() {
        const animate = () => {
            requestAnimationFrame(animate);
            
            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }
        };
        animate();
    }

    /**
     * Hide loading screen and show application
     */
    showApplication() {
        const loadingOverlay = document.getElementById('loading-overlay');
        const applicationContainer = document.getElementById('application-container');
        
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
        
        if (applicationContainer) {
            applicationContainer.classList.remove('hidden');
        }
        
        console.log('Application interface displayed');
        
        // Emit startup event
        window.wmgEventDispatcher.emit(window.WMG_EVENTS.SYSTEM_STARTUP, {
            timestamp: Date.now(),
            version: '1.0.0'
        });
    }

    /**
     * Show error message
     */
    showErrorMessage(message) {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            const loadingContent = loadingOverlay.querySelector('.loading-content');
            if (loadingContent) {
                loadingContent.innerHTML = `
                    <div class="wmg-logo"></div>
                    <h2 style="color: #e74c3c;">Initialization Error</h2>
                    <p style="color: #f39c12;">${message}</p>
                    <button onclick="location.reload()" style="margin-top: 20px; padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Retry
                    </button>
                `;
            }
        }
    }

    /**
     * Setup simulation mode when WebSocket unavailable
     */
    setupSimulationMode() {
        console.log('Running in simulation mode');
        
        // Simulate performance metrics updates
        setInterval(() => {
            this.performanceMetrics.oee = 75 + Math.random() * 20;
            this.performanceMetrics.efficiency = 80 + Math.random() * 15;
            this.performanceMetrics.cycleTime = 25 + Math.random() * 10;
            this.performanceMetrics.throughput = 100 + Math.random() * 40;
            this.updatePerformanceDisplay();
        }, 3000);
    }

    /**
     * Handle training start
     */
    async startTraining() {
        console.log('Starting training...');
        this.isTraining = true;
        
        const startBtn = document.getElementById('start-training');
        const stopBtn = document.getElementById('stop-training');
        
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        
        this.updateTrainingStatus('Running');
        
        if (this.websocketManager) {
            this.websocketManager.send({ action: 'start_training' });
        } else {
            this.simulateTraining();
        }
        
        window.wmgEventDispatcher.emit(window.WMG_EVENTS.TRAINING_STARTED);
    }

    /**
     * Handle training stop
     */
    async stopTraining() {
        console.log('Stopping training...');
        this.isTraining = false;
        
        const startBtn = document.getElementById('start-training');
        const stopBtn = document.getElementById('stop-training');
        
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        
        this.updateTrainingStatus('Stopped');
        
        if (this.websocketManager) {
            this.websocketManager.send({ action: 'stop_training' });
        }
        
        window.wmgEventDispatcher.emit(window.WMG_EVENTS.TRAINING_STOPPED);
    }

    /**
     * Simulate training for demonstration
     */
    simulateTraining() {
        let episode = 0;
        let epsilon = 1.0;
        
        const trainingInterval = setInterval(() => {
            if (!this.isTraining) {
                clearInterval(trainingInterval);
                return;
            }
            
            episode++;
            epsilon = Math.max(0.01, epsilon * 0.995);
            
            const reward = -50 + Math.random() * 150;
            
            this.updateTrainingMetrics(episode, epsilon, reward);
            
            if (episode >= 1000) {
                this.stopTraining();
            }
        }, 100);
    }

    /**
     * Update training metrics display
     */
    updateTrainingMetrics(episode, epsilon, reward) {
        const episodeEl = document.getElementById('current-episode');
        const epsilonEl = document.getElementById('epsilon-value');
        const rewardEl = document.getElementById('current-reward');
        
        if (episodeEl) episodeEl.textContent = episode;
        if (epsilonEl) epsilonEl.textContent = epsilon.toFixed(3);
        if (rewardEl) rewardEl.textContent = reward.toFixed(1);
    }

    /**
     * Update performance metrics display
     */
    updatePerformanceDisplay() {
        const elements = {
            'oee-value': this.performanceMetrics.oee.toFixed(1) + '%',
            'header-oee': this.performanceMetrics.oee.toFixed(1) + '%',
            'header-efficiency': this.performanceMetrics.efficiency.toFixed(1) + '%',
            'cycle-time': this.performanceMetrics.cycleTime.toFixed(1) + 's',
            'throughput': this.performanceMetrics.throughput.toFixed(1) + '/min',
            'safety-score': this.performanceMetrics.safetyScore.toFixed(0) + '%',
            'energy-efficiency': this.performanceMetrics.energyEfficiency.toFixed(0) + '%'
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });
    }

    /**
     * Update system uptime display
     */
    updateSystemUptime() {
        const uptimeMs = Date.now() - this.systemUptime;
        const hours = Math.floor(uptimeMs / (1000 * 60 * 60));
        const minutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
        
        const uptimeElement = document.getElementById('system-uptime');
        if (uptimeElement) {
            uptimeElement.textContent = `${hours}h ${minutes}m`;
        }
    }

    /**
     * Update connection status display
     */
    updateConnectionStatus() {
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        
        if (statusDot && statusText) {
            statusDot.className = 'status-dot';
            
            switch (this.connectionStatus) {
                case 'connected':
                    statusDot.classList.add('connected');
                    statusText.textContent = 'Connected';
                    break;
                case 'simulation':
                    statusDot.classList.add('warning');
                    statusText.textContent = 'Simulation Mode';
                    break;
                case 'error':
                    statusDot.classList.add('error');
                    statusText.textContent = 'Connection Error';
                    break;
                default:
                    statusText.textContent = 'Disconnected';
            }
        }
    }

    /**
     * Update training status display
     */
    updateTrainingStatus(status) {
        const statusIndicator = document.getElementById('training-status');
        const statusText = document.getElementById('training-text');
        
        if (statusIndicator) {
            statusIndicator.className = 'status-indicator';
            if (status === 'Running') {
                statusIndicator.classList.add('connected');
            }
        }
        
        if (statusText) {
            statusText.textContent = status;
        }
    }

    /**
     * Handle joint slider changes
     */
    handleJointChange(jointIndex, value) {
        const valueElement = document.getElementById(`joint${jointIndex}-value`);
        if (valueElement) {
            valueElement.textContent = value + '°';
        }
        
        // Convert to radians and update robot if available
        const radians = (parseFloat(value) * Math.PI) / 180;
        
        if (this.robotController) {
            const angles = new Array(6).fill(0);
            angles[jointIndex] = radians;
            this.robotController.updateJointAngles(angles);
        }
    }

    /**
     * Update robot coordinates display
     */
    updateRobotCoordinates() {
        // Simulate robot position updates
        const time = Date.now() * 0.001;
        const x = Math.sin(time * 0.5) * 2;
        const y = 1 + Math.sin(time * 0.3) * 0.5;
        const z = Math.cos(time * 0.5) * 2;
        
        const xElement = document.getElementById('robot-x');
        const yElement = document.getElementById('robot-y');
        const zElement = document.getElementById('robot-z');
        
        if (xElement) xElement.textContent = x.toFixed(3);
        if (yElement) yElement.textContent = y.toFixed(3);
        if (zElement) zElement.textContent = z.toFixed(3);
    }

    /**
     * Handle window resize
     */
    handleResize() {
        if (this.camera && this.renderer) {
            const canvas = this.renderer.domElement;
            this.camera.aspect = canvas.clientWidth / canvas.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        }
    }

    /**
     * Generate PLC code
     */
    async generateCode() {
        console.log('Generating PLC code...');
        
        const generateBtn = document.getElementById('generate-code');
        const downloadBtn = document.getElementById('download-code');
        
        if (generateBtn) {
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
        }
        
        try {
            if (this.websocketManager) {
                this.websocketManager.send({ action: 'generate_code' });
            } else {
                // Simulate code generation
                setTimeout(() => {
                    if (generateBtn) {
                        generateBtn.disabled = false;
                        generateBtn.textContent = 'Generate PLC Code';
                    }
                    if (downloadBtn) {
                        downloadBtn.disabled = false;
                    }
                    this.addLogEntry('info', 'PLC code generated successfully (simulation)');
                }, 2000);
            }
        } catch (error) {
            console.error('Code generation failed:', error);
            if (generateBtn) {
                generateBtn.disabled = false;
                generateBtn.textContent = 'Generate PLC Code';
            }
        }
    }

    /**
     * Reset system
     */
    async resetSystem() {
        console.log('Resetting system...');
        
        if (this.websocketManager) {
            this.websocketManager.send({ action: 'reset_system' });
        }
        
        // Reset local state
        this.isTraining = false;
        this.updateTrainingStatus('Stopped');
        this.updateTrainingMetrics(0, 1.0, 0);
        
        this.addLogEntry('info', 'System reset completed');
    }

    /**
     * Add log entry
     */
    addLogEntry(level, message) {
        const logContent = document.getElementById('log-content');
        if (logContent) {
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = `log-entry ${level}`;
            entry.innerHTML = `
                <span class="log-timestamp">[${timestamp}]</span>
                <span class="log-level">${level.toUpperCase()}</span>
                <span class="log-message">${message}</span>
            `;
            logContent.appendChild(entry);
            logContent.scrollTop = logContent.scrollHeight;
        }
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM loaded, starting WMG Digital Twin Platform...');
    
    const app = new WMGDigitalTwinApplication();
    window.wmgApp = app; // Make globally accessible for debugging
    
    try {
        await app.initialize();
    } catch (error) {
        console.error('Application failed to start:', error);
    }
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (window.wmgApp) {
        if (document.hidden) {
            console.log('Application paused (tab hidden)');
        } else {
            console.log('Application resumed (tab visible)');
        }
    }
});

console.log('WMG Digital Twin Platform main script loaded');