/**
 * WMG RL Digital Twin Platform - Main Application Entry Point
 * University of Warwick - WMG Automation Systems Group
 * 
 * This module initializes the complete digital twin platform, coordinating
 * all subsystems including robotics visualization, reinforcement learning
 * integration, and industrial automation components.
 */

import { StateManager } from './core/state-manager.js';
import { WebSocketManager } from './core/websocket-manager.js';
import { EventDispatcher } from './core/event-dispatcher.js';
import { InverseKinematics } from './robotics/inverse-kinematics.js';
import { RobotVisualizer } from './robotics/robot-visualizer.js';
import { ConveyorSystem } from './robotics/conveyor-system.js';
import { RobotController } from './robotics/robot-controller.js';
import { PerformanceDashboard } from './ui/performance-dashboard.js';
import { TrainingControls } from './ui/training-controls.js';
import { CodeDisplay } from './ui/code-display.js';

class WMGDigitalTwinPlatform {
    constructor() {
        this.initializationSteps = [
            'Initializing Core Systems',
            'Loading Robot Configurations',
            'Establishing Backend Connection',
            'Setting up 3D Environment',
            'Configuring User Interface',
            'Starting Animation Systems',
            'Platform Ready'
        ];
        this.currentStep = 0;
        
        // Core system components
        this.stateManager = null;
        this.webSocketManager = null;
        this.eventDispatcher = null;
        
        // Robotics components
        this.robotController = null;
        this.robotVisualizer = null;
        this.conveyorSystem = null;
        this.inverseKinematics = null;
        
        // User interface components
        this.performanceDashboard = null;
        this.trainingControls = null;
        this.codeDisplay = null;
        
        // Three.js scene components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.animationLoop = null;
        
        // Application state
        this.isInitialized = false;
        this.loadingProgress = 0;
        
        this.initializePlatform();
    }

    async initializePlatform() {
        console.log('WMG Digital Twin Platform initialization started');
        
        try {
            await this.updateLoadingProgress(0, this.initializationSteps[0]);
            await this.initializeCoreServices();
            
            await this.updateLoadingProgress(1, this.initializationSteps[1]);
            await this.configureRobotSystems();
            
            await this.updateLoadingProgress(2, this.initializationSteps[2]);
            await this.establishBackendConnection();
            
            await this.updateLoadingProgress(3, this.initializationSteps[3]);
            await this.setupThreeJSEnvironment();
            
            await this.updateLoadingProgress(4, this.initializationSteps[4]);
            await this.initializeUserInterface();
            
            await this.updateLoadingProgress(5, this.initializationSteps[5]);
            await this.startAnimationSystems();
            
            await this.updateLoadingProgress(6, this.initializationSteps[6]);
            await this.finalizeInitialization();
            
            console.log('WMG Digital Twin Platform successfully initialized');
            
        } catch (error) {
            console.error('Platform initialization failed:', error);
            this.handleInitializationError(error);
        }
    }

    async initializeCoreServices() {
        // Initialize state management system
        this.stateManager = new StateManager();
        
        // Create application state structure
        this.stateManager.createStore('application', {
            isLoading: true,
            currentView: 'main',
            systemStatus: 'initializing',
            lastUpdate: Date.now()
        });
        
        this.stateManager.createStore('robotics', {
            robots: [],
            selectedRobot: 0,
            kinematicsMode: 'automatic',
            jointAngles: [0, 0, 0, 0, 0, 0],
            targetPosition: [0, 0, 0],
            targetOrientation: [0, 0, 0]
        });
        
        this.stateManager.createStore('training', {
            isActive: false,
            currentEpisode: 0,
            maxEpisodes: 1000,
            reward: 0,
            performance: {
                cycleTime: 0,
                throughput: 0,
                safetyScore: 0,
                energyEfficiency: 0,
                oee: 0
            }
        });
        
        // Initialize event coordination system
        this.eventDispatcher = new EventDispatcher();
        
        // Set up global event listeners
        this.setupGlobalEventHandlers();
        
        await this.delay(300);
    }

    async configureRobotSystems() {
        // Define standard industrial robot geometry (DH parameters)
        const robotGeometry = [
            [0, 0, 0.6],      // Base link
            [0, 0, 0.5],      // Shoulder link  
            [1.2, 0, 0],      // Upper arm
            [0.95, 0, 0],     // Forearm
            [0, 0, 0.25],     // Wrist
            [0, 0, 0.12]      // Tool
        ];
        
        const jointLimits = [
            [-Math.PI, Math.PI],      // Base rotation
            [-Math.PI/2, Math.PI/2],  // Shoulder pitch
            [-2.35, 0.7],             // Elbow pitch
            [-Math.PI, Math.PI],      // Wrist roll
            [-2.09, 2.09],            // Wrist pitch
            [-2*Math.PI, 2*Math.PI]   // Tool rotation
        ];
        
        // Initialize inverse kinematics calculator
        this.inverseKinematics = new InverseKinematics(robotGeometry, jointLimits);
        
        // Initialize robot controller
        this.robotController = new RobotController(this.stateManager, this.eventDispatcher);
        
        await this.delay(400);
    }

    async establishBackendConnection() {
        // Initialize WebSocket communication manager
        this.webSocketManager = new WebSocketManager('ws://localhost:8000/ws');
        
        // Set up message handlers for different data types
        this.webSocketManager.onMessage('performance_update', (data) => {
            this.handlePerformanceUpdate(data);
        });
        
        this.webSocketManager.onMessage('training_progress', (data) => {
            this.handleTrainingProgress(data);
        });
        
        this.webSocketManager.onMessage('robot_state', (data) => {
            this.handleRobotStateUpdate(data);
        });
        
        this.webSocketManager.onMessage('plc_code', (data) => {
            this.handleGeneratedCode(data);
        });
        
        // Attempt to establish connection
        await this.webSocketManager.connect();
        
        await this.delay(500);
    }

    async setupThreeJSEnvironment() {
        // Initialize Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0f1a);
        this.scene.fog = new THREE.Fog(0x0a0f1a, 15, 60);
        
        // Configure camera with optimal viewing angle
        const aspect = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(70, aspect, 0.1, 1000);
        this.camera.position.set(10, 8, 10);
        this.camera.lookAt(0, 0, 0);
        
        // Initialize renderer with advanced settings
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true,
            powerPreference: "high-performance"
        });
        
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.3;
        
        // Attach renderer to DOM
        const container = document.getElementById('threejsContainer');
        if (container) {
            container.appendChild(this.renderer.domElement);
        }
        
        // Set up advanced lighting system
        this.setupSceneLighting();
        
        // Create industrial environment
        this.createIndustrialEnvironment();
        
        // Initialize robot visualization system
        this.robotVisualizer = new RobotVisualizer(this.scene, this.inverseKinematics);
        
        // Initialize conveyor system
        this.conveyorSystem = new ConveyorSystem(this.scene);
        
        // Set up camera controls
        this.setupCameraControls();
        
        await this.delay(600);
    }

    setupSceneLighting() {
        // Ambient lighting for overall scene illumination
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Main directional light for factory simulation
        const mainLight = new THREE.DirectionalLight(0xffffff, 1.5);
        mainLight.position.set(12, 12, 8);
        mainLight.castShadow = true;
        
        // Configure shadow camera for optimal quality
        mainLight.shadow.mapSize.width = 4096;
        mainLight.shadow.mapSize.height = 4096;
        mainLight.shadow.camera.near = 0.5;
        mainLight.shadow.camera.far = 60;
        mainLight.shadow.camera.left = -15;
        mainLight.shadow.camera.right = 15;
        mainLight.shadow.camera.top = 15;
        mainLight.shadow.camera.bottom = -15;
        
        this.scene.add(mainLight);
        
        // Accent spot lighting for dramatic effect
        const spotLight1 = new THREE.SpotLight(0xff6b35, 1.0, 35, Math.PI / 6, 0.3, 2);
        spotLight1.position.set(-10, 10, 0);
        spotLight1.target.position.set(-4, 0, 0);
        spotLight1.castShadow = true;
        this.scene.add(spotLight1);
        this.scene.add(spotLight1.target);
        
        const spotLight2 = new THREE.SpotLight(0x3498db, 0.8, 30, Math.PI / 8, 0.4, 2);
        spotLight2.position.set(10, 8, 10);
        spotLight2.target.position.set(4, 0, 0);
        this.scene.add(spotLight2);
        this.scene.add(spotLight2.target);
        
        // Point light for ambient factory atmosphere
        const pointLight = new THREE.PointLight(0xf39c12, 0.6, 20);
        pointLight.position.set(0, 6, 0);
        this.scene.add(pointLight);
    }

    createIndustrialEnvironment() {
        // Factory floor with professional appearance
        const floorGeometry = new THREE.PlaneGeometry(25, 25);
        const floorMaterial = new THREE.MeshLambertMaterial({
            color: 0x2c3e50,
            transparent: true,
            opacity: 0.9
        });
        
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);
        
        // Industrial grid system for spatial reference
        const gridHelper = new THREE.GridHelper(25, 25, 0x34495e, 0x2c3e50);
        gridHelper.material.transparent = true;
        gridHelper.material.opacity = 0.7;
        this.scene.add(gridHelper);
        
        // Add factory infrastructure elements
        this.addFactoryInfrastructure();
    }

    addFactoryInfrastructure() {
        // Safety barrier posts around work area
        const barrierPositions = [
            [-8, 0, 3], [-4, 0, 3], [0, 0, 3], [4, 0, 3], [8, 0, 3],
            [-8, 0, -8], [-4, 0, -8], [0, 0, -8], [4, 0, -8], [8, 0, -8]
        ];
        
        barrierPositions.forEach(position => {
            const postGeometry = new THREE.CylinderGeometry(0.06, 0.06, 1.8, 12);
            const postMaterial = new THREE.MeshPhongMaterial({ color: 0xf39c12 });
            const post = new THREE.Mesh(postGeometry, postMaterial);
            post.position.set(position[0], position[1] + 0.9, position[2]);
            post.castShadow = true;
            this.scene.add(post);
        });
        
        // Control station panels
        this.addControlStations();
        
        // Overhead structural elements
        this.addOverheadStructure();
    }

    addControlStations() {
        const stationConfigurations = [
            { position: [-10, 0, 0], rotation: Math.PI / 2 },
            { position: [10, 0, 0], rotation: -Math.PI / 2 }
        ];
        
        stationConfigurations.forEach(config => {
            const stationGroup = new THREE.Group();
            
            // Main control panel
            const panelGeometry = new THREE.BoxGeometry(0.3, 1.5, 1.0);
            const panelMaterial = new THREE.MeshPhongMaterial({ color: 0x34495e });
            const panelMesh = new THREE.Mesh(panelGeometry, panelMaterial);
            panelMesh.position.set(0, 0.75, 0);
            panelMesh.castShadow = true;
            stationGroup.add(panelMesh);
            
            // HMI screen
            const screenGeometry = new THREE.PlaneGeometry(0.7, 0.5);
            const screenMaterial = new THREE.MeshBasicMaterial({ 
                color: 0x2ecc71,
                transparent: true,
                opacity: 0.8
            });
            const screen = new THREE.Mesh(screenGeometry, screenMaterial);
            screen.position.set(config.rotation > 0 ? -0.16 : 0.16, 1.0, 0);
            screen.rotation.y = config.rotation;
            stationGroup.add(screen);
            
            stationGroup.position.copy(new THREE.Vector3(config.position[0], config.position[1], config.position[2]));
            stationGroup.rotation.y = config.rotation;
            this.scene.add(stationGroup);
        });
    }

    addOverheadStructure() {
        // Overhead support beams for industrial authenticity
        const beamConfigurations = [
            { start: [-12, 5, -6], end: [12, 5, -6], dimensions: [24, 0.3, 0.3] },
            { start: [-12, 5, 6], end: [12, 5, 6], dimensions: [24, 0.3, 0.3] },
            { start: [-12, 5, -6], end: [-12, 5, 6], dimensions: [0.3, 0.3, 12] },
            { start: [12, 5, -6], end: [12, 5, 6], dimensions: [0.3, 0.3, 12] }
        ];
        
        beamConfigurations.forEach(beam => {
            const beamGeometry = new THREE.BoxGeometry(beam.dimensions[0], beam.dimensions[1], beam.dimensions[2]);
            const beamMaterial = new THREE.MeshPhongMaterial({ color: 0x7f8c8d });
            const beamMesh = new THREE.Mesh(beamGeometry, beamMaterial);
            
            beamMesh.position.set(
                (beam.start[0] + beam.end[0]) / 2,
                beam.start[1],
                (beam.start[2] + beam.end[2]) / 2
            );
            
            beamMesh.castShadow = true;
            this.scene.add(beamMesh);
        });
    }

    setupCameraControls() {
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        
        this.renderer.domElement.addEventListener('mousedown', (event) => {
            isDragging = true;
            previousMousePosition = { x: event.clientX, y: event.clientY };
        });
        
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!isDragging) return;
            
            const deltaX = event.clientX - previousMousePosition.x;
            const deltaY = event.clientY - previousMousePosition.y;
            
            const spherical = new THREE.Spherical();
            spherical.setFromVector3(this.camera.position);
            spherical.theta -= deltaX * 0.01;
            spherical.phi += deltaY * 0.01;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
            
            this.camera.position.setFromSpherical(spherical);
            this.camera.lookAt(0, 0, 0);
            
            previousMousePosition = { x: event.clientX, y: event.clientY };
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        this.renderer.domElement.addEventListener('wheel', (event) => {
            const zoomFactor = event.deltaY > 0 ? 1.1 : 0.9;
            this.camera.position.multiplyScalar(zoomFactor);
            this.camera.position.clampLength(4, 40);
        });
    }

    async initializeUserInterface() {
        // Initialize performance dashboard
        this.performanceDashboard = new PerformanceDashboard(this.stateManager, this.eventDispatcher);
        
        // Initialize training controls
        this.trainingControls = new TrainingControls(this.stateManager, this.webSocketManager, this.eventDispatcher);
        
        // Initialize code display system
        this.codeDisplay = new CodeDisplay(this.stateManager, this.eventDispatcher);
        
        // Set up robot status display
        this.setupRobotStatusDisplay();
        
        // Configure joint control interface
        this.setupJointControlInterface();
        
        await this.delay(400);
    }

    setupRobotStatusDisplay() {
        const robotStatusList = document.getElementById('robotStatusList');
        if (!robotStatusList) return;
        
        const robotConfigurations = [
            { id: 'robot_cell_a1', name: 'Industrial Robot Cell A1', status: 'active' },
            { id: 'robot_cell_b2', name: 'Industrial Robot Cell B2', status: 'active' },
            { id: 'robot_cell_c3', name: 'Industrial Robot Cell C3', status: 'idle' },
            { id: 'conveyor_system', name: 'Conveyor System', status: 'active' }
        ];
        
        robotConfigurations.forEach(robot => {
            const statusElement = document.createElement('div');
            statusElement.className = 'robot-status-item';
            statusElement.innerHTML = `
                <span class="status-indicator ${robot.status}"></span>
                <span class="status-text">${robot.name}: ${robot.status.charAt(0).toUpperCase() + robot.status.slice(1)}</span>
            `;
            robotStatusList.appendChild(statusElement);
        });
    }

    setupJointControlInterface() {
        const jointControlsContainer = document.getElementById('jointControlsContainer');
        if (!jointControlsContainer) return;
        
        for (let i = 0; i < 6; i++) {
            const jointControl = document.createElement('div');
            jointControl.className = 'joint-control-item';
            
            const jointLimits = [
                { min: -180, max: 180 },   // Joint 1
                { min: -90, max: 90 },     // Joint 2
                { min: -135, max: 40 },    // Joint 3
                { min: -180, max: 180 },   // Joint 4
                { min: -120, max: 120 },   // Joint 5
                { min: -360, max: 360 }    // Joint 6
            ];
            
            jointControl.innerHTML = `
                <label class="joint-label">Joint ${i + 1}:</label>
                <input type="range" 
                       class="joint-slider" 
                       id="joint${i + 1}" 
                       min="${jointLimits[i].min}" 
                       max="${jointLimits[i].max}" 
                       value="0"
                       data-joint-index="${i}">
                <span class="joint-value" id="joint${i + 1}Value">0.0°</span>
            `;
            
            jointControlsContainer.appendChild(jointControl);
            
            // Add event listener for joint control
            const slider = jointControl.querySelector('.joint-slider');
            const valueDisplay = jointControl.querySelector('.joint-value');
            
            slider.addEventListener('input', (event) => {
                const jointIndex = parseInt(event.target.dataset.jointIndex);
                const angle = parseFloat(event.target.value);
                
                valueDisplay.textContent = `${angle.toFixed(1)}°`;
                this.handleJointAngleChange(jointIndex, angle);
            });
        }
    }

    async startAnimationSystems() {
        // Start main rendering loop
        this.startRenderLoop();
        
        // Initialize performance monitoring
        this.startPerformanceMonitoring();
        
        // Set up periodic system updates
        this.setupPeriodicUpdates();
        
        await this.delay(300);
    }

    startRenderLoop() {
        const animate = () => {
            if (!this.isInitialized) return;
            
            // Update conveyor system animation
            if (this.conveyorSystem) {
                const trainingState = this.stateManager.getStore('training').getState();
                const conveyorSpeed = (trainingState.performance.throughput || 100) / 100;
                this.conveyorSystem.animate(conveyorSpeed);
            }
            
            // Update robot visualizations
            if (this.robotVisualizer) {
                const trainingState = this.stateManager.getStore('training').getState();
                this.robotVisualizer.updateRobotAnimations(trainingState.performance);
            }
            
            // Render the scene
            this.renderer.render(this.scene, this.camera);
            
            // Schedule next frame
            this.animationLoop = requestAnimationFrame(animate);
        };
        
        animate();
    }

    startPerformanceMonitoring() {
        // Initialize with demo data
        setTimeout(() => {
            this.handlePerformanceUpdate({
                cycleTime: 28.4,
                throughput: 127,
                safetyScore: 99.8,
                energyEfficiency: 92.1,
                oee: 94.3
            });
        }, 2000);
    }

    setupPeriodicUpdates() {
        // Update system status every 5 seconds
        setInterval(() => {
            this.updateSystemStatus();
        }, 5000);
        
        // Check WebSocket connection every 10 seconds
        setInterval(() => {
            this.checkConnectionStatus();
        }, 10000);
    }

    async finalizeInitialization() {
        this.isInitialized = true;
        
        // Update application state
        this.stateManager.getStore('application').setState({
            isLoading: false,
            systemStatus: 'operational',
            lastUpdate: Date.now()
        });
        
        // Hide loading overlay
        await this.hideLoadingOverlay();
        
        // Dispatch initialization complete event
        this.eventDispatcher.emit('platform:initialized', {
            timestamp: Date.now(),
            version: '1.0.0',
            components: this.getInitializedComponents()
        });
        
        await this.delay(500);
    }

    async hideLoadingOverlay() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.opacity = '0';
            
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
            }, 800);
        }
    }

    async updateLoadingProgress(stepIndex, stepMessage) {
        this.currentStep = stepIndex;
        const progress = (stepIndex / (this.initializationSteps.length - 1)) * 100;
        
        console.log(`Initialization Step ${stepIndex + 1}: ${stepMessage}`);
        
        // Update loading progress if elements exist
        const progressElement = document.getElementById('loadingProgress');
        if (progressElement) {
            progressElement.style.width = `${progress}%`;
        }
        
        await this.delay(200);
    }

    setupGlobalEventHandlers() {
        // Handle window resize
        window.addEventListener('resize', () => {
            if (this.camera && this.renderer) {
                this.camera.aspect = window.innerWidth / window.innerHeight;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(window.innerWidth, window.innerHeight);
            }
        });
        
        // Handle visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseAnimations();
            } else {
                this.resumeAnimations();
            }
        });
        
        // Handle application errors
        window.addEventListener('error', (event) => {
            console.error('Application error:', event.error);
            this.handleApplicationError(event.error);
        });
    }

    // Event Handlers
    handlePerformanceUpdate(performanceData) {
        const trainingStore = this.stateManager.getStore('training');
        trainingStore.setState({
            performance: performanceData
        });
        
        this.eventDispatcher.emit('performance:updated', performanceData);
    }

    handleTrainingProgress(progressData) {
        const trainingStore = this.stateManager.getStore('training');
        trainingStore.setState({
            currentEpisode: progressData.episode,
            reward: progressData.reward,
            isActive: progressData.episode > 0 && progressData.episode < 1000
        });
        
        this.eventDispatcher.emit('training:progress', progressData);
    }

    handleRobotStateUpdate(robotData) {
        const roboticsStore = this.stateManager.getStore('robotics');
        roboticsStore.setState({
            jointAngles: robotData.angles || roboticsStore.getState().jointAngles
        });
        
        this.eventDispatcher.emit('robot:state_updated', robotData);
    }

    handleGeneratedCode(codeData) {
        this.eventDispatcher.emit('code:generated', codeData);
    }

    handleJointAngleChange(jointIndex, angle) {
        if (this.robotController) {
            this.robotController.updateJointAngle(jointIndex, angle);
        }
        
        this.eventDispatcher.emit('joint:angle_changed', {
            jointIndex,
            angle,
            timestamp: Date.now()
        });
    }

    // Utility Methods
    updateSystemStatus() {
        const applicationStore = this.stateManager.getStore('application');
        applicationStore.setState({
            lastUpdate: Date.now()
        });
    }

    checkConnectionStatus() {
        if (this.webSocketManager) {
            const isConnected = this.webSocketManager.isConnected();
            this.updateConnectionIndicator(isConnected);
        }
    }

    updateConnectionIndicator(isConnected) {
        const indicator = document.getElementById('connectionIndicator');
        const statusDot = indicator?.querySelector('.status-dot');
        const statusText = indicator?.querySelector('.status-text');
        
        if (statusDot && statusText) {
            if (isConnected) {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Backend Connected';
            } else {
                statusDot.className = 'status-dot';
                statusText.textContent = 'Connecting...';
            }
        }
    }

    pauseAnimations() {
        if (this.animationLoop) {
            cancelAnimationFrame(this.animationLoop);
            this.animationLoop = null;
        }
    }

    resumeAnimations() {
        if (!this.animationLoop && this.isInitialized) {
            this.startRenderLoop();
        }
    }

    getInitializedComponents() {
        return {
            stateManager: !!this.stateManager,
            webSocketManager: !!this.webSocketManager,
            eventDispatcher: !!this.eventDispatcher,
            robotController: !!this.robotController,
            robotVisualizer: !!this.robotVisualizer,
            conveyorSystem: !!this.conveyorSystem,
            inverseKinematics: !!this.inverseKinematics,
            performanceDashboard: !!this.performanceDashboard,
            trainingControls: !!this.trainingControls,
            codeDisplay: !!this.codeDisplay,
            threeJsEnvironment: !!(this.scene && this.camera && this.renderer)
        };
    }

    handleInitializationError(error) {
        console.error('Platform initialization error:', error);
        
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            const loadingContent = loadingOverlay.querySelector('.loading-content');
            if (loadingContent) {
                loadingContent.innerHTML = `
                    <h2>Initialization Error</h2>
                    <p>Failed to initialize the digital twin platform. Please check the console for details.</p>
                    <button onclick="location.reload()">Retry</button>
                `;
            }
        }
    }

    handleApplicationError(error) {
        this.eventDispatcher.emit('application:error', {
            error: error.message,
            stack: error.stack,
            timestamp: Date.now()
        });
    }

    delay(milliseconds) {
        return new Promise(resolve => setTimeout(resolve, milliseconds));
    }
}

// Initialize the platform when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.wmgPlatform = new WMGDigitalTwinPlatform();
});

// Export for potential external access
export { WMGDigitalTwinPlatform };