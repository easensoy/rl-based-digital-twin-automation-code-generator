/**
 * WMG Digital Twin Platform - Essential Main Application
 * University of Warwick - WMG Automation Systems Group
 */

class WMGDigitalTwin {
    constructor() {
        this.isInitialized = false;
        this.isTraining = false;
        this.connectionStatus = 'disconnected';
        this.websocketManager = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.robotGroup = null;
        
        // Performance metrics
        this.metrics = {
            oee: 91.8,
            efficiency: 87.4,
            cycleTime: 28.5,
            throughput: 125.0,
            safetyScore: 100.0,
            energyEfficiency: 85.0
        };
        
        console.log('WMG Digital Twin Platform starting...');
    }

    async initialize() {
        try {
            console.log('Initializing WMG Digital Twin Platform...');
            
            // Initialize WebSocket connection
            await this.initializeWebSocket();
            
            // Initialize 3D visualization
            this.initialize3D();
            
            // Setup UI handlers
            this.setupUIHandlers();
            
            // Start update loops
            this.startUpdateLoops();
            
            // Show application
            this.showApplication();
            
            this.isInitialized = true;
            this.addLogEntry('info', 'WMG Digital Twin Platform initialized successfully');
            
        } catch (error) {
            console.error('Initialization failed:', error);
            this.addLogEntry('error', 'Initialization failed: ' + error.message);
            // Still show the application in simulation mode
            this.showApplication();
            this.startSimulationMode();
        }
    }

    async initializeWebSocket() {
        try {
            if (typeof WebSocketManager !== 'undefined') {
                this.websocketManager = new WebSocketManager('ws://localhost:8000/ws');
                
                // Set connection timeout
                const timeout = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Connection timeout')), 3000)
                );
                
                await Promise.race([this.websocketManager.connect(), timeout]);
                
                this.connectionStatus = 'connected';
                this.addLogEntry('success', 'WebSocket connected successfully');
                
                // Setup message handlers
                this.websocketManager.onMessage('training_progress', (data) => {
                    this.updateTrainingProgress(data);
                });
                
                this.websocketManager.onMessage('performance_update', (data) => {
                    this.updateMetrics(data.data);
                });
                
            } else {
                throw new Error('WebSocketManager not available');
            }
        } catch (error) {
            console.warn('WebSocket connection failed:', error);
            this.connectionStatus = 'simulation';
            this.addLogEntry('warning', 'Running in simulation mode - backend not connected');
        }
        
        this.updateConnectionStatus();
    }

    initialize3D() {
        try {
            const canvas = document.getElementById('robot-canvas');
            if (!canvas || typeof THREE === 'undefined') {
                throw new Error('3D initialization failed - missing canvas or Three.js');
            }

            // Create scene
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x1a1d3a);

            // Create camera
            this.camera = new THREE.PerspectiveCamera(
                75, 
                canvas.clientWidth / canvas.clientHeight, 
                0.1, 
                1000
            );
            this.camera.position.set(5, 5, 5);
            this.camera.lookAt(0, 0, 0);

            // Create renderer
            this.renderer = new THREE.WebGLRenderer({ 
                canvas: canvas,
                antialias: true 
            });
            this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            this.renderer.setPixelRatio(window.devicePixelRatio);
            this.renderer.shadowMap.enabled = true;

            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            this.scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 10, 5);
            directionalLight.castShadow = true;
            this.scene.add(directionalLight);

            // Add grid
            const gridHelper = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
            this.scene.add(gridHelper);

            // Create simple robot
            this.createRobot();

            // Start render loop
            this.startRenderLoop();
            
            this.addLogEntry('info', '3D visualization initialized');
            
        } catch (error) {
            console.error('3D initialization failed:', error);
            this.addLogEntry('error', '3D visualization failed to initialize');
        }
    }

    createRobot() {
        this.robotGroup = new THREE.Group();
        
        // Robot base
        const baseGeometry = new THREE.CylinderGeometry(0.5, 0.5, 0.2, 16);
        const baseMaterial = new THREE.MeshPhongMaterial({ color: 0x6C1D45 });
        const base = new THREE.Mesh(baseGeometry, baseMaterial);
        base.position.y = 0.1;
        base.castShadow = true;
        this.robotGroup.add(base);
        
        // Robot arm
        const armGeometry = new THREE.BoxGeometry(0.1, 1.5, 0.1);
        const armMaterial = new THREE.MeshPhongMaterial({ color: 0xFFB81C });
        const arm = new THREE.Mesh(armGeometry, armMaterial);
        arm.position.set(0, 1.0, 0);
        arm.castShadow = true;
        this.robotGroup.add(arm);
        
        // End effector
        const endGeometry = new THREE.SphereGeometry(0.08, 8, 8);
        const endMaterial = new THREE.MeshPhongMaterial({ color: 0x00B7EB });
        const endEffector = new THREE.Mesh(endGeometry, endMaterial);
        endEffector.position.set(0, 1.8, 0);
        endEffector.castShadow = true;
        this.robotGroup.add(endEffector);
        
        this.scene.add(this.robotGroup);
    }

    setupUIHandlers() {
        // Training controls
        document.getElementById('start-training')?.addEventListener('click', () => this.startTraining());
        document.getElementById('stop-training')?.addEventListener('click', () => this.stopTraining());
        document.getElementById('reset-system')?.addEventListener('click', () => this.resetSystem());
        
        // Code generation
        document.getElementById('generate-code')?.addEventListener('click', () => this.generateCode());
        
        // Joint controls
        for (let i = 0; i < 3; i++) {
            const slider = document.getElementById(`joint${i}-slider`);
            if (slider) {
                slider.addEventListener('input', (e) => this.handleJointChange(i, e.target.value));
            }
        }
        
        // Log controls
        document.getElementById('clear-log')?.addEventListener('click', () => this.clearLog());
        
        // Window resize
        window.addEventListener('resize', () => this.handleResize());
        
        // Preset controls
        document.querySelectorAll('.preset-button').forEach(btn => {
            btn.addEventListener('click', (e) => this.handlePreset(e.target.dataset.preset));
        });
        
        console.log('UI handlers initialized');
    }

    startUpdateLoops() {
        // Update metrics display
        setInterval(() => {
            this.updateMetricsDisplay();
        }, 2000);
        
        // Update robot animation
        setInterval(() => {
            this.updateRobotAnimation();
        }, 100);
        
        // Update coordinates
        setInterval(() => {
            this.updateCoordinates();
        }, 200);
    }

    startRenderLoop() {
        const animate = () => {
            requestAnimationFrame(animate);
            
            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }
        };
        animate();
    }

    showApplication() {
        const loadingOverlay = document.getElementById('loading-overlay');
        const applicationContainer = document.getElementById('application-container');
        
        if (loadingOverlay) {
            loadingOverlay.style.opacity = '0';
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
            }, 500);
        }
        
        if (applicationContainer) {
            applicationContainer.classList.remove('hidden');
        }
        
        // Initialize display values
        this.updateMetricsDisplay();
        this.updateConnectionStatus();
    }

    startSimulationMode() {
        console.log('Starting simulation mode...');
        
        // Simulate changing metrics
        setInterval(() => {
            this.metrics.oee = 85 + Math.random() * 15;
            this.metrics.efficiency = 80 + Math.random() * 20;
            this.metrics.cycleTime = 25 + Math.random() * 10;
            this.metrics.throughput = 100 + Math.random() * 50;
        }, 3000);
    }

    async startTraining() {
        this.isTraining = true;
        
        document.getElementById('start-training').disabled = true;
        document.getElementById('stop-training').disabled = false;
        
        this.updateTrainingStatus('Running');
        this.addLogEntry('info', 'Training started');
        
        if (this.websocketManager) {
            this.websocketManager.send({ action: 'start_training' });
        } else {
            this.simulateTraining();
        }
    }

    async stopTraining() {
        this.isTraining = false;
        
        document.getElementById('start-training').disabled = false;
        document.getElementById('stop-training').disabled = true;
        
        this.updateTrainingStatus('Stopped');
        this.addLogEntry('info', 'Training stopped');
        
        if (this.websocketManager) {
            this.websocketManager.send({ action: 'stop_training' });
        }
    }

    simulateTraining() {
        let episode = 0;
        let epsilon = 1.0;
        
        const interval = setInterval(() => {
            if (!this.isTraining) {
                clearInterval(interval);
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

    async resetSystem() {
        this.isTraining = false;
        this.updateTrainingStatus('Stopped');
        this.updateTrainingMetrics(0, 1.0, 0);
        this.addLogEntry('info', 'System reset completed');
        
        if (this.websocketManager) {
            this.websocketManager.send({ action: 'reset_system' });
        }
    }

    async generateCode() {
        const generateBtn = document.getElementById('generate-code');
        const downloadBtn = document.getElementById('download-code');
        
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        
        try {
            if (this.websocketManager) {
                this.websocketManager.send({ action: 'generate_code' });
            } else {
                // Simulate
                setTimeout(() => {
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate PLC Code';
                    downloadBtn.disabled = false;
                    this.addLogEntry('success', 'PLC code generated (simulation)');
                }, 2000);
            }
        } catch (error) {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate PLC Code';
            this.addLogEntry('error', 'Code generation failed');
        }
    }

    handleJointChange(jointIndex, value) {
        const valueElement = document.getElementById(`joint${jointIndex}-value`);
        if (valueElement) {
            valueElement.textContent = value + '°';
        }
        
        // Update robot if available
        if (this.robotGroup && jointIndex === 0) {
            const radians = (parseFloat(value) * Math.PI) / 180;
            this.robotGroup.rotation.y = radians;
        }
    }

    handlePreset(preset) {
        this.addLogEntry('info', `Moving to ${preset} position`);
        
        // Reset sliders to home position
        for (let i = 0; i < 3; i++) {
            const slider = document.getElementById(`joint${i}-slider`);
            const valueEl = document.getElementById(`joint${i}-value`);
            if (slider && valueEl) {
                slider.value = 0;
                valueEl.textContent = '0°';
            }
        }
        
        if (this.robotGroup) {
            this.robotGroup.rotation.set(0, 0, 0);
        }
    }

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
                default:
                    statusText.textContent = 'Disconnected';
            }
        }
    }

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

    updateTrainingMetrics(episode, epsilon, reward) {
        const episodeEl = document.getElementById('current-episode');
        const epsilonEl = document.getElementById('epsilon-value');
        const rewardEl = document.getElementById('current-reward');
        
        if (episodeEl) episodeEl.textContent = episode;
        if (epsilonEl) epsilonEl.textContent = epsilon.toFixed(3);
        if (rewardEl) rewardEl.textContent = reward.toFixed(1);
    }

    updateMetricsDisplay() {
        const elements = {
            'header-oee': this.metrics.oee.toFixed(1) + '%',
            'header-efficiency': this.metrics.efficiency.toFixed(1) + '%',
            'cycle-time': this.metrics.cycleTime.toFixed(1) + 's',
            'throughput': this.metrics.throughput.toFixed(1) + '/min',
            'safety-score': this.metrics.safetyScore.toFixed(0) + '%',
            'energy-efficiency': this.metrics.energyEfficiency.toFixed(0) + '%',
            'oee-value': this.metrics.oee.toFixed(1) + '%'
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });
    }

    updateRobotAnimation() {
        if (this.robotGroup) {
            const time = Date.now() * 0.001;
            // Gentle rotation animation
            this.robotGroup.rotation.y += 0.005;
        }
    }

    updateCoordinates() {
        if (this.robotGroup) {
            const time = Date.now() * 0.001;
            const x = Math.sin(time * 0.5) * 2;
            const y = 1 + Math.sin(time * 0.3) * 0.3;
            const z = Math.cos(time * 0.5) * 2;
            
            const xEl = document.getElementById('robot-x');
            const yEl = document.getElementById('robot-y');
            const zEl = document.getElementById('robot-z');
            
            if (xEl) xEl.textContent = x.toFixed(3);
            if (yEl) yEl.textContent = y.toFixed(3);
            if (zEl) zEl.textContent = z.toFixed(3);
        }
    }

    updateTrainingProgress(data) {
        this.updateTrainingMetrics(data.episode, data.epsilon, data.reward);
    }

    updateMetrics(data) {
        if (data) {
            Object.assign(this.metrics, data);
        }
    }

    clearLog() {
        const logContent = document.getElementById('log-content');
        if (logContent) {
            logContent.innerHTML = '';
            this.addLogEntry('info', 'Log cleared');
        }
    }

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

    handleResize() {
        if (this.camera && this.renderer) {
            const canvas = this.renderer.domElement;
            this.camera.aspect = canvas.clientWidth / canvas.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Starting WMG Digital Twin Platform...');
    
    const app = new WMGDigitalTwin();
    window.wmgApp = app; // For debugging
    
    await app.initialize();
});

console.log('WMG Digital Twin Platform loaded');