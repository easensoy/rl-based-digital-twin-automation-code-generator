/**
 * WMG Digital Twin Platform - Enhanced Main Application
 * University of Warwick - WMG Automation Systems Group
 * Robot-Focused Digital Twin with Advanced 3D Visualization
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
        this.robotJoints = [];
        this.animationMixer = null;
        
        // Performance metrics
        this.metrics = {
            oee: 91.8,
            efficiency: 87.4,
            cycleTime: 28.5,
            throughput: 125.0,
            safetyScore: 100.0,
            energyEfficiency: 85.0
        };
        
        // Robot joint angles in degrees
        this.jointAngles = [-38, 26, -33, -11, 60, -14];
        
        console.log('WMG Digital Twin Platform starting...');
    }

    async initialize() {
        try {
            console.log('Initializing WMG Digital Twin Platform...');
            
            // Initialize WebSocket connection
            await this.initializeWebSocket();
            
            // Initialize enhanced 3D visualization
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
            console.log('Starting 3D initialization...');
            const canvas = document.getElementById('robot-canvas');
            
            if (!canvas) {
                throw new Error('Canvas element not found');
            }
            
            if (typeof THREE === 'undefined') {
                throw new Error('Three.js library not loaded');
            }

            console.log('Canvas found, Three.js loaded');

            // Create scene
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x1a1d3a);

            // Create camera with proper aspect ratio
            const width = canvas.clientWidth || 800;
            const height = canvas.clientHeight || 600;
            
            this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
            this.camera.position.set(5, 5, 5);
            this.camera.lookAt(0, 0, 0);

            console.log('Camera positioned at:', this.camera.position);

            // Create renderer
            this.renderer = new THREE.WebGLRenderer({ 
                canvas: canvas,
                antialias: true 
            });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

            console.log('Renderer created with size:', width, 'x', height);

            // Add bright lighting to ensure visibility
            const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
            this.scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
            directionalLight.position.set(10, 10, 5);
            directionalLight.castShadow = true;
            this.scene.add(directionalLight);

            // Add simple grid for reference
            const gridHelper = new THREE.GridHelper(10, 20, 0x6C1D45, 0x444444);
            this.scene.add(gridHelper);

            console.log('Lighting and grid added');

            // Create robot model
            this.createSimpleRobot();

            // Add camera controls
            this.addCameraControls();

            // Start render loop
            this.startRenderLoop();
            
            console.log('3D scene fully initialized');
            this.addLogEntry('info', '3D robot visualization initialized successfully');
            
        } catch (error) {
            console.error('3D initialization failed:', error);
            this.addLogEntry('error', '3D initialization failed: ' + error.message);
            
            // Try to create a fallback canvas message
            this.createFallbackCanvas();
        }
    }

    createAdvancedRobot() {
        this.robotGroup = new THREE.Group();
        this.robotJoints = [];
        
        // Enhanced robot base with industrial styling
        const baseGeometry = new THREE.CylinderGeometry(0.6, 0.8, 0.3, 20);
        const baseMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x6C1D45,
            shininess: 100,
            specular: 0x444444
        });
        const base = new THREE.Mesh(baseGeometry, baseMaterial);
        base.position.y = 0.15;
        base.castShadow = true;
        base.receiveShadow = true;
        this.robotGroup.add(base);
        this.robotJoints.push(base);
        
        // Robot shoulder joint with enhanced geometry
        const shoulderGeometry = new THREE.BoxGeometry(0.4, 0.6, 0.3);
        const shoulderMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xFFB81C,
            shininess: 120,
            specular: 0x666666
        });
        const shoulder = new THREE.Mesh(shoulderGeometry, shoulderMaterial);
        shoulder.position.set(0, 0.6, 0);
        shoulder.castShadow = true;
        shoulder.receiveShadow = true;
        this.robotGroup.add(shoulder);
        this.robotJoints.push(shoulder);
        
        // Upper arm link with industrial detailing
        const upperArmGeometry = new THREE.BoxGeometry(0.15, 1.8, 0.15);
        const upperArmMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x00B7EB,
            shininess: 110,
            specular: 0x555555
        });
        const upperArm = new THREE.Mesh(upperArmGeometry, upperArmMaterial);
        upperArm.position.set(0, 1.5, 0);
        upperArm.castShadow = true;
        upperArm.receiveShadow = true;
        this.robotGroup.add(upperArm);
        this.robotJoints.push(upperArm);
        
        // Add cable conduits for realism
        const cableGeometry = new THREE.CylinderGeometry(0.02, 0.02, 1.6, 8);
        const cableMaterial = new THREE.MeshBasicMaterial({ color: 0x333333 });
        const cable1 = new THREE.Mesh(cableGeometry, cableMaterial);
        cable1.position.set(0.08, 1.5, 0.08);
        this.robotGroup.add(cable1);
        
        // Elbow joint with spherical design
        const elbowGeometry = new THREE.SphereGeometry(0.2, 16, 16);
        const elbowMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x2ECC71,
            shininess: 130,
            specular: 0x777777
        });
        const elbow = new THREE.Mesh(elbowGeometry, elbowMaterial);
        elbow.position.set(0, 2.4, 0);
        elbow.castShadow = true;
        elbow.receiveShadow = true;
        this.robotGroup.add(elbow);
        this.robotJoints.push(elbow);
        
        // Forearm with enhanced proportions
        const forearmGeometry = new THREE.BoxGeometry(0.12, 1.2, 0.12);
        const forearmMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xE74C3C,
            shininess: 100,
            specular: 0x444444
        });
        const forearm = new THREE.Mesh(forearmGeometry, forearmMaterial);
        forearm.position.set(0, 3.2, 0);
        forearm.castShadow = true;
        forearm.receiveShadow = true;
        this.robotGroup.add(forearm);
        this.robotJoints.push(forearm);
        
        // Wrist assembly with industrial design
        const wristGeometry = new THREE.CylinderGeometry(0.15, 0.15, 0.2, 12);
        const wristMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x9B59B6,
            shininess: 140,
            specular: 0x888888
        });
        const wrist = new THREE.Mesh(wristGeometry, wristMaterial);
        wrist.position.set(0, 3.9, 0);
        wrist.castShadow = true;
        wrist.receiveShadow = true;
        this.robotGroup.add(wrist);
        this.robotJoints.push(wrist);
        
        // End effector with gripper detail
        const endGeometry = new THREE.BoxGeometry(0.3, 0.15, 0.1);
        const endMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xF39C12,
            shininess: 80,
            specular: 0x333333
        });
        const endEffector = new THREE.Mesh(endGeometry, endMaterial);
        endEffector.position.set(0, 4.2, 0);
        endEffector.castShadow = true;
        endEffector.receiveShadow = true;
        this.robotGroup.add(endEffector);
        
        // Add gripper fingers
        const fingerGeometry = new THREE.BoxGeometry(0.05, 0.1, 0.02);
        const fingerMaterial = new THREE.MeshPhongMaterial({ color: 0x666666 });
        
        const finger1 = new THREE.Mesh(fingerGeometry, fingerMaterial);
        finger1.position.set(0.1, 4.15, 0);
        this.robotGroup.add(finger1);
        
        const finger2 = new THREE.Mesh(fingerGeometry, fingerMaterial);
        finger2.position.set(-0.1, 4.15, 0);
        this.robotGroup.add(finger2);
        
        // Add coordinate frame at tool center point
        const frameSize = 0.4;
        const axisGroup = new THREE.Group();
        
        // X-axis (red)
        const xAxisGeometry = new THREE.CylinderGeometry(0.01, 0.01, frameSize, 8);
        const xAxisMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const xAxis = new THREE.Mesh(xAxisGeometry, xAxisMaterial);
        xAxis.rotation.z = -Math.PI / 2;
        xAxis.position.x = frameSize / 2;
        axisGroup.add(xAxis);
        
        // Y-axis (green)
        const yAxisGeometry = new THREE.CylinderGeometry(0.01, 0.01, frameSize, 8);
        const yAxisMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const yAxis = new THREE.Mesh(yAxisGeometry, yAxisMaterial);
        yAxis.position.y = frameSize / 2;
        axisGroup.add(yAxis);
        
        // Z-axis (blue)
        const zAxisGeometry = new THREE.CylinderGeometry(0.01, 0.01, frameSize, 8);
        const zAxisMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });
        const zAxis = new THREE.Mesh(zAxisGeometry, zAxisMaterial);
        zAxis.rotation.x = Math.PI / 2;
        zAxis.position.z = frameSize / 2;
        axisGroup.add(zAxis);
        
        endEffector.add(axisGroup);
        
        // Add work envelope visualization
        this.addWorkEnvelope();
        
        this.scene.add(this.robotGroup);
    }

    addWorkEnvelope() {
        // Create work envelope visualization
        const envelopeGeometry = new THREE.SphereGeometry(5, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.7);
        const envelopeMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x00B7EB,
            transparent: true,
            opacity: 0.1,
            wireframe: true
        });
        const envelope = new THREE.Mesh(envelopeGeometry, envelopeMaterial);
        envelope.position.y = 2;
        this.scene.add(envelope);
    }

    addCameraControls() {
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        const canvas = this.renderer.domElement;
        
        // Mouse controls for camera
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
            canvas.style.cursor = 'grabbing';
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const deltaMove = {
                x: e.clientX - previousMousePosition.x,
                y: e.clientY - previousMousePosition.y
            };
            
            // Rotate camera around the robot
            const spherical = new THREE.Spherical();
            spherical.setFromVector3(this.camera.position);
            
            spherical.theta -= deltaMove.x * 0.01;
            spherical.phi += deltaMove.y * 0.01;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
            
            this.camera.position.setFromSpherical(spherical);
            this.camera.lookAt(0, 2, 0);
            
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });
        
        canvas.addEventListener('mouseup', () => {
            isDragging = false;
            canvas.style.cursor = 'grab';
        });
        
        canvas.addEventListener('mouseleave', () => {
            isDragging = false;
            canvas.style.cursor = 'grab';
        });
        
        // Zoom with mouse wheel
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const scale = e.deltaY > 0 ? 1.1 : 0.9;
            this.camera.position.multiplyScalar(scale);
            
            // Constrain zoom limits
            const distance = this.camera.position.length();
            if (distance < 3) {
                this.camera.position.normalize().multiplyScalar(3);
            } else if (distance > 20) {
                this.camera.position.normalize().multiplyScalar(20);
            }
        });
    }

    setupUIHandlers() {
        // Training controls
        document.getElementById('start-training')?.addEventListener('click', () => this.startTraining());
        document.getElementById('stop-training')?.addEventListener('click', () => this.stopTraining());
        document.getElementById('reset-system')?.addEventListener('click', () => this.resetSystem());
        
        // Code generation
        document.getElementById('generate-code')?.addEventListener('click', () => this.generateCode());
        
        // Joint controls with enhanced robot animation
        for (let i = 0; i < 6; i++) {
            const slider = document.getElementById(`joint${i}-slider`);
            if (slider) {
                slider.addEventListener('input', (e) => this.handleJointChange(i, e.target.value));
            }
        }
        
        // Log controls
        document.getElementById('clear-log')?.addEventListener('click', () => this.clearLog());
        
        // Window resize
        window.addEventListener('resize', () => this.handleResize());
        
        console.log('Enhanced UI handlers initialized');
    }

    startUpdateLoops() {
        // Update metrics display every 2 seconds
        setInterval(() => {
            this.updateMetricsDisplay();
        }, 2000);
        
        // Update robot animation every 100ms
        setInterval(() => {
            this.updateRobotAnimation();
        }, 100);
        
        // Update coordinates every 200ms
        setInterval(() => {
            this.updateCoordinates();
        }, 200);
        
        // Update robot joint visualization
        setInterval(() => {
            this.updateRobotJoints();
        }, 50);
    }

    startRenderLoop() {
        const animate = () => {
            requestAnimationFrame(animate);
            
            if (this.renderer && this.scene && this.camera) {
                // Add subtle camera sway for dynamic feel
                const time = Date.now() * 0.0005;
                this.camera.position.y += Math.sin(time * 0.5) * 0.01;
                
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
        this.updateJointDisplays();
    }

    startSimulationMode() {
        console.log('Starting enhanced simulation mode...');
        
        // Simulate realistic changing metrics
        setInterval(() => {
            this.metrics.oee = 85 + Math.random() * 15;
            this.metrics.efficiency = 80 + Math.random() * 20;
            this.metrics.cycleTime = 25 + Math.random() * 10;
            this.metrics.throughput = 100 + Math.random() * 50;
            this.metrics.safetyScore = 95 + Math.random() * 5;
            this.metrics.energyEfficiency = 80 + Math.random() * 15;
        }, 3000);
        
        // Simulate robot joint movement
        setInterval(() => {
            if (this.isTraining) {
                for (let i = 0; i < this.jointAngles.length; i++) {
                    this.jointAngles[i] += (Math.random() - 0.5) * 2;
                    this.jointAngles[i] = Math.max(-180, Math.min(180, this.jointAngles[i]));
                }
                this.updateJointDisplays();
            }
        }, 500);
    }

    async startTraining() {
        this.isTraining = true;
        
        document.getElementById('start-training').disabled = true;
        document.getElementById('stop-training').disabled = false;
        
        this.updateTrainingStatus('Running');
        this.addLogEntry('info', 'RL training started - robot motion active');
        
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
        this.addLogEntry('info', 'RL training stopped - robot motion paused');
        
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
            const reward = 1500 + Math.random() * 300 - 150;
            
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
        
        // Reset robot to home position
        this.jointAngles = [0, 0, 0, 0, 0, 0];
        this.updateJointDisplays();
        
        this.addLogEntry('info', 'System reset completed - robot returned to home position');
        
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
                // Simulate code generation
                setTimeout(() => {
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate';
                    downloadBtn.disabled = false;
                    this.addLogEntry('success', 'IEC 61499 PLC code generated successfully');
                }, 2000);
            }
        } catch (error) {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate';
            this.addLogEntry('error', 'Code generation failed: ' + error.message);
        }
    }

    handleJointChange(jointIndex, value) {
        const valueElement = document.getElementById(`joint${jointIndex}-value`);
        if (valueElement) {
            valueElement.textContent = value + '°';
        }
        
        // Update internal joint angle
        this.jointAngles[jointIndex] = parseFloat(value);
        
        // Update 3D robot visualization
        this.updateRobotJoints();
        
        // Send joint update via websocket if connected
        if (this.websocketManager) {
            this.websocketManager.send({
                action: 'update_robot_angles',
                robot_data: {
                    joint_angles: this.jointAngles,
                    timestamp: Date.now()
                }
            });
        }
    }

    updateRobotJoints() {
        if (!this.robotJoints || this.robotJoints.length === 0) return;
        
        // Apply joint rotations to robot model
        try {
            // Base rotation (joint 0)
            if (this.robotJoints[0]) {
                this.robotJoints[0].rotation.y = this.jointAngles[0] * Math.PI / 180;
            }
            
            // Shoulder pitch (joint 1)
            if (this.robotJoints[1]) {
                this.robotJoints[1].rotation.z = this.jointAngles[1] * Math.PI / 180;
            }
            
            // Elbow pitch (joint 2)
            if (this.robotJoints[2]) {
                this.robotJoints[2].rotation.z = this.jointAngles[2] * Math.PI / 180;
            }
            
            // Wrist rotations (joints 3, 4, 5)
            if (this.robotJoints[3]) {
                this.robotJoints[3].rotation.x = this.jointAngles[3] * Math.PI / 180;
            }
            
            if (this.robotJoints[4]) {
                this.robotJoints[4].rotation.z = this.jointAngles[4] * Math.PI / 180;
            }
            
            if (this.robotJoints[5]) {
                this.robotJoints[5].rotation.y = this.jointAngles[5] * Math.PI / 180;
            }
        } catch (error) {
            console.warn('Robot joint update error:', error);
        }
    }

    updateJointDisplays() {
        for (let i = 0; i < this.jointAngles.length; i++) {
            const slider = document.getElementById(`joint${i}-slider`);
            const valueEl = document.getElementById(`joint${i}-value`);
            if (slider && valueEl) {
                slider.value = Math.round(this.jointAngles[i]);
                valueEl.textContent = Math.round(this.jointAngles[i]) + '°';
            }
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
                    statusText.textContent = 'Simulation';
                    break;
                default:
                    statusText.textContent = 'Disconnected';
            }
        }
    }

    updateTrainingStatus(status) {
        const statusEl = document.getElementById('system-training-status');
        if (statusEl) {
            statusEl.textContent = status;
            statusEl.className = 'status-value';
            if (status === 'Running') {
                statusEl.classList.add('active');
            } else {
                statusEl.classList.add('connected');
            }
        }
    }

    updateTrainingMetrics(episode, epsilon, reward) {
        const episodeEl = document.getElementById('current-episode');
        const epsilonEl = document.getElementById('epsilon-value');
        const rewardEl = document.getElementById('current-reward');
        
        if (episodeEl) episodeEl.textContent = episode;
        if (epsilonEl) epsilonEl.textContent = epsilon.toFixed(3);
        if (rewardEl) rewardEl.textContent = reward.toFixed(1);
        
        // Add log entry for significant episodes
        if (episode % 25 === 0) {
            this.addLogEntry('info', `Episode ${episode}: Reward=${reward.toFixed(2)}`);
        }
    }

    updateMetricsDisplay() {
        const elements = {
            'header-oee': this.metrics.oee.toFixed(1) + '%',
            'header-efficiency': this.metrics.efficiency.toFixed(1) + '%',
            'cycle-time': this.metrics.cycleTime.toFixed(1) + 's',
            'throughput': this.metrics.throughput.toFixed(0) + '/min',
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
        if (this.robotGroup && this.isTraining) {
            const time = Date.now() * 0.001;
            // Gentle animation during training
            this.robotGroup.children.forEach((child, index) => {
                if (child.material && child.material.emissive) {
                    const intensity = 0.1 + Math.sin(time + index) * 0.05;
                    child.material.emissive.setScalar(intensity * 0.1);
                }
            });
        }
    }

    updateCoordinates() {
        // Calculate forward kinematics for end effector position
        const time = Date.now() * 0.001;
        const baseRotation = this.jointAngles[0] * Math.PI / 180;
        
        // Simplified forward kinematics calculation
        const x = Math.cos(baseRotation) * (2.5 + Math.sin(time * 0.3) * 0.5);
        const y = 2.0 + Math.sin(time * 0.2) * 0.3;
        const z = Math.sin(baseRotation) * (2.5 + Math.sin(time * 0.3) * 0.5);
        
        const xEl = document.getElementById('robot-x');
        const yEl = document.getElementById('robot-y');
        const zEl = document.getElementById('robot-z');
        
        if (xEl) xEl.textContent = x.toFixed(3);
        if (yEl) yEl.textContent = y.toFixed(3);
        if (zEl) zEl.textContent = z.toFixed(3);
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
            this.addLogEntry('info', 'System log cleared');
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
            
            // Limit log entries
            while (logContent.children.length > 100) {
                logContent.removeChild(logContent.firstChild);
            }
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
    console.log('Starting Enhanced WMG Digital Twin Platform...');
    
    const app = new WMGDigitalTwin();
    window.wmgApp = app; // For debugging and external access
    
    await app.initialize();
});

console.log('WMG Digital Twin Platform Enhanced Main Module loaded');