/**
 * WMG Digital Twin Platform - Enhanced Main Application
 * University of Warwick - WMG Automation Systems Group
 * Robot-Focused Digital Twin with Advanced 3D Visualization
 */

class WMGDigitalTwin {
    constructor() {
        this.isInitialized = false;
        this.isTraining = false;
        this.connectionStatus = 'connecting'; // 'connecting' | 'connected' | 'disconnected'
        this.websocketManager = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.robotGroup = null;
        this.robotJoints = [];
        this.animationMixer = null;
        this.codeExport = null; // last reply of POST /api/code/generate

        // Telemetry of the last training episode, as the server reports it; null (shown
        // as "—") until it arrives. Fractions stay fractions here and become % on display.
        this.metrics = {
            cycleCounter: null, // AutomationEnvironment.cycle_time at the end of the episode (steps)
            throughput: null,   // parts per 100 simulation steps
            safety: null,       // safety_score, 0..1
            energy: null,       // energy_efficiency, 0..1
            oee: null,          // 0..1
            reward: null        // total reward of the episode
        };
        this.training = { episodes: null, epsilon: null };
        this.telemetryState = 'none'; // 'none' | 'live' | 'stale' (connection lost since)
        this.staleSince = null;

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
            this.showApplication();
        }
    }

    // The backend serves this page at "/", so talk to the server the page came from;
    // a page opened as a file uses the default backend address.
    serverOrigin() {
        return /^https?:$/.test(window.location.protocol) ? window.location.origin : 'http://localhost:8000';
    }

    async initializeWebSocket() {
        if (typeof WebSocketManager === 'undefined') {
            this.connectionStatus = 'disconnected';
            this.addLogEntry('error', 'websocket-manager.js did not load: no connection to the training server');
            this.updateConnectionStatus();
            return;
        }
        const ws = new WebSocketManager(this.serverOrigin().replace(/^http/, 'ws') + '/ws');
        this.websocketManager = ws;

        // Registered before connecting, so they also serve the manager's own reconnections
        ws.onMessage('system_state', (message) => this.onSystemState(message.data || {}));
        ws.onMessage('training_status', (message) => this.onTrainingStatus(message.status));
        ws.onMessage('training_complete', () => this.onTrainingStatus('completed'));
        ws.onMessage('training_progress', (message) => this.onTrainingProgress(message));
        ws.onMessage('performance_update', (message) => this.onPerformanceUpdate(message.data || {}));
        ws.onMessage('system_reset', () => this.onSystemReset());
        ws.addEventListener('connect', () => {
            this.connectionStatus = 'connected';
            this.updateConnectionStatus();
            this.addLogEntry('success', 'Connected to the training server');
        });
        ws.addEventListener('disconnect', () => {
            const wasConnected = this.connectionStatus === 'connected';
            this.connectionStatus = 'disconnected';
            this.markTelemetryStale();
            this.updateConnectionStatus();
            if (wasConnected) {
                this.addLogEntry('warning', 'Connection to the training server lost; the values shown are stale');
            }
        });

        try {
            const timeout = new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Connection timeout')), 3000));
            await Promise.race([ws.connect(), timeout]);
        } catch (error) {
            console.warn('WebSocket connection failed:', error);
            if (this.connectionStatus !== 'connected') {
                this.connectionStatus = 'disconnected';
                this.addLogEntry('warning', 'Training server not reachable; retrying in the background');
            }
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
            this.createAdvancedRobot();

            // Add camera controls
            this.addCameraControls();

            // Start render loop
            this.startRenderLoop();
            
            console.log('3D scene fully initialized');
            this.addLogEntry('info', '3D robot visualization initialized successfully');
            
        } catch (error) {
            console.error('3D initialization failed:', error);
            this.addLogEntry('error', '3D initialization failed: ' + error.message);
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
        document.getElementById('download-fbt')?.addEventListener('click', () => this.downloadCode('fbt'));
        document.getElementById('download-st')?.addEventListener('click', () => this.downloadCode('st'));

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
        // Telemetry is redrawn when the server sends it (see initializeWebSocket)

        // Update robot animation every 100ms
        setInterval(() => {
            this.updateRobotAnimation();
        }, 100);

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
        this.updateTrainingDisplay();
        this.updateConnectionStatus();
        this.updateJointDisplays();
    }

    // Training runs on the server only: the buttons send the request, and the page
    // shows what the server reports back (training_status, system_reset).
    sendToServer(action, what) {
        if (this.connectionStatus !== 'connected' || !this.websocketManager) {
            this.addLogEntry('warning', `${what}: not connected to the training server`);
            return false;
        }
        this.websocketManager.send({ action });
        return true;
    }

    async startTraining() {
        this.sendToServer('start_training', 'Start');
    }

    async stopTraining() {
        this.sendToServer('stop_training', 'Stop');
    }

    async resetSystem() {
        // The robot pose in the 3D view belongs to this page
        this.jointAngles = [0, 0, 0, 0, 0, 0];
        this.updateJointDisplays();
        this.sendToServer('reset_system', 'Reset');
    }

    onSystemState(state) {
        this.setTrainingRunning(Boolean(state.training_active));
        if (Number.isFinite(state.current_episode)) {
            this.training.episodes = state.current_episode;
        }
        this.updateTrainingDisplay();
    }

    onTrainingStatus(status) {
        const running = status === 'started';
        this.setTrainingRunning(running, status === 'completed' ? 'Completed' : null);
        this.addLogEntry('info', running ? 'RL training started on the server'
            : status === 'completed' ? 'RL training completed on the server' : 'RL training stopped on the server');
    }

    setTrainingRunning(running, label = null) {
        this.isTraining = running;
        document.getElementById('start-training').disabled = running;
        document.getElementById('stop-training').disabled = !running;
        this.updateTrainingStatus(label || (running ? 'Running' : 'Stopped'));
    }

    // training_progress (backend/main.py broadcast_training_progress): "episode" is the
    // 0-based index of the episode that just finished.
    onTrainingProgress(message) {
        this.training = { episodes: message.episode + 1, epsilon: message.epsilon };
        this.metrics.reward = Number.isFinite(message.reward) ? message.reward : null;
        this.setTelemetryLive();
        if (this.training.episodes % 25 === 0 && this.metrics.reward !== null) {
            this.addLogEntry('info', `Episode ${this.training.episodes}: reward ${this.metrics.reward.toFixed(2)}`);
        }
    }

    // performance_update (backend/main.py broadcast_performance_update): snake_case
    // fields of AutomationEnvironment.get_performance_metrics(), mapped one by one.
    onPerformanceUpdate(data) {
        const number = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);
        Object.assign(this.metrics, {
            cycleCounter: number(data.cycle_time),
            throughput: number(data.throughput),
            safety: number(data.safety_score),
            energy: number(data.energy_efficiency),
            oee: number(data.oee)
        });
        this.setTelemetryLive();
    }

    onSystemReset() {
        Object.keys(this.metrics).forEach((key) => { this.metrics[key] = null; });
        this.training = { episodes: 0, epsilon: null };
        this.telemetryState = 'none';
        this.setTrainingRunning(false);
        this.updateTrainingDisplay();
        this.updateMetricsDisplay();
        this.addLogEntry('info', 'Training system reset on the server');
    }

    setTelemetryLive() {
        this.telemetryState = 'live';
        this.staleSince = null;
        this.updateTrainingDisplay();
        this.updateMetricsDisplay();
    }

    markTelemetryStale() {
        if (this.telemetryState === 'live') {
            this.telemetryState = 'stale';
            this.staleSince = new Date();
        }
        this.updateTrainingStatus('Unknown');
        this.updateTrainingDisplay();
        this.updateMetricsDisplay();
    }

    async generateCode() {
        const generateBtn = document.getElementById('generate-code');
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        this.showCodeError(null);

        try {
            this.showCodeExport(await this.requestCodeGeneration());
        } catch (error) {
            const message = 'Code generation failed: ' + error.message;
            this.showCodeError(message);
            this.addLogEntry('error', this.escapeHtml(this.shorten(message, 600)));
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate';
        }
    }

    // POST /api/code/generate exports the trained Q-network as an IEC 61499 Basic FB and checks
    // the FB against the network. It is called over HTTP rather than as the WebSocket action so
    // that the reply, or the reason generation failed, comes back to this request.
    async requestCodeGeneration() {
        let response;
        try {
            response = await fetch(this.serverOrigin() + '/api/code/generate', { method: 'POST' });
        } catch (error) {
            throw new Error(`cannot reach the server at ${this.serverOrigin()}`);
        }
        const body = await response.json().catch(() => ({}));
        if (response.ok) {
            return body;
        }
        if (body.detail === 'Model not trained') {
            throw new Error('the model is not trained yet. Start training, then generate again after a few episodes.');
        }
        throw new Error(body.detail || `the server answered ${response.status} ${response.statusText}`);
    }

    showCodeExport(result) {
        this.codeExport = result;
        const count = (n) => Number(n).toLocaleString('en-GB');
        const check = result.verification;
        const eae = result.eae_compile;
        const files = result.files || {};

        this.setCodeField('code-fb-type', result.fb_type || '-', result.format || '');
        this.setCodeField('code-network', (result.network || []).join('-') || '-',
            result.parameters ? `${count(result.parameters)} parameters` : '');
        this.setCodeField('code-macs', result.macs_per_decision ? count(result.macs_per_decision) : '-',
            'Multiply-accumulates per decision (one REQ event)');

        let summary = 'not checked against the network';
        if (check) {
            // The server only keeps an FB whose check passed: Q-values, argmax and setpoints
            const agreement = `${+(check.argmax_agreement * 100).toFixed(2)}%`;
            const ties = check.near_tie_disagreements
                ? ` (${check.near_tie_disagreements} near-tie differences within ${check.tie_tolerance} excused)` : '';
            const qDiff = check.max_abs_q_diff.toExponential(1);
            const title = `The generated .fbt was run on ${count(check.states)} recent replay-buffer states and `
                + `compared with the network: same action on ${agreement} of them${ties}, largest Q-value difference `
                + `${check.max_abs_q_diff.toExponential(2)} (limit ${check.q_tolerance}), `
                + `${count(check.setpoint_mismatches || 0)} states with setpoints other than the action table's`;
            this.setCodeField('code-check', `${count(check.states)} states`, title, check.ok ? 'ok' : 'failed');
            this.setCodeField('code-agreement', agreement + (check.near_tie_disagreements ? '*' : ''), title);
            this.setCodeField('code-q-diff', qDiff, title);
            summary = `same action as the network on ${agreement} of ${count(check.states)} states${ties}, `
                + `max |ΔQ| ${qDiff}, setpoints as in the action table`;
        } else {
            this.setCodeField('code-check', 'not checked', 'There were no replay-buffer states to check the FB on', 'warn');
            this.setCodeField('code-agreement', '-');
            this.setCodeField('code-q-diff', '-');
        }

        // FBTypeCompiler's verdict on the type only: no EAE solution import, no controller run
        const errors = eae?.errors || [];
        const warnings = eae?.warnings || [];
        const known = eae?.context_messages || [];
        if (eae) {
            const title = eae.ok
                ? `FBTypeCompiler accepted the type: ${count(eae.bin_bytes)} B .bin; ${known.length} known `
                    + 'standalone-compile messages. This is the FB type compiler only: the FB was not imported '
                    + 'into an EAE solution or run on a controller.'
                : `FBTypeCompiler rejected the type (exit code ${eae.exit_code}): ${errors.length} errors, `
                    + `${warnings.length} warnings`;
            this.setCodeField('code-eae', eae.ok ? 'accepted' : 'rejected', title, eae.ok ? 'ok' : 'failed');
            if (eae.ok) summary += `; FBTypeCompiler accepted the type (${known.length} known standalone-compile messages)`;
        } else {
            this.setCodeField('code-eae', 'not run', 'EAE FBTypeCompiler was not found on the server, or eae_check is off');
        }
        this.showCompilerMessages(eae);

        const folder = (files.fbt || '').replace(/[\\/][^\\/]*$/, '');
        const folderElement = document.getElementById('code-folder');
        if (folderElement) folderElement.textContent = folder ? `Saved on the server in ${folder}` : '';
        document.getElementById('code-result')?.classList.remove('hidden');

        const downloads = result.downloads || {};
        document.getElementById('download-fbt').disabled = !downloads.fbt;
        document.getElementById('download-st').disabled = !downloads.st;

        this.addLogEntry('success', this.escapeHtml(`IEC 61499 FB ${result.fb_type} generated: ${summary}`));
        if (eae && !eae.ok) {
            this.addLogEntry('warning', this.escapeHtml(
                `FBTypeCompiler rejected the type: ${errors[0] || warnings[0] || 'exit code ' + eae.exit_code}`));
        }
    }

    // Every message line the compiler printed, in the server's three groups. Lines that
    // echo a whole array initialiser are shortened on screen only; the API keeps them whole.
    showCompilerMessages(eae) {
        const box = document.getElementById('code-eae-messages');
        if (!box) return;
        const groups = eae ? [['errors', eae.errors || []], ['warnings', eae.warnings || []],
            ['known standalone-compile messages', eae.context_messages || []]] : [];
        const total = groups.reduce((n, [, lines]) => n + lines.length, 0);
        box.classList.toggle('hidden', total === 0);
        if (!total) return;
        const limit = 200;
        const shorten = (line) => (line.length > limit
            ? `${line.slice(0, limit)} … [${(line.length - limit).toLocaleString('en-GB')} more characters]` : line);
        document.getElementById('code-eae-messages-title').textContent =
            'Compiler messages: ' + groups.map(([name, lines]) => `${lines.length} ${name}`).join(', ');
        document.getElementById('code-eae-messages-text').textContent = groups
            .filter(([, lines]) => lines.length)
            .map(([name, lines]) => `${name[0].toUpperCase()}${name.slice(1)} (${lines.length}):\n`
                + lines.map(shorten).join('\n'))
            .join('\n\n');
    }

    setCodeField(id, text, title = '', state = '') {
        const element = document.getElementById(id);
        if (!element) return;
        element.textContent = text;
        element.title = title;
        element.classList.remove('ok', 'warn', 'failed');
        if (state) element.classList.add(state);
    }

    showCodeError(message) {
        const element = document.getElementById('code-error');
        if (!element) return;
        const text = message || '';
        element.textContent = this.shorten(text, 320); // a failed check carries the whole report
        element.title = text;
        element.classList.toggle('hidden', !text);
    }

    shorten(text, limit) {
        return text.length > limit ? `${text.slice(0, limit)} …` : text;
    }

    // Save one of the generated files from the server's download URL (GET /api/code/files/...).
    downloadCode(kind) {
        const url = this.codeExport?.downloads?.[kind];
        if (!url) return;
        const name = decodeURIComponent(url.split('/').pop());
        this.saveFile(this.serverOrigin() + url, name);
        this.addLogEntry('info', this.escapeHtml(`Downloading ${name}`));
    }

    saveFile(href, name) {
        const link = document.createElement('a');
        link.href = href;
        link.download = name;
        document.body.appendChild(link);
        link.click();
        link.remove();
    }

    escapeHtml(text) {
        const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
        return String(text).replace(/[&<>"']/g, (c) => entities[c]);
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
        const connected = this.connectionStatus === 'connected';
        const text = connected ? 'Connected' : this.connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected';
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        if (statusDot) statusDot.className = 'status-dot' + (connected ? ' connected' : '');
        if (statusText) statusText.textContent = text;
        const panelStatus = document.getElementById('system-connection-status');
        if (panelStatus) {
            panelStatus.textContent = text;
            panelStatus.className = 'status-value' + (connected ? ' connected' : ' disconnected');
        }
    }

    updateTrainingStatus(status) {
        const statusEl = document.getElementById('system-training-status');
        if (statusEl) {
            statusEl.textContent = status;
            statusEl.className = 'status-value' + (status === 'Running' ? ' active' : status === 'Unknown' ? '' : ' connected');
        }
    }

    updateTrainingDisplay() {
        const { episodes, epsilon } = this.training;
        this.setTelemetryText('current-episode', episodes == null ? '—' : String(episodes));
        this.setTelemetryText('epsilon-value', epsilon == null ? '—' : epsilon.toFixed(3));
    }

    updateMetricsDisplay() {
        const m = this.metrics;
        const percent = (value, digits) => (value == null ? '—' : `${(value * 100).toFixed(digits)}%`);
        const fixed = (value, digits, unit = '') => (value == null ? '—' : `${value.toFixed(digits)}${unit}`);
        const values = {
            'header-oee': percent(m.oee, 1),
            'header-energy': percent(m.energy, 0),
            'cycle-time': fixed(m.cycleCounter, 1),
            'throughput': fixed(m.throughput, 1, ' /100 steps'),
            'safety-score': percent(m.safety, 0),
            'energy-efficiency': percent(m.energy, 0),
            'oee-value': percent(m.oee, 1),
            'current-reward': fixed(m.reward, 1)
        };
        Object.entries(values).forEach(([id, text]) => this.setTelemetryText(id, text));

        const badge = document.getElementById('telemetry-state');
        if (badge) {
            badge.textContent = { none: 'no data', live: 'live', stale: 'stale' }[this.telemetryState];
            badge.className = `telemetry-state ${this.telemetryState}`;
            badge.title = this.telemetryState === 'stale'
                ? `Connection lost at ${this.staleSince.toLocaleTimeString()}: these values are from before`
                : this.telemetryState === 'live' ? 'Last training episode, from the server'
                    : 'No training episode reported yet';
        }
    }

    // Writes one telemetry value; stale values are dimmed until fresh data arrives
    setTelemetryText(id, text) {
        const element = document.getElementById(id);
        if (!element) return;
        element.textContent = text;
        element.classList.toggle('stale', this.telemetryState === 'stale');
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