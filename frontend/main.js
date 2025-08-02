/**
 * WMG Digital Twin Frontend - Robot Visualization and Control
 * University of Warwick - WMG Automation Systems Group
 */

class RobotVisualization {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.robotGroup = null;
        this.joints = [];
        this.jointAngles = [0, 0, 0, 0, 0, 0];
        
        this.initialized = false;
        this.animationId = null;
    }

    async init() {
        if (this.initialized) return;
        
        try {
            await this.setupThreeJS();
            await this.createRobotModel();
            this.setupEventListeners();
            this.startRenderLoop();
            
            this.initialized = true;
            console.log('Robot visualization initialized successfully');
        } catch (error) {
            console.error('Failed to initialize robot visualization:', error);
        }
    }

    async setupThreeJS() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);

        // Renderer setup
        const canvas = document.getElementById('robot-canvas');
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);

        // Grid helper
        const gridHelper = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);

        // Axes helper
        const axesHelper = new THREE.AxesHelper(2);
        this.scene.add(axesHelper);
    }

    async createRobotModel() {
        this.robotGroup = new THREE.Group();
        this.scene.add(this.robotGroup);

        // Robot geometry based on industrial 6-DOF robot
        const linkGeometries = [
            { length: 0.6, radius: 0.15, color: 0x2d5a87 },  // Base
            { length: 0.5, radius: 0.12, color: 0x4a7ba7 },  // Shoulder
            { length: 1.2, radius: 0.08, color: 0x2d5a87 },  // Upper arm
            { length: 0.95, radius: 0.06, color: 0x4a7ba7 }, // Forearm
            { length: 0.25, radius: 0.05, color: 0x2d5a87 }, // Wrist
            { length: 0.12, radius: 0.04, color: 0x00ff88 }  // Tool
        ];

        this.joints = [];

        for (let i = 0; i < 6; i++) {
            const joint = new THREE.Group();
            
            // Create link geometry
            const geometry = new THREE.CylinderGeometry(
                linkGeometries[i].radius,
                linkGeometries[i].radius,
                linkGeometries[i].length,
                16
            );
            
            const material = new THREE.MeshPhongMaterial({
                color: linkGeometries[i].color,
                shininess: 100
            });
            
            const link = new THREE.Mesh(geometry, material);
            link.castShadow = true;
            link.receiveShadow = true;
            
            // Position link relative to joint
            link.position.y = linkGeometries[i].length / 2;
            joint.add(link);

            // Add joint indicator
            const jointGeometry = new THREE.SphereGeometry(linkGeometries[i].radius * 1.2, 16, 16);
            const jointMaterial = new THREE.MeshPhongMaterial({
                color: 0x333333,
                shininess: 50
            });
            const jointIndicator = new THREE.Mesh(jointGeometry, jointMaterial);
            joint.add(jointIndicator);

            this.joints.push(joint);
            
            // Create hierarchy: each joint is child of previous joint's end
            if (i === 0) {
                this.robotGroup.add(joint);
            } else {
                // Position at end of previous link
                joint.position.y = linkGeometries[i-1].length;
                this.joints[i-1].add(joint);
            }
        }

        // Position robot group
        this.robotGroup.position.y = 0.1;
    }

    setupEventListeners() {
        // Window resize handler
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Mouse controls (basic orbit)
        let isMouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let cameraRadius = 8;
        let cameraTheta = 0;
        let cameraPhi = Math.PI / 4;

        const canvas = this.renderer.domElement;

        canvas.addEventListener('mousedown', (event) => {
            isMouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });

        canvas.addEventListener('mousemove', (event) => {
            if (!isMouseDown) return;

            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;

            cameraTheta += deltaX * 0.01;
            cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, cameraPhi + deltaY * 0.01));

            mouseX = event.clientX;
            mouseY = event.clientY;

            this.updateCameraPosition(cameraRadius, cameraTheta, cameraPhi);
        });

        canvas.addEventListener('mouseup', () => {
            isMouseDown = false;
        });

        canvas.addEventListener('wheel', (event) => {
            event.preventDefault();
            cameraRadius = Math.max(2, Math.min(20, cameraRadius + event.deltaY * 0.01));
            this.updateCameraPosition(cameraRadius, cameraTheta, cameraPhi);
        });
    }

    updateCameraPosition(radius, theta, phi) {
        this.camera.position.x = radius * Math.sin(phi) * Math.cos(theta);
        this.camera.position.y = radius * Math.cos(phi);
        this.camera.position.z = radius * Math.sin(phi) * Math.sin(theta);
        this.camera.lookAt(0, 2, 0);
    }

    updateJointAngles(angles) {
        if (!this.joints || this.joints.length === 0) return;

        this.jointAngles = angles;

        // Apply rotations to joints
        for (let i = 0; i < Math.min(angles.length, this.joints.length); i++) {
            if (this.joints[i]) {
                // Different rotation axes for different joints
                switch (i) {
                    case 0: // Base rotation (Y-axis)
                        this.joints[i].rotation.y = angles[i];
                        break;
                    case 1: // Shoulder pitch (Z-axis)
                        this.joints[i].rotation.z = angles[i];
                        break;
                    case 2: // Elbow pitch (Z-axis)
                        this.joints[i].rotation.z = angles[i];
                        break;
                    case 3: // Wrist roll (X-axis)
                        this.joints[i].rotation.x = angles[i];
                        break;
                    case 4: // Wrist pitch (Z-axis)
                        this.joints[i].rotation.z = angles[i];
                        break;
                    case 5: // Tool rotation (X-axis)
                        this.joints[i].rotation.x = angles[i];
                        break;
                }
            }
        }
    }

    startRenderLoop() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            // Add subtle animation to make the scene feel alive
            if (this.robotGroup) {
                this.robotGroup.rotation.y += 0.001;
            }
            
            this.renderer.render(this.scene, this.camera);
        };
        animate();
    }

    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
}

// Global robot visualization instance
let robotViz = null;

// Initialize robot visualization when called from HTML
window.initRobotVisualization = function() {
    if (!robotViz) {
        robotViz = new RobotVisualization();
        robotViz.init().then(() => {
            console.log('Robot visualization ready');
        }).catch(error => {
            console.error('Robot visualization failed:', error);
        });
    }
};

// Update robot from external controls
window.updateRobotJoints = function(jointAngles) {
    if (robotViz && robotViz.initialized) {
        robotViz.updateJointAngles(jointAngles);
    }
};

// Export for module loading
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        init: window.initRobotVisualization,
        updateJoints: window.updateRobotJoints,
        RobotVisualization
    };
}

// Auto-initialize if loaded directly
if (typeof window !== 'undefined' && window.document) {
    document.addEventListener('DOMContentLoaded', () => {
        // Wait a bit for other scripts to load
        setTimeout(() => {
            if (typeof window.initRobotVisualization === 'function') {
                window.initRobotVisualization();
            }
        }, 500);
    });
}

console.log('WMG Robot Visualization module loaded');