/**
 * WMG RL Digital Twin Platform - Robot Visualization Engine
 * University of Warwick - WMG Automation Systems Group
 * 
 * This module provides advanced 3D visualization of industrial robot arms using
 * Three.js with realistic materials, lighting, and animation capabilities for
 * the digital twin environment.
 */

export class RobotVisualizer {
    constructor(scene, inverseKinematicsEngine) {
        this.scene = scene;
        this.ikEngine = inverseKinematicsEngine;
        this.robots = new Map();
        this.animationGroups = new Map();
        
        // Robot configuration
        this.robotConfigurations = this.getDefaultRobotConfigurations();
        this.materials = this.createRobotMaterials();
        
        // Animation settings
        this.animationSettings = {
            smoothing: 0.15,
            enableInertiaa: true,
            showTrajectories: false,
            showWorkspace: false,
            showJointLimits: true
        };
        
        // Performance optimization
        this.lodEnabled = true;
        this.frustumCulling = true;
        this.shadowOptimization = true;
        
        // Initialize robot cells
        this.initializeRobotCells();
        
        console.log('RobotVisualizer initialized with', this.robots.size, 'robot cells');
    }

    /**
     * Get default robot configurations for the factory setup
     * @returns {Array} Array of robot configuration objects
     */
    getDefaultRobotConfigurations() {
        return [
            {
                id: 'robot_a1',
                position: new THREE.Vector3(-4, 0, 0),
                rotation: new THREE.Euler(0, 0, 0),
                name: 'Industrial Robot Cell A1',
                type: 'articulated_6dof',
                baseColor: 0x2c3e50,
                status: 'active'
            },
            {
                id: 'robot_b2',
                position: new THREE.Vector3(0, 0, 0),
                rotation: new THREE.Euler(0, 0, 0),
                name: 'Industrial Robot Cell B2',
                type: 'articulated_6dof',
                baseColor: 0x34495e,
                status: 'active'
            },
            {
                id: 'robot_c3',
                position: new THREE.Vector3(4, 0, 0),
                rotation: new THREE.Euler(0, 0, 0),
                name: 'Industrial Robot Cell C3',
                type: 'articulated_6dof',
                baseColor: 0x3498db,
                status: 'idle'
            }
        ];
    }

    /**
     * Create sophisticated materials for robot components
     * @returns {Object} Material library
     */
    createRobotMaterials() {
        return {
            // Base materials for different robot components
            robotBase: new THREE.MeshPhongMaterial({
                color: 0x2c3e50,
                shininess: 100,
                transparent: true,
                opacity: 0.95,
                side: THREE.DoubleSide
            }),
            
            robotLink: new THREE.MeshPhongMaterial({
                color: 0x34495e,
                shininess: 120,
                transparent: true,
                opacity: 0.95,
                reflectivity: 0.3
            }),
            
            robotJoint: new THREE.MeshPhongMaterial({
                color: 0x95a5a6,
                shininess: 150,
                transparent: true,
                opacity: 0.9,
                reflectivity: 0.4
            }),
            
            robotTool: new THREE.MeshPhongMaterial({
                color: 0xf39c12,
                shininess: 80,
                transparent: true,
                opacity: 0.9
            }),
            
            // Status indicator materials
            statusActive: new THREE.MeshBasicMaterial({
                color: 0x27ae60,
                transparent: true,
                opacity: 0.8
            }),
            
            statusIdle: new THREE.MeshBasicMaterial({
                color: 0xf39c12,
                transparent: true,
                opacity: 0.8
            }),
            
            statusError: new THREE.MeshBasicMaterial({
                color: 0xe74c3c,
                transparent: true,
                opacity: 0.8
            }),
            
            // Joint limit indicators
            jointLimitPositive: new THREE.MeshBasicMaterial({
                color: 0x27ae60,
                transparent: true,
                opacity: 0.6,
                side: THREE.DoubleSide
            }),
            
            jointLimitNegative: new THREE.MeshBasicMaterial({
                color: 0xe74c3c,
                transparent: true,
                opacity: 0.6,
                side: THREE.DoubleSide
            })
        };
    }

    /**
     * Initialize all robot cells in the factory
     */
    initializeRobotCells() {
        this.robotConfigurations.forEach(config => {
            const robot = this.createIndustrialRobot(config);
            this.robots.set(config.id, robot);
        });
    }

    /**
     * Create a complete industrial robot with advanced visualization
     * @param {Object} config - Robot configuration object
     * @returns {Object} Robot object with 3D components
     */
    createIndustrialRobot(config) {
        const robotGroup = new THREE.Group();
        robotGroup.name = config.id;
        robotGroup.position.copy(config.position);
        robotGroup.rotation.copy(config.rotation);
        
        const robot = {
            id: config.id,
            config: config,
            group: robotGroup,
            bones: [],
            joints: [],
            currentAngles: [0, 0, 0, 0, 0, 0],
            targetAngles: [0, 0, 0, 0, 0, 0],
            animationState: {
                isAnimating: false,
                speed: 1.0,
                startTime: 0
            },
            status: config.status,
            toolFrame: null,
            workspace: null,
            trajectoryPath: null
        };
        
        // Build robot structure
        this.buildRobotStructure(robot);
        
        // Add tool frame visualization
        this.addToolFrame(robot);
        
        // Add joint limit indicators
        if (this.animationSettings.showJointLimits) {
            this.addJointLimitIndicators(robot);
        }
        
        // Add workspace visualization
        if (this.animationSettings.showWorkspace) {
            this.addWorkspaceVisualization(robot);
        }
        
        // Add to scene
        this.scene.add(robotGroup);
        
        return robot;
    }

    /**
     * Build the complete robot structure with all links and joints
     * @param {Object} robot - Robot object to build
     */
    buildRobotStructure(robot) {
        const geometry = this.ikEngine.geometry;
        let parentObject = robot.group;
        let x = 0, y = 0, z = 0;

        for (let i = 0; i < geometry.length; i++) {
            const link = geometry[i];
            const linkAssembly = this.createRobotLink(
                x, y, z, link[0], link[1], link[2], i, robot.config.baseColor
            );

            x += link[0];
            y += link[1];
            z += link[2];

            parentObject.add(linkAssembly);
            parentObject = linkAssembly;
            robot.bones.push(linkAssembly);
        }
    }

    /**
     * Create individual robot link with sophisticated geometry
     * @param {Number} x,y,z - Position coordinates
     * @param {Number} w,h,d - Link dimensions
     * @param {Number} linkIndex - Index of the link
     * @param {Number} baseColor - Base color for the link
     * @returns {THREE.Group} Link assembly group
     */
    createRobotLink(x, y, z, w, h, d, linkIndex, baseColor) {
        const linkGroup = new THREE.Group();
        linkGroup.position.set(x, y, z);

        // Calculate optimal dimensions based on link type
        const dimensions = this.calculateLinkDimensions(w, h, d, linkIndex);
        
        // Create main link geometry
        let linkGeometry;
        if (linkIndex === 0) {
            // Base - cylindrical for realistic appearance
            linkGeometry = new THREE.CylinderGeometry(
                dimensions.width * 0.8, 
                dimensions.width, 
                dimensions.height, 
                20
            );
        } else if (linkIndex === 5) {
            // Tool - specialized end effector geometry
            linkGeometry = new THREE.CylinderGeometry(
                0.05, 0.08, dimensions.height, 12
            );
        } else {
            // Standard links - box geometry with rounded edges
            linkGeometry = new THREE.BoxGeometry(
                dimensions.width, 
                dimensions.height, 
                dimensions.depth
            );
        }

        // Apply appropriate material
        const linkMaterial = this.getLinkMaterial(linkIndex, baseColor);
        const linkMesh = new THREE.Mesh(linkGeometry, linkMaterial);
        
        // Position link appropriately
        linkMesh.position.set(w/2, h/2, d/2);
        linkMesh.castShadow = true;
        linkMesh.receiveShadow = true;
        linkGroup.add(linkMesh);

        // Add joint housing
        this.addJointHousing(linkGroup, linkIndex);
        
        // Add joint axis indicator
        this.addJointAxisIndicator(linkGroup, linkIndex);
        
        // Add cable management (cosmetic detail)
        if (linkIndex > 0 && linkIndex < 5) {
            this.addCableManagement(linkGroup, linkIndex);
        }

        return linkGroup;
    }

    /**
     * Calculate optimal link dimensions based on robot geometry
     * @private
     */
    calculateLinkDimensions(w, h, d, linkIndex) {
        const baseScale = 0.8;
        const thickness = 0.12;
        
        // Scale factors for different link types
        const scaleFactors = [1.6, 1.4, 1.2, 1.0, 0.8, 0.6];
        const scale = scaleFactors[linkIndex] * baseScale;
        
        return {
            width: Math.max(Math.abs(w) + thickness, 0.08) * scale,
            height: Math.max(Math.abs(h) + thickness, 0.08) * scale,
            depth: Math.max(Math.abs(d) + thickness, 0.08) * scale
        };
    }

    /**
     * Get appropriate material for link type
     * @private
     */
    getLinkMaterial(linkIndex, baseColor) {
        const material = this.materials.robotLink.clone();
        
        // Color variation based on link index
        const colorVariations = [
            baseColor,           // Base
            baseColor * 1.1,     // Shoulder
            0x3498db,           // Upper arm - distinctive blue
            0x2980b9,           // Forearm - darker blue
            0xe74c3c,           // Wrist - safety red
            0xf39c12            // Tool - warning orange
        ];
        
        material.color.setHex(colorVariations[linkIndex]);
        return material;
    }

    /**
     * Add sophisticated joint housing with realistic appearance
     * @private
     */
    addJointHousing(parentGroup, linkIndex) {
        const jointRadius = 0.12 + linkIndex * 0.015;
        const jointHeight = 0.20;

        // Main joint housing
        const jointGeometry = new THREE.CylinderGeometry(jointRadius, jointRadius, jointHeight, 24);
        const jointMaterial = this.materials.robotJoint.clone();
        const jointMesh = new THREE.Mesh(jointGeometry, jointMaterial);
        
        jointMesh.castShadow = true;
        jointMesh.receiveShadow = true;
        
        // Configure joint orientation
        this.configureJointOrientation(jointMesh, linkIndex);
        
        parentGroup.add(jointMesh);
        
        // Add joint details (bolts, housing details)
        this.addJointDetails(parentGroup, jointRadius, linkIndex);
    }

    /**
     * Configure joint orientation based on kinematic requirements
     * @private
     */
    configureJointOrientation(jointMesh, linkIndex) {
        const orientations = [
            { axis: 'x', angle: Math.PI / 2 },  // Base
            { axis: 'y', angle: 0 },            // Shoulder
            { axis: 'y', angle: 0 },            // Elbow
            { axis: 'z', angle: Math.PI / 2 },  // Wrist roll
            { axis: 'x', angle: Math.PI / 2 },  // Wrist pitch
            { axis: 'z', angle: 0 }             // Tool rotation
        ];

        const config = orientations[linkIndex];
        jointMesh.rotation[config.axis] = config.angle;
    }

    /**
     * Add realistic joint details (bolts, housing features)
     * @private
     */
    addJointDetails(parentGroup, radius, linkIndex) {
        // Add bolt pattern around joint
        const boltCount = 8;
        const boltRadius = 0.02;
        const boltGeometry = new THREE.CylinderGeometry(boltRadius, boltRadius, 0.05, 8);
        const boltMaterial = new THREE.MeshPhongMaterial({ color: 0x2c3e50 });

        for (let i = 0; i < boltCount; i++) {
            const angle = (i / boltCount) * Math.PI * 2;
            const boltMesh = new THREE.Mesh(boltGeometry, boltMaterial);
            
            boltMesh.position.set(
                Math.cos(angle) * radius * 0.8,
                0,
                Math.sin(angle) * radius * 0.8
            );
            boltMesh.rotation.x = Math.PI / 2;
            boltMesh.castShadow = true;
            
            parentGroup.add(boltMesh);
        }
    }

    /**
     * Add joint axis indicator arrows
     * @private
     */
    addJointAxisIndicator(parentGroup, linkIndex) {
        const axisLength = 0.15;
        const axisConfigurations = [
            { direction: new THREE.Vector3(0, 0, 1), color: 0x0000ff }, // Z-axis
            { direction: new THREE.Vector3(0, 1, 0), color: 0x00ff00 }, // Y-axis
            { direction: new THREE.Vector3(0, 1, 0), color: 0x00ff00 }, // Y-axis
            { direction: new THREE.Vector3(1, 0, 0), color: 0xff0000 }, // X-axis
            { direction: new THREE.Vector3(0, 1, 0), color: 0x00ff00 }, // Y-axis
            { direction: new THREE.Vector3(0, 0, 1), color: 0x0000ff }  // Z-axis
        ];

        const config = axisConfigurations[linkIndex];
        const axisArrow = new THREE.ArrowHelper(
            config.direction, 
            new THREE.Vector3(0, 0, 0), 
            axisLength, 
            config.color,
            axisLength * 0.3,
            axisLength * 0.2
        );
        
        axisArrow.line.material.linewidth = 3;
        parentGroup.add(axisArrow);
    }

    /**
     * Add cable management details for realism
     * @private
     */
    addCableManagement(parentGroup, linkIndex) {
        // Create cable conduit
        const conduitGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.3, 8);
        const conduitMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x2c3e50,
            transparent: true,
            opacity: 0.8
        });
        
        const conduitMesh = new THREE.Mesh(conduitGeometry, conduitMaterial);
        conduitMesh.position.set(0.05, 0, 0);
        conduitMesh.rotation.z = Math.PI / 2;
        conduitMesh.castShadow = true;
        
        parentGroup.add(conduitMesh);
    }

    /**
     * Add tool coordinate frame visualization
     * @private
     */
    addToolFrame(robot) {
        if (robot.bones.length === 0) return;

        const toolBone = robot.bones[robot.bones.length - 1];
        const frameSize = 0.25;
        const frameGroup = new THREE.Group();

        // X-axis (red)
        const xArrow = new THREE.ArrowHelper(
            new THREE.Vector3(1, 0, 0), 
            new THREE.Vector3(0, 0, 0), 
            frameSize, 
            0xff0000,
            frameSize * 0.3,
            frameSize * 0.2
        );

        // Y-axis (green)
        const yArrow = new THREE.ArrowHelper(
            new THREE.Vector3(0, 1, 0), 
            new THREE.Vector3(0, 0, 0), 
            frameSize, 
            0x00ff00,
            frameSize * 0.3,
            frameSize * 0.2
        );

        // Z-axis (blue)
        const zArrow = new THREE.ArrowHelper(
            new THREE.Vector3(0, 0, 1), 
            new THREE.Vector3(0, 0, 0), 
            frameSize, 
            0x0000ff,
            frameSize * 0.3,
            frameSize * 0.2
        );

        frameGroup.add(xArrow);
        frameGroup.add(yArrow);
        frameGroup.add(zArrow);
        
        toolBone.add(frameGroup);
        robot.toolFrame = frameGroup;
    }

    /**
     * Add joint limit indicators
     * @private
     */
    addJointLimitIndicators(robot) {
        const jointLimits = this.ikEngine.jointLimits;
        
        robot.bones.forEach((bone, index) => {
            if (index >= jointLimits.length) return;
            
            const [minLimit, maxLimit] = jointLimits[index];
            const indicatorRadius = 0.15 + index * 0.01;
            const indicatorThickness = 0.02;

            // Positive limit arc (green)
            const positiveGeometry = new THREE.RingGeometry(
                indicatorRadius, 
                indicatorRadius + indicatorThickness, 
                0, 
                Math.abs(maxLimit), 
                32
            );
            
            const positiveMesh = new THREE.Mesh(positiveGeometry, this.materials.jointLimitPositive);
            positiveMesh.rotation.x = -Math.PI / 2;
            bone.add(positiveMesh);

            // Negative limit arc (red)
            const negativeGeometry = new THREE.RingGeometry(
                indicatorRadius + indicatorThickness + 0.01, 
                indicatorRadius + indicatorThickness * 2 + 0.01, 
                0, 
                Math.abs(minLimit), 
                32
            );
            
            const negativeMesh = new THREE.Mesh(negativeGeometry, this.materials.jointLimitNegative);
            negativeMesh.rotation.x = -Math.PI / 2;
            bone.add(negativeMesh);
        });
    }

    /**
     * Add workspace visualization
     * @private
     */
    addWorkspaceVisualization(robot) {
        // Create a simplified workspace representation
        const workspaceGeometry = new THREE.SphereGeometry(2.5, 16, 16);
        const workspaceMaterial = new THREE.MeshBasicMaterial({
            color: 0x3498db,
            transparent: true,
            opacity: 0.1,
            wireframe: true
        });
        
        const workspaceMesh = new THREE.Mesh(workspaceGeometry, workspaceMaterial);
        robot.group.add(workspaceMesh);
        robot.workspace = workspaceMesh;
    }

    /**
     * Update robot joint angles with smooth animation
     * @param {String} robotId - Robot identifier
     * @param {Array} targetAngles - Target joint angles in radians
     */
    setRobotAngles(robotId, targetAngles) {
        const robot = this.robots.get(robotId);
        if (!robot) {
            console.warn(`Robot ${robotId} not found`);
            return;
        }

        robot.targetAngles = [...targetAngles];
        
        if (!robot.animationState.isAnimating) {
            robot.animationState.isAnimating = true;
            this.animateRobotToTarget(robot);
        }
    }

    /**
     * Animate robot smoothly to target angles
     * @private
     */
    animateRobotToTarget(robot) {
        const animate = () => {
            let hasReachedTarget = true;
            
            for (let i = 0; i < robot.currentAngles.length; i++) {
                const diff = robot.targetAngles[i] - robot.currentAngles[i];
                
                if (Math.abs(diff) > 0.001) {
                    robot.currentAngles[i] += diff * this.animationSettings.smoothing;
                    hasReachedTarget = false;
                }
            }
            
            // Apply angles to robot bones
            this.applyJointAngles(robot);
            
            if (hasReachedTarget) {
                robot.animationState.isAnimating = false;
            } else {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }

    /**
     * Apply joint angles to robot bones
     * @private
     */
    applyJointAngles(robot) {
        if (robot.bones.length < 6) return;

        // Apply rotations to each joint
        robot.bones[0].rotation.z = robot.currentAngles[0]; // Base
        robot.bones[1].rotation.y = robot.currentAngles[1]; // Shoulder
        robot.bones[2].rotation.y = robot.currentAngles[2]; // Elbow
        robot.bones[3].rotation.x = robot.currentAngles[3]; // Wrist roll
        robot.bones[4].rotation.y = robot.currentAngles[4]; // Wrist pitch
        robot.bones[5].rotation.z = robot.currentAngles[5]; // Tool rotation
    }

    /**
     * Update robot animations based on RL performance data
     * @param {Object} performanceData - Performance metrics from RL training
     */
    updateRobotAnimations(performanceData) {
        this.robots.forEach(robot => {
            if (robot.status === 'active') {
                this.animateRobotBasedOnPerformance(robot, performanceData);
            }
        });
    }

    /**
     * Create realistic robot motion based on performance metrics
     * @private
     */
    animateRobotBasedOnPerformance(robot, performanceData) {
        const time = Date.now() * 0.001;
        const efficiency = (performanceData.oee || 90) / 100;
        const speed = (performanceData.throughput || 100) / 100;

        // Generate realistic industrial motion patterns
        const motionAngles = [
            Math.sin(time * speed * 0.3) * 1.2 * efficiency,                    // Base rotation
            Math.sin(time * speed * 0.4 + Math.PI/3) * 0.7 + 0.2,               // Shoulder pitch
            Math.sin(time * speed * 0.5 + Math.PI/2) * 0.9 - 0.3,               // Elbow pitch
            Math.sin(time * speed * 0.6 + Math.PI/4) * 1.1,                     // Wrist roll
            Math.cos(time * speed * 0.4 + Math.PI/6) * 0.6,                     // Wrist pitch
            time * speed * 0.25 * efficiency                                     // Tool rotation
        ];

        // Add performance-based variations
        const variationAmplitude = (1 - efficiency) * 0.1;
        motionAngles.forEach((angle, index) => {
            motionAngles[index] += (Math.random() - 0.5) * variationAmplitude;
        });

        this.setRobotAngles(robot.id, motionAngles);
    }

    /**
     * Highlight specific robot joint
     * @param {String} robotId - Robot identifier
     * @param {Number} jointIndex - Joint index to highlight
     * @param {Number} color - Highlight color (optional)
     */
    highlightJoint(robotId, jointIndex, color = 0xff6b35) {
        const robot = this.robots.get(robotId);
        if (!robot || jointIndex >= robot.joints.length) return;

        // Reset all joint colors first
        robot.joints.forEach(joint => {
            joint.material.color.setHex(0x95a5a6);
            joint.material.emissive.setHex(0x000000);
        });

        // Highlight specified joint
        if (robot.joints[jointIndex]) {
            robot.joints[jointIndex].material.color.setHex(color);
            robot.joints[jointIndex].material.emissive.setHex(color * 0.1);
        }
    }

    /**
     * Update robot status and visual indicators
     * @param {String} robotId - Robot identifier
     * @param {String} status - New status ('active', 'idle', 'error')
     */
    updateRobotStatus(robotId, status) {
        const robot = this.robots.get(robotId);
        if (!robot) return;

        robot.status = status;
        
        // Update visual status indicators
        const statusMaterial = this.materials[`status${status.charAt(0).toUpperCase() + status.slice(1)}`];
        
        // Apply status color to robot base
        if (robot.bones[0]) {
            robot.bones[0].children.forEach(child => {
                if (child.material) {
                    child.material.emissive.copy(statusMaterial.color);
                    child.material.emissive.multiplyScalar(0.1);
                }
            });
        }
    }

    /**
     * Get robot pose information
     * @param {String} robotId - Robot identifier
     * @returns {Object} Robot pose data
     */
    getRobotPose(robotId) {
        const robot = this.robots.get(robotId);
        if (!robot) return null;

        const forwardKinematics = this.ikEngine.calculateForwardKinematics(robot.currentAngles);
        
        return {
            jointAngles: [...robot.currentAngles],
            endEffectorPosition: forwardKinematics.position,
            endEffectorOrientation: forwardKinematics.orientation,
            reachable: forwardKinematics.reachable,
            singularity: forwardKinematics.singularity
        };
    }

    /**
     * Enable or disable specific visualization features
     * @param {String} feature - Feature name
     * @param {Boolean} enabled - Whether to enable the feature
     */
    setVisualizationFeature(feature, enabled) {
        this.animationSettings[feature] = enabled;
        
        this.robots.forEach(robot => {
            switch (feature) {
                case 'showJointLimits':
                    // Toggle joint limit visibility
                    robot.bones.forEach(bone => {
                        bone.children.forEach(child => {
                            if (child.geometry instanceof THREE.RingGeometry) {
                                child.visible = enabled;
                            }
                        });
                    });
                    break;
                    
                case 'showWorkspace':
                    if (robot.workspace) {
                        robot.workspace.visible = enabled;
                    }
                    break;
                    
                case 'showTrajectories':
                    if (robot.trajectoryPath) {
                        robot.trajectoryPath.visible = enabled;
                    }
                    break;
            }
        });
    }

    /**
     * Get all robots information
     * @returns {Array} Array of robot information objects
     */
    getAllRobots() {
        const robotsInfo = [];
        
        this.robots.forEach(robot => {
            robotsInfo.push({
                id: robot.id,
                name: robot.config.name,
                status: robot.status,
                currentAngles: [...robot.currentAngles],
                pose: this.getRobotPose(robot.id)
            });
        });
        
        return robotsInfo;
    }

    /**
     * Cleanup and dispose of robot visualizations
     */
    dispose() {
        this.robots.forEach(robot => {
            this.scene.remove(robot.group);
            
            // Dispose of geometries and materials
            robot.group.traverse(child => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(material => material.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            });
        });
        
        this.robots.clear();
        
        // Dispose of materials
        Object.values(this.materials).forEach(material => {
            material.dispose();
        });
        
        console.log('RobotVisualizer disposed');
    }
}

export default RobotVisualizer;