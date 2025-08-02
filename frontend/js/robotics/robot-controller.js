/**
 * WMG Digital Twin Platform - Advanced Robot Controller
 * University of Warwick - WMG Automation Systems Group
 * 
 * Comprehensive robot control system with trajectory planning, safety monitoring,
 * and real-time motion control for industrial automation applications.
 */

class RobotController {
    constructor(config = {}) {
        this.config = {
            dof: config.dof || 6,
            maxVelocity: config.maxVelocity || [2.0, 2.0, 3.0, 3.0, 3.0, 6.0], // rad/s
            maxAcceleration: config.maxAcceleration || [10.0, 10.0, 15.0, 15.0, 15.0, 20.0], // rad/s²
            jointLimits: config.jointLimits || [
                [-Math.PI, Math.PI],
                [-Math.PI/2, Math.PI/2],
                [-2.35, 0.7],
                [-Math.PI, Math.PI],
                [-2.09, 2.09],
                [-2*Math.PI, 2*Math.PI]
            ],
            safetyZones: config.safetyZones || [],
            controlFrequency: config.controlFrequency || 100, // Hz
            ...config
        };

        // Robot state
        this.currentJointAngles = new Array(this.config.dof).fill(0);
        this.currentJointVelocities = new Array(this.config.dof).fill(0);
        this.targetJointAngles = new Array(this.config.dof).fill(0);
        this.isMoving = false;
        this.isConnected = false;
        this.isEnabled = false;
        this.emergencyStop = false;

        // Motion planning
        this.motionPlanner = new RobotMotionPlanner(this.config);
        this.trajectoryExecutor = new TrajectoryExecutor(this.config);
        this.safetyMonitor = new RobotSafetyMonitor(this.config);

        // Program execution
        this.currentProgram = null;
        this.programQueue = [];
        this.programExecutor = new ProgramExecutor(this);

        // Tool and coordinate systems
        this.toolOffset = { x: 0, y: 0, z: 0.1, rx: 0, ry: 0, rz: 0 };
        this.workCoordinates = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0 };

        // Performance monitoring
        this.performance = {
            cycleTime: 0,
            accuracy: 0,
            repeatability: 0,
            totalMoves: 0,
            faultCount: 0,
            lastFaultTime: null
        };

        // Event dispatcher
        this.eventDispatcher = window.wmgEventDispatcher;

        // Robot visualization reference
        this.robotVisualizer = null;
        this.inverseKinematics = null;

        // Control loop
        this.controlLoopId = null;
        this.lastControlTime = Date.now();

        console.log('Robot Controller initialized with', this.config.dof, 'DOF');
        this._setupEventListeners();
    }

    /**
     * Initialize robot controller with visualization and kinematics
     */
    async initialize(robotVisualizer, inverseKinematics) {
        this.robotVisualizer = robotVisualizer;
        this.inverseKinematics = inverseKinematics;

        // Start control loop
        this._startControlLoop();

        // Initialize safety monitoring
        await this.safetyMonitor.initialize();

        this.eventDispatcher.emit(WMG_EVENTS.ROBOT_CONNECTED, {
            dof: this.config.dof,
            timestamp: Date.now()
        });

        this.isConnected = true;
        console.log('Robot Controller fully initialized');
    }

    /**
     * Enable robot for operation
     */
    enable() {
        if (!this.isConnected) {
            throw new Error('Robot not connected');
        }

        if (this.emergencyStop) {
            throw new Error('Cannot enable robot - emergency stop active');
        }

        this.isEnabled = true;
        this.eventDispatcher.emit(WMG_EVENTS.SYSTEM_STARTUP, {
            system: 'robot',
            timestamp: Date.now()
        });

        console.log('Robot enabled and ready for operation');
    }

    /**
     * Disable robot operation
     */
    disable() {
        this.isEnabled = false;
        this.stopMotion();
        
        this.eventDispatcher.emit(WMG_EVENTS.SYSTEM_SHUTDOWN, {
            system: 'robot',
            timestamp: Date.now()
        });

        console.log('Robot disabled');
    }

    /**
     * Move robot to joint angles
     */
    async moveToJoints(targetAngles, options = {}) {
        if (!this._checkOperationalState()) return false;

        const moveOptions = {
            velocity: options.velocity || 0.5,
            acceleration: options.acceleration || 0.3,
            blending: options.blending || 0.0,
            waitForCompletion: options.waitForCompletion !== false,
            ...options
        };

        // Validate joint angles
        if (!this._validateJointAngles(targetAngles)) {
            throw new Error('Invalid joint angles - exceeding limits');
        }

        // Plan trajectory
        const trajectory = this.motionPlanner.planJointTrajectory(
            this.currentJointAngles,
            targetAngles,
            moveOptions
        );

        // Execute trajectory
        return await this._executeTrajectory(trajectory, moveOptions);
    }

    /**
     * Move robot to Cartesian position
     */
    async moveToPose(targetPose, options = {}) {
        if (!this._checkOperationalState()) return false;

        // Solve inverse kinematics
        const ikSolution = this.inverseKinematics.calculateInverseKinematics(
            [targetPose.x, targetPose.y, targetPose.z],
            [targetPose.rx || 0, targetPose.ry || 0, targetPose.rz || 0],
            this.currentJointAngles
        );

        if (!ikSolution.success) {
            throw new Error('Inverse kinematics solution failed');
        }

        return await this.moveToJoints(ikSolution.jointAngles, options);
    }

    /**
     * Move robot to home position
     */
    async moveToHome() {
        const homePosition = new Array(this.config.dof).fill(0);
        
        this.eventDispatcher.emit(WMG_EVENTS.ROBOT_MOVEMENT_START, {
            operation: 'home',
            timestamp: Date.now()
        });

        const success = await this.moveToJoints(homePosition, {
            velocity: 0.3,
            acceleration: 0.2
        });

        if (success) {
            this.eventDispatcher.emit(WMG_EVENTS.ROBOT_HOME_COMPLETE, {
                timestamp: Date.now()
            });
        }

        return success;
    }

    /**
     * Execute pick operation
     */
    async pick(position, options = {}) {
        if (!this._checkOperationalState()) return false;

        try {
            // Move to approach position
            const approachPos = {
                x: position.x,
                y: position.y,
                z: position.z + (options.approachHeight || 0.1),
                rx: position.rx || 0,
                ry: position.ry || 0,
                rz: position.rz || 0
            };

            await this.moveToPose(approachPos, { velocity: 0.8 });

            // Move to pick position
            await this.moveToPose(position, { velocity: 0.3 });

            // Activate gripper/tool
            await this._activateTool(true);

            // Move back to approach position
            await this.moveToPose(approachPos, { velocity: 0.5 });

            this.eventDispatcher.emit(WMG_EVENTS.ROBOT_MOVEMENT_COMPLETE, {
                operation: 'pick',
                position: position,
                timestamp: Date.now()
            });

            return true;

        } catch (error) {
            console.error('Pick operation failed:', error);
            this.eventDispatcher.emit(WMG_EVENTS.SYSTEM_ERROR, {
                system: 'robot',
                operation: 'pick',
                error: error.message,
                timestamp: Date.now()
            });
            return false;
        }
    }

    /**
     * Execute place operation
     */
    async place(position, options = {}) {
        if (!this._checkOperationalState()) return false;

        try {
            // Move to approach position
            const approachPos = {
                x: position.x,
                y: position.y,
                z: position.z + (options.approachHeight || 0.1),
                rx: position.rx || 0,
                ry: position.ry || 0,
                rz: position.rz || 0
            };

            await this.moveToPose(approachPos, { velocity: 0.8 });

            // Move to place position
            await this.moveToPose(position, { velocity: 0.3 });

            // Deactivate gripper/tool
            await this._activateTool(false);

            // Move back to approach position
            await this.moveToPose(approachPos, { velocity: 0.5 });

            this.eventDispatcher.emit(WMG_EVENTS.ROBOT_MOVEMENT_COMPLETE, {
                operation: 'place',
                position: position,
                timestamp: Date.now()
            });

            return true;

        } catch (error) {
            console.error('Place operation failed:', error);
            this.eventDispatcher.emit(WMG_EVENTS.SYSTEM_ERROR, {
                system: 'robot',
                operation: 'place',
                error: error.message,
                timestamp: Date.now()
            });
            return false;
        }
    }

    /**
     * Stop all robot motion
     */
    stopMotion() {
        this.isMoving = false;
        this.trajectoryExecutor.stop();
        
        this.eventDispatcher.emit(WMG_EVENTS.ROBOT_MOVEMENT_COMPLETE, {
            operation: 'stop',
            forced: true,
            timestamp: Date.now()
        });

        console.log('Robot motion stopped');
    }

    /**
     * Emergency stop
     */
    emergencyStopActivate() {
        this.emergencyStop = true;
        this.isEnabled = false;
        this.stopMotion();

        this.eventDispatcher.emit(WMG_EVENTS.EMERGENCY_STOP, {
            system: 'robot',
            timestamp: Date.now()
        });

        console.warn('Robot emergency stop activated');
    }

    /**
     * Reset emergency stop
     */
    emergencyStopReset() {
        this.emergencyStop = false;
        
        this.eventDispatcher.emit(WMG_EVENTS.SYSTEM_WARNING, {
            system: 'robot',
            message: 'Emergency stop reset - robot requires re-enabling',
            timestamp: Date.now()
        });

        console.log('Robot emergency stop reset');
    }

    /**
     * Load and execute robot program
     */
    async executeProgram(program) {
        if (!this._checkOperationalState()) return false;

        this.currentProgram = program;
        
        this.eventDispatcher.emit(WMG_EVENTS.ROBOT_PROGRAM_START, {
            program: program.name,
            timestamp: Date.now()
        });

        try {
            const result = await this.programExecutor.execute(program);
            
            this.eventDispatcher.emit(WMG_EVENTS.ROBOT_PROGRAM_COMPLETE, {
                program: program.name,
                success: result,
                timestamp: Date.now()
            });

            return result;

        } catch (error) {
            console.error('Program execution failed:', error);
            this.eventDispatcher.emit(WMG_EVENTS.SYSTEM_ERROR, {
                system: 'robot',
                program: program.name,
                error: error.message,
                timestamp: Date.now()
            });
            return false;
        }
    }

    /**
     * Get current robot status
     */
    getStatus() {
        const forwardKinematics = this.inverseKinematics ? 
            this.inverseKinematics.calculateForwardKinematics(this.currentJointAngles) : null;

        return {
            isConnected: this.isConnected,
            isEnabled: this.isEnabled,
            isMoving: this.isMoving,
            emergencyStop: this.emergencyStop,
            currentJointAngles: [...this.currentJointAngles],
            currentJointVelocities: [...this.currentJointVelocities],
            targetJointAngles: [...this.targetJointAngles],
            currentPose: forwardKinematics ? {
                x: forwardKinematics.position[0],
                y: forwardKinematics.position[1],
                z: forwardKinematics.position[2],
                rx: forwardKinematics.orientation[0],
                ry: forwardKinematics.orientation[1],
                rz: forwardKinematics.orientation[2]
            } : null,
            performance: { ...this.performance },
            safetyStatus: this.safetyMonitor.getStatus(),
            currentProgram: this.currentProgram ? this.currentProgram.name : null
        };
    }

    /**
     * Update joint angles (from external control)
     */
    updateJointAngles(angles) {
        if (!Array.isArray(angles) || angles.length !== this.config.dof) {
            console.warn('Invalid joint angles array');
            return;
        }

        // Validate angles against limits
        for (let i = 0; i < angles.length; i++) {
            const [min, max] = this.config.jointLimits[i];
            if (angles[i] < min || angles[i] > max) {
                console.warn(`Joint ${i} angle ${angles[i]} exceeds limits [${min}, ${max}]`);
                angles[i] = Math.max(min, Math.min(max, angles[i]));
            }
        }

        this.targetJointAngles = [...angles];
        
        // Update visualization
        if (this.robotVisualizer) {
            this.robotVisualizer.updateJointAngles(angles);
        }

        this.eventDispatcher.emit(WMG_EVENTS.ROBOT_JOINT_UPDATE, {
            jointAngles: angles,
            timestamp: Date.now()
        });
    }

    // Private methods

    _setupEventListeners() {
        // Emergency stop handling
        this.eventDispatcher.on(WMG_EVENTS.EMERGENCY_STOP, (event) => {
            if (event.data.system !== 'robot') {
                this.emergencyStopActivate();
            }
        });

        // Safety alarm handling
        this.eventDispatcher.on(WMG_EVENTS.SAFETY_ALARM, (event) => {
            this.stopMotion();
        });
    }

    _startControlLoop() {
        const controlLoop = () => {
            if (!this.isConnected) return;

            const currentTime = Date.now();
            const deltaTime = (currentTime - this.lastControlTime) / 1000;
            this.lastControlTime = currentTime;

            // Update robot state
            this._updateRobotState(deltaTime);

            // Check safety conditions
            this._checkSafetyConditions();

            // Update performance metrics
            this._updatePerformanceMetrics();

            // Schedule next iteration
            this.controlLoopId = setTimeout(controlLoop, 1000 / this.config.controlFrequency);
        };

        controlLoop();
    }

    _updateRobotState(deltaTime) {
        // Simulate smooth joint motion towards targets
        for (let i = 0; i < this.config.dof; i++) {
            const error = this.targetJointAngles[i] - this.currentJointAngles[i];
            const maxVel = this.config.maxVelocity[i];
            
            if (Math.abs(error) > 0.001) { // Dead zone
                this.isMoving = true;
                
                // Simple P controller for smooth motion
                const velocity = Math.sign(error) * Math.min(Math.abs(error) * 2.0, maxVel);
                this.currentJointVelocities[i] = velocity;
                
                // Update position
                const deltaAngle = velocity * deltaTime;
                if (Math.abs(deltaAngle) >= Math.abs(error)) {
                    this.currentJointAngles[i] = this.targetJointAngles[i];
                    this.currentJointVelocities[i] = 0;
                } else {
                    this.currentJointAngles[i] += deltaAngle;
                }
            } else {
                this.currentJointVelocities[i] = 0;
            }
        }

        // Check if robot has stopped moving
        const totalVelocity = this.currentJointVelocities.reduce((sum, vel) => sum + Math.abs(vel), 0);
        if (totalVelocity < 0.01 && this.isMoving) {
            this.isMoving = false;
        }
    }

    _checkSafetyConditions() {
        // Check joint limits
        for (let i = 0; i < this.config.dof; i++) {
            const [min, max] = this.config.jointLimits[i];
            if (this.currentJointAngles[i] < min || this.currentJointAngles[i] > max) {
                this._handleSafetyViolation(`Joint ${i} limit exceeded`);
                return;
            }
        }

        // Check velocity limits
        for (let i = 0; i < this.config.dof; i++) {
            if (Math.abs(this.currentJointVelocities[i]) > this.config.maxVelocity[i]) {
                this._handleSafetyViolation(`Joint ${i} velocity limit exceeded`);
                return;
            }
        }

        // Check workspace limits
        if (this.inverseKinematics) {
            const fk = this.inverseKinematics.calculateForwardKinematics(this.currentJointAngles);
            if (fk.singularity) {
                this._handleSafetyViolation('Robot approaching singular configuration');
                return;
            }
        }
    }

    _handleSafetyViolation(message) {
        this.stopMotion();
        this.performance.faultCount++;
        this.performance.lastFaultTime = Date.now();

        this.eventDispatcher.emit(WMG_EVENTS.SAFETY_ALARM, {
            system: 'robot',
            message: message,
            timestamp: Date.now()
        });

        console.warn('Safety violation:', message);
    }

    _updatePerformanceMetrics() {
        this.performance.totalMoves = this.trajectoryExecutor.getTotalMoves();
        
        // Calculate accuracy based on positioning error
        const positionError = this.targetJointAngles.reduce((sum, target, i) => 
            sum + Math.abs(target - this.currentJointAngles[i]), 0
        ) / this.config.dof;
        
        this.performance.accuracy = Math.max(0, 1 - positionError);
    }

    _checkOperationalState() {
        if (!this.isConnected) {
            console.error('Robot not connected');
            return false;
        }

        if (!this.isEnabled) {
            console.error('Robot not enabled');
            return false;
        }

        if (this.emergencyStop) {
            console.error('Emergency stop active');
            return false;
        }

        return true;
    }

    _validateJointAngles(angles) {
        if (!Array.isArray(angles) || angles.length !== this.config.dof) {
            return false;
        }

        return angles.every((angle, i) => {
            const [min, max] = this.config.jointLimits[i];
            return angle >= min && angle <= max;
        });
    }

    async _executeTrajectory(trajectory, options) {
        if (!trajectory || trajectory.length === 0) {
            throw new Error('Invalid trajectory');
        }

        this.isMoving = true;
        
        this.eventDispatcher.emit(WMG_EVENTS.ROBOT_MOVEMENT_START, {
            trajectory: trajectory.length,
            timestamp: Date.now()
        });

        try {
            await this.trajectoryExecutor.execute(trajectory, options);
            
            this.eventDispatcher.emit(WMG_EVENTS.ROBOT_MOVEMENT_COMPLETE, {
                timestamp: Date.now()
            });

            return true;

        } catch (error) {
            this.stopMotion();
            throw error;
        }
    }

    async _activateTool(activate) {
        // Simulate tool activation (gripper, etc.)
        return new Promise(resolve => {
            setTimeout(() => {
                console.log(`Tool ${activate ? 'activated' : 'deactivated'}`);
                resolve(true);
            }, 200);
        });
    }
}

// Supporting classes

class RobotMotionPlanner {
    constructor(config) {
        this.config = config;
    }

    planJointTrajectory(startAngles, endAngles, options) {
        const trajectory = [];
        const steps = Math.ceil((options.velocity || 0.5) * 100); // Simple step calculation
        
        for (let step = 0; step <= steps; step++) {
            const t = step / steps;
            const smoothT = this._smoothStep(t); // S-curve profile
            
            const waypoint = startAngles.map((start, i) => 
                start + (endAngles[i] - start) * smoothT
            );
            
            trajectory.push({
                jointAngles: waypoint,
                timestamp: step * 10, // 10ms intervals
                velocity: this._calculateVelocity(t, options.velocity || 0.5)
            });
        }
        
        return trajectory;
    }

    _smoothStep(t) {
        // S-curve profile for smooth motion
        return t * t * (3 - 2 * t);
    }

    _calculateVelocity(t, maxVel) {
        // Bell curve velocity profile
        const center = 0.5;
        const width = 0.3;
        return maxVel * Math.exp(-Math.pow(t - center, 2) / (2 * Math.pow(width, 2)));
    }
}

class TrajectoryExecutor {
    constructor(config) {
        this.config = config;
        this.totalMoves = 0;
        this.isExecuting = false;
    }

    async execute(trajectory, options) {
        this.isExecuting = true;
        this.totalMoves++;

        for (const waypoint of trajectory) {
            if (!this.isExecuting) break;
            
            // Execute waypoint
            await this._executeWaypoint(waypoint);
            
            // Wait for timing
            await new Promise(resolve => setTimeout(resolve, waypoint.timestamp || 10));
        }

        this.isExecuting = false;
    }

    stop() {
        this.isExecuting = false;
    }

    getTotalMoves() {
        return this.totalMoves;
    }

    async _executeWaypoint(waypoint) {
        // In real implementation, this would send commands to robot controller
        return Promise.resolve();
    }
}

class RobotSafetyMonitor {
    constructor(config) {
        this.config = config;
        this.alarmCount = 0;
        this.lastAlarmTime = null;
    }

    async initialize() {
        console.log('Robot safety monitor initialized');
    }

    getStatus() {
        return {
            alarmCount: this.alarmCount,
            lastAlarmTime: this.lastAlarmTime,
            safetyZones: this.config.safetyZones.length
        };
    }
}

class ProgramExecutor {
    constructor(robotController) {
        this.robot = robotController;
    }

    async execute(program) {
        console.log(`Executing program: ${program.name}`);
        
        // Example program execution
        for (const instruction of program.instructions || []) {
            await this._executeInstruction(instruction);
        }
        
        return true;
    }

    async _executeInstruction(instruction) {
        switch (instruction.type) {
            case 'move_joints':
                return await this.robot.moveToJoints(instruction.angles, instruction.options);
            case 'move_pose':
                return await this.robot.moveToPose(instruction.pose, instruction.options);
            case 'pick':
                return await this.robot.pick(instruction.position, instruction.options);
            case 'place':
                return await this.robot.place(instruction.position, instruction.options);
            case 'wait':
                return await new Promise(resolve => setTimeout(resolve, instruction.duration || 1000));
            default:
                console.warn(`Unknown instruction type: ${instruction.type}`);
                return true;
        }
    }
}

// Export for global use
window.RobotController = RobotController;

console.log('WMG Robot Controller module loaded');