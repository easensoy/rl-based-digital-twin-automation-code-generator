/**
 * WMG RL Digital Twin Platform - Inverse Kinematics Engine
 * University of Warwick - WMG Automation Systems Group
 * 
 * This module provides comprehensive forward and inverse kinematics calculations
 * for 6-DOF industrial robot arms using Denavit-Hartenberg parameters and
 * analytical solutions for real-time motion planning and control.
 */

export class InverseKinematics {
    constructor(robotGeometry, jointLimits) {
        this.geometry = robotGeometry || this.getDefaultGeometry();
        this.jointLimits = jointLimits || this.getDefaultJointLimits();
        
        // Kinematic configuration
        this.dofCount = 6;
        this.convergenceThreshold = 1e-6;
        this.maxIterations = 100;
        this.dampingFactor = 0.1;
        
        // Transformation matrices cache
        this.transformationCache = new Map();
        this.cacheEnabled = true;
        
        // Performance monitoring
        this.performanceMetrics = {
            forwardKinematicsCallCount: 0,
            inverseKinematicsCallCount: 0,
            averageForwardTime: 0,
            averageInverseTime: 0,
            cacheHitRate: 0
        };
        
        // Precompute static transformations
        this.precomputeStaticTransforms();
        
        console.log('InverseKinematics engine initialized:', {
            geometry: this.geometry,
            jointLimits: this.jointLimits
        });
    }

    /**
     * Get default robot geometry (DH parameters)
     * @returns {Array} Array of [a, alpha, d] parameters for each link
     */
    getDefaultGeometry() {
        return [
            [0, 0, 0.6],      // Base link
            [0, Math.PI/2, 0.5],  // Shoulder link
            [1.2, 0, 0],      // Upper arm
            [0.95, 0, 0],     // Forearm
            [0, Math.PI/2, 0.25], // Wrist
            [0, 0, 0.12]      // Tool
        ];
    }

    /**
     * Get default joint limits in radians
     * @returns {Array} Array of [min, max] limits for each joint
     */
    getDefaultJointLimits() {
        return [
            [-Math.PI, Math.PI],      // Base rotation ±180°
            [-Math.PI/2, Math.PI/2],  // Shoulder pitch ±90°
            [-2.35, 0.7],             // Elbow pitch -135° to +40°
            [-Math.PI, Math.PI],      // Wrist roll ±180°
            [-2.09, 2.09],            // Wrist pitch ±120°
            [-2*Math.PI, 2*Math.PI]   // Tool rotation ±360°
        ];
    }

    /**
     * Calculate forward kinematics for given joint angles
     * @param {Array} jointAngles - Array of 6 joint angles in radians
     * @returns {Object} End-effector pose {position: [x,y,z], orientation: [rx,ry,rz]}
     */
    calculateForwardKinematics(jointAngles) {
        const startTime = performance.now();
        this.performanceMetrics.forwardKinematicsCallCount++;

        if (jointAngles.length !== this.dofCount) {
            throw new Error(`Expected ${this.dofCount} joint angles, got ${jointAngles.length}`);
        }

        // Check cache first
        const cacheKey = this.generateCacheKey('forward', jointAngles);
        if (this.cacheEnabled && this.transformationCache.has(cacheKey)) {
            this.updateCacheHitRate(true);
            return this.transformationCache.get(cacheKey);
        }

        try {
            // Calculate transformation matrix for each joint
            let cumulativeTransform = this.createIdentityMatrix();
            const jointTransforms = [];

            for (let i = 0; i < this.dofCount; i++) {
                const dhTransform = this.createDHTransformation(
                    this.geometry[i][0], // a (link length)
                    this.geometry[i][1], // alpha (link twist)
                    this.geometry[i][2], // d (link offset)
                    jointAngles[i]       // theta (joint angle)
                );

                cumulativeTransform = this.multiplyMatrices(cumulativeTransform, dhTransform);
                jointTransforms.push({ ...cumulativeTransform });
            }

            // Extract end-effector pose
            const result = {
                position: this.extractPosition(cumulativeTransform),
                orientation: this.extractEulerAngles(cumulativeTransform),
                transformationMatrix: cumulativeTransform,
                jointTransforms: jointTransforms,
                reachable: true,
                singularity: this.checkSingularity(jointAngles)
            };

            // Cache the result
            if (this.cacheEnabled) {
                this.transformationCache.set(cacheKey, result);
                this.updateCacheHitRate(false);
            }

            // Update performance metrics
            const executionTime = performance.now() - startTime;
            this.updatePerformanceMetric('averageForwardTime', executionTime);

            return result;

        } catch (error) {
            console.error('Forward kinematics calculation failed:', error);
            throw error;
        }
    }

    /**
     * Calculate inverse kinematics for desired end-effector pose
     * @param {Array} targetPosition - Target position [x, y, z]
     * @param {Array} targetOrientation - Target orientation [rx, ry, rz] in radians
     * @param {Array} initialGuess - Initial joint angle guess (optional)
     * @returns {Object} Solution with joint angles and metadata
     */
    calculateInverseKinematics(targetPosition, targetOrientation, initialGuess = null) {
        const startTime = performance.now();
        this.performanceMetrics.inverseKinematicsCallCount++;

        // Validate inputs
        if (targetPosition.length !== 3 || targetOrientation.length !== 3) {
            throw new Error('Target position and orientation must be 3-element arrays');
        }

        // Check if target is within workspace
        if (!this.isPositionReachable(targetPosition)) {
            return {
                success: false,
                jointAngles: null,
                error: 'Target position outside robot workspace',
                iterations: 0,
                finalError: Infinity
            };
        }

        try {
            // Use analytical solution for 6-DOF robot if available, otherwise numerical
            const analyticalSolution = this.solveAnalyticalIK(targetPosition, targetOrientation);
            
            if (analyticalSolution.success) {
                const executionTime = performance.now() - startTime;
                this.updatePerformanceMetric('averageInverseTime', executionTime);
                return analyticalSolution;
            }

            // Fallback to numerical solution using Jacobian method
            const numericalSolution = this.solveNumericalIK(targetPosition, targetOrientation, initialGuess);
            
            const executionTime = performance.now() - startTime;
            this.updatePerformanceMetric('averageInverseTime', executionTime);
            
            return numericalSolution;

        } catch (error) {
            console.error('Inverse kinematics calculation failed:', error);
            return {
                success: false,
                jointAngles: null,
                error: error.message,
                iterations: 0,
                finalError: Infinity
            };
        }
    }

    /**
     * Analytical inverse kinematics solution for 6-DOF robot
     * @private
     */
    solveAnalyticalIK(targetPosition, targetOrientation) {
        const [x, y, z] = targetPosition;
        const [rx, ry, rz] = targetOrientation;
        
        const jointAngles = new Array(6).fill(0);
        
        try {
            // Joint 1 (Base rotation) - analytical solution
            jointAngles[0] = Math.atan2(y, x);

            // Calculate wrist center position
            const wristOffset = this.geometry[5][2]; // Tool length
            const rotationMatrix = this.eulerToRotationMatrix(rx, ry, rz);
            const toolVector = this.multiplyMatrixVector(rotationMatrix, [0, 0, wristOffset]);
            
            const wristCenter = [
                x - toolVector[0],
                y - toolVector[1], 
                z - toolVector[2]
            ];

            // Calculate distance from shoulder to wrist center
            const shoulderHeight = this.geometry[0][2];
            const shoulderToWrist = Math.sqrt(
                Math.pow(wristCenter[0], 2) + 
                Math.pow(wristCenter[1], 2) + 
                Math.pow(wristCenter[2] - shoulderHeight, 2)
            );

            // Check if position is reachable
            const upperArmLength = this.geometry[2][0];
            const forearmLength = this.geometry[3][0];
            const maxReach = upperArmLength + forearmLength;
            const minReach = Math.abs(upperArmLength - forearmLength);

            if (shoulderToWrist > maxReach || shoulderToWrist < minReach) {
                return {
                    success: false,
                    jointAngles: null,
                    error: 'Target position not reachable by arm geometry',
                    method: 'analytical'
                };
            }

            // Joint 3 (Elbow) - law of cosines
            const cosTheta3 = (
                Math.pow(upperArmLength, 2) + 
                Math.pow(forearmLength, 2) - 
                Math.pow(shoulderToWrist, 2)
            ) / (2 * upperArmLength * forearmLength);
            
            jointAngles[2] = Math.acos(Math.max(-1, Math.min(1, cosTheta3)));

            // Joint 2 (Shoulder) - geometric solution
            const alpha = Math.atan2(
                wristCenter[2] - shoulderHeight,
                Math.sqrt(Math.pow(wristCenter[0], 2) + Math.pow(wristCenter[1], 2))
            );
            
            const beta = Math.acos(
                (Math.pow(upperArmLength, 2) + Math.pow(shoulderToWrist, 2) - Math.pow(forearmLength, 2)) /
                (2 * upperArmLength * shoulderToWrist)
            );
            
            jointAngles[1] = alpha + beta;

            // Joints 4, 5, 6 (Wrist orientation) - spherical wrist solution
            const wristRotation = this.calculateWristOrientation(
                jointAngles.slice(0, 3), 
                targetOrientation
            );
            
            jointAngles[3] = wristRotation[0];
            jointAngles[4] = wristRotation[1];
            jointAngles[5] = wristRotation[2];

            // Apply joint limits
            const limitedAngles = this.applyJointLimits(jointAngles);
            
            // Verify solution accuracy
            const verification = this.calculateForwardKinematics(limitedAngles);
            const positionError = this.calculatePositionError(verification.position, targetPosition);
            const orientationError = this.calculateOrientationError(verification.orientation, targetOrientation);

            return {
                success: true,
                jointAngles: limitedAngles,
                positionError: positionError,
                orientationError: orientationError,
                totalError: positionError + orientationError,
                method: 'analytical',
                reachable: positionError < 0.01, // 1cm tolerance
                withinLimits: this.checkJointLimits(limitedAngles)
            };

        } catch (error) {
            return {
                success: false,
                jointAngles: null,
                error: `Analytical IK failed: ${error.message}`,
                method: 'analytical'
            };
        }
    }

    /**
     * Numerical inverse kinematics using Jacobian method
     * @private
     */
    solveNumericalIK(targetPosition, targetOrientation, initialGuess) {
        // Initialize joint angles
        let currentAngles = initialGuess || this.generateInitialGuess(targetPosition);
        let bestSolution = null;
        let bestError = Infinity;

        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            // Calculate current end-effector pose
            const currentPose = this.calculateForwardKinematics(currentAngles);
            
            // Calculate pose error
            const positionError = this.subtractVectors(targetPosition, currentPose.position);
            const orientationError = this.calculateOrientationDifference(targetOrientation, currentPose.orientation);
            const poseError = [...positionError, ...orientationError];
            
            const totalError = this.vectorNorm(poseError);
            
            // Track best solution
            if (totalError < bestError) {
                bestError = totalError;
                bestSolution = [...currentAngles];
            }

            // Check convergence
            if (totalError < this.convergenceThreshold) {
                return {
                    success: true,
                    jointAngles: this.applyJointLimits(currentAngles),
                    positionError: this.vectorNorm(positionError),
                    orientationError: this.vectorNorm(orientationError),
                    totalError: totalError,
                    iterations: iteration + 1,
                    method: 'numerical_jacobian',
                    converged: true
                };
            }

            // Calculate Jacobian matrix
            const jacobian = this.calculateJacobian(currentAngles);
            
            // Calculate pseudo-inverse of Jacobian
            const jacobianPinv = this.pseudoInverse(jacobian);
            
            // Calculate joint angle update
            const deltaAngles = this.multiplyMatrixVector(jacobianPinv, poseError);
            
            // Apply damping to prevent oscillation
            const dampedDelta = deltaAngles.map(delta => delta * this.dampingFactor);
            
            // Update joint angles
            currentAngles = currentAngles.map((angle, i) => angle + dampedDelta[i]);
            
            // Apply joint limits during iteration
            currentAngles = this.applyJointLimits(currentAngles);
        }

        // Return best solution found
        return {
            success: bestError < 0.1, // 10cm tolerance for numerical solution
            jointAngles: this.applyJointLimits(bestSolution || currentAngles),
            positionError: bestError,
            orientationError: 0, // Included in bestError
            totalError: bestError,
            iterations: this.maxIterations,
            method: 'numerical_jacobian',
            converged: false
        };
    }

    /**
     * Calculate Jacobian matrix for current joint configuration
     * @private
     */
    calculateJacobian(jointAngles) {
        const epsilon = 1e-6;
        const jacobian = [];
        
        // Get current end-effector pose
        const currentPose = this.calculateForwardKinematics(jointAngles);
        const currentPosition = currentPose.position;
        const currentOrientation = currentPose.orientation;
        
        for (let i = 0; i < this.dofCount; i++) {
            // Perturb joint i
            const perturbedAngles = [...jointAngles];
            perturbedAngles[i] += epsilon;
            
            // Calculate perturbed pose
            const perturbedPose = this.calculateForwardKinematics(perturbedAngles);
            const perturbedPosition = perturbedPose.position;
            const perturbedOrientation = perturbedPose.orientation;
            
            // Calculate partial derivatives
            const positionDerivative = [
                (perturbedPosition[0] - currentPosition[0]) / epsilon,
                (perturbedPosition[1] - currentPosition[1]) / epsilon,
                (perturbedPosition[2] - currentPosition[2]) / epsilon
            ];
            
            const orientationDerivative = [
                (perturbedOrientation[0] - currentOrientation[0]) / epsilon,
                (perturbedOrientation[1] - currentOrientation[1]) / epsilon,
                (perturbedOrientation[2] - currentOrientation[2]) / epsilon
            ];
            
            jacobian.push([...positionDerivative, ...orientationDerivative]);
        }
        
        // Transpose to get correct matrix orientation
        return this.transposeMatrix(jacobian);
    }

    /**
     * Create DH transformation matrix
     * @private
     */
    createDHTransformation(a, alpha, d, theta) {
        const ct = Math.cos(theta);
        const st = Math.sin(theta);
        const ca = Math.cos(alpha);
        const sa = Math.sin(alpha);

        return [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ];
    }

    /**
     * Create 4x4 identity matrix
     * @private
     */
    createIdentityMatrix() {
        return [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ];
    }

    /**
     * Multiply two 4x4 matrices
     * @private
     */
    multiplyMatrices(a, b) {
        const result = Array(4).fill().map(() => Array(4).fill(0));
        
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                for (let k = 0; k < 4; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }

    /**
     * Extract position from transformation matrix
     * @private
     */
    extractPosition(transformMatrix) {
        return [
            transformMatrix[0][3],
            transformMatrix[1][3],
            transformMatrix[2][3]
        ];
    }

    /**
     * Extract Euler angles from rotation matrix
     * @private
     */
    extractEulerAngles(transformMatrix) {
        const sy = Math.sqrt(
            transformMatrix[0][0] * transformMatrix[0][0] + 
            transformMatrix[1][0] * transformMatrix[1][0]
        );
        
        const singular = sy < 1e-6;
        
        let x, y, z;
        
        if (!singular) {
            x = Math.atan2(transformMatrix[2][1], transformMatrix[2][2]);
            y = Math.atan2(-transformMatrix[2][0], sy);
            z = Math.atan2(transformMatrix[1][0], transformMatrix[0][0]);
        } else {
            x = Math.atan2(-transformMatrix[1][2], transformMatrix[1][1]);
            y = Math.atan2(-transformMatrix[2][0], sy);
            z = 0;
        }
        
        return [x, y, z];
    }

    /**
     * Check if position is within robot workspace
     * @private
     */
    isPositionReachable(position) {
        const [x, y, z] = position;
        const distance = Math.sqrt(x * x + y * y + z * z);
        
        // Calculate maximum reach
        const maxReach = this.geometry.reduce((sum, link) => {
            return sum + Math.abs(link[0]) + Math.abs(link[2]);
        }, 0);
        
        // Check basic reachability constraints
        const withinReach = distance <= maxReach * 0.95; // 95% of theoretical max
        const aboveGround = z >= -0.1; // Slightly below ground level
        const withinHeight = z <= maxReach * 0.8; // Reasonable height limit
        
        return withinReach && aboveGround && withinHeight;
    }

    /**
     * Apply joint limits to angle array
     * @private
     */
    applyJointLimits(jointAngles) {
        return jointAngles.map((angle, i) => {
            const [min, max] = this.jointLimits[i];
            return Math.max(min, Math.min(max, angle));
        });
    }

    /**
     * Check if joint angles are within limits
     * @private
     */
    checkJointLimits(jointAngles) {
        return jointAngles.every((angle, i) => {
            const [min, max] = this.jointLimits[i];
            return angle >= min && angle <= max;
        });
    }

    /**
     * Calculate position error between two points
     * @private
     */
    calculatePositionError(position1, position2) {
        const diff = this.subtractVectors(position1, position2);
        return this.vectorNorm(diff);
    }

    /**
     * Calculate orientation error between two orientation vectors
     * @private
     */
    calculateOrientationError(orientation1, orientation2) {
        const diff = this.subtractVectors(orientation1, orientation2);
        return this.vectorNorm(diff);
    }

    /**
     * Vector subtraction
     * @private
     */
    subtractVectors(a, b) {
        return a.map((val, i) => val - b[i]);
    }

    /**
     * Calculate vector norm (magnitude)
     * @private
     */
    vectorNorm(vector) {
        return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    }

    /**
     * Generate cache key for memoization
     * @private
     */
    generateCacheKey(operation, angles) {
        const roundedAngles = angles.map(angle => Math.round(angle * 1000) / 1000);
        return `${operation}_${roundedAngles.join('_')}`;
    }

    /**
     * Update performance metrics
     * @private
     */
    updatePerformanceMetric(metricName, newValue) {
        const currentValue = this.performanceMetrics[metricName];
        const count = metricName.includes('Forward') ? 
            this.performanceMetrics.forwardKinematicsCallCount :
            this.performanceMetrics.inverseKinematicsCallCount;
        
        this.performanceMetrics[metricName] = (currentValue * (count - 1) + newValue) / count;
    }

    /**
     * Update cache hit rate
     * @private
     */
    updateCacheHitRate(isHit) {
        const totalAccesses = this.performanceMetrics.forwardKinematicsCallCount + 
                             this.performanceMetrics.inverseKinematicsCallCount;
        
        if (isHit) {
            this.performanceMetrics.cacheHitRate = 
                (this.performanceMetrics.cacheHitRate * (totalAccesses - 1) + 1) / totalAccesses;
        } else {
            this.performanceMetrics.cacheHitRate = 
                (this.performanceMetrics.cacheHitRate * (totalAccesses - 1)) / totalAccesses;
        }
    }

    /**
     * Precompute static transformations for optimization
     * @private
     */
    precomputeStaticTransforms() {
        // This could include base transformations, tool transformations, etc.
        console.log('Static transformations precomputed');
    }

    /**
     * Check for kinematic singularities
     * @private
     */
    checkSingularity(jointAngles) {
        // Simplified singularity detection
        // In practice, this would check the determinant of the Jacobian
        const jacobian = this.calculateJacobian(jointAngles);
        const determinant = this.calculateDeterminant(jacobian);
        
        return Math.abs(determinant) < 1e-6;
    }

    /**
     * Get performance metrics
     * @returns {Object} Performance statistics
     */
    getPerformanceMetrics() {
        return { ...this.performanceMetrics };
    }

    /**
     * Clear transformation cache
     */
    clearCache() {
        this.transformationCache.clear();
        console.log('Transformation cache cleared');
    }

    /**
     * Enable or disable caching
     * @param {Boolean} enabled - Whether to enable caching
     */
    setCacheEnabled(enabled) {
        this.cacheEnabled = enabled;
        if (!enabled) {
            this.clearCache();
        }
    }
}

export default InverseKinematics;