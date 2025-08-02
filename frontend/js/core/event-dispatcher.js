/**
 * WMG Digital Twin Platform - Event Dispatcher System
 * University of Warwick - WMG Automation Systems Group
 * 
 * Advanced event management system for coordinating industrial automation events
 * across multiple subsystems with priority handling and real-time processing.
 */

class EventDispatcher {
    constructor() {
        this.listeners = new Map();
        this.eventQueue = [];
        this.isProcessing = false;
        this.eventHistory = [];
        this.maxHistorySize = 1000;
        this.eventFilters = new Map();
        this.eventMiddleware = [];
        this.performanceMetrics = {
            totalEvents: 0,
            processedEvents: 0,
            droppedEvents: 0,
            averageProcessingTime: 0,
            lastEventTime: null
        };
        
        // Event priorities for industrial automation
        this.eventPriorities = {
            'EMERGENCY_STOP': 0,
            'SAFETY_ALARM': 1,
            'SYSTEM_ERROR': 2,
            'ROBOT_COLLISION': 3,
            'PROCESS_ALARM': 4,
            'TRAINING_COMPLETE': 5,
            'ROBOT_MOVEMENT': 6,
            'SENSOR_UPDATE': 7,
            'UI_INTERACTION': 8,
            'LOG_MESSAGE': 9,
            'PERFORMANCE_UPDATE': 10
        };
        
        console.log('WMG Event Dispatcher initialized');
    }

    /**
     * Subscribe to events with optional filters and priority
     * @param {string} eventType - Type of event to listen for
     * @param {function} callback - Function to call when event fires
     * @param {Object} options - Optional configuration
     */
    on(eventType, callback, options = {}) {
        if (typeof callback !== 'function') {
            throw new Error('Event callback must be a function');
        }

        const listener = {
            id: this._generateListenerId(),
            callback,
            priority: options.priority || 10,
            once: options.once || false,
            filter: options.filter || null,
            context: options.context || null,
            errorHandler: options.errorHandler || null,
            active: true,
            callCount: 0,
            lastCalled: null
        };

        if (!this.listeners.has(eventType)) {
            this.listeners.set(eventType, []);
        }

        this.listeners.get(eventType).push(listener);
        
        // Sort by priority (lower number = higher priority)
        this.listeners.get(eventType).sort((a, b) => a.priority - b.priority);

        console.log(`Event listener registered: ${eventType} (ID: ${listener.id})`);
        return listener.id;
    }

    /**
     * Subscribe to event once only
     * @param {string} eventType - Type of event to listen for
     * @param {function} callback - Function to call when event fires
     * @param {Object} options - Optional configuration
     */
    once(eventType, callback, options = {}) {
        return this.on(eventType, callback, { ...options, once: true });
    }

    /**
     * Unsubscribe from events
     * @param {string} eventType - Type of event to unsubscribe from
     * @param {string|function} identifier - Listener ID or callback function
     */
    off(eventType, identifier) {
        if (!this.listeners.has(eventType)) {
            return false;
        }

        const listeners = this.listeners.get(eventType);
        let removed = false;

        if (typeof identifier === 'string') {
            // Remove by listener ID
            const index = listeners.findIndex(listener => listener.id === identifier);
            if (index !== -1) {
                listeners.splice(index, 1);
                removed = true;
            }
        } else if (typeof identifier === 'function') {
            // Remove by callback function
            const index = listeners.findIndex(listener => listener.callback === identifier);
            if (index !== -1) {
                listeners.splice(index, 1);
                removed = true;
            }
        }

        if (listeners.length === 0) {
            this.listeners.delete(eventType);
        }

        if (removed) {
            console.log(`Event listener removed: ${eventType}`);
        }

        return removed;
    }

    /**
     * Emit an event to all registered listeners
     * @param {string} eventType - Type of event to emit
     * @param {*} data - Data to pass to event handlers
     * @param {Object} options - Emission options
     */
    emit(eventType, data = null, options = {}) {
        const event = {
            type: eventType,
            data,
            timestamp: Date.now(),
            id: this._generateEventId(),
            priority: this.eventPriorities[eventType] || 10,
            source: options.source || 'unknown',
            target: options.target || null,
            bubbles: options.bubbles !== false,
            cancelable: options.cancelable !== false,
            cancelled: false,
            processed: false
        };

        this.performanceMetrics.totalEvents++;
        
        // Apply middleware
        for (const middleware of this.eventMiddleware) {
            try {
                event = middleware(event) || event;
                if (event.cancelled) {
                    console.log(`Event cancelled by middleware: ${eventType}`);
                    return false;
                }
            } catch (error) {
                console.error('Event middleware error:', error);
            }
        }

        // Add to queue based on priority
        if (options.immediate || event.priority <= 3) {
            // High priority events are processed immediately
            this._processEvent(event);
        } else {
            // Queue for batch processing
            this._queueEvent(event);
        }

        return true;
    }

    /**
     * Emit event and wait for all async handlers to complete
     * @param {string} eventType - Type of event to emit
     * @param {*} data - Data to pass to event handlers
     * @param {Object} options - Emission options
     */
    async emitAsync(eventType, data = null, options = {}) {
        const event = {
            type: eventType,
            data,
            timestamp: Date.now(),
            id: this._generateEventId(),
            priority: this.eventPriorities[eventType] || 10,
            source: options.source || 'unknown',
            target: options.target || null,
            async: true
        };

        return await this._processEventAsync(event);
    }

    /**
     * Add middleware function to process events before dispatch
     * @param {function} middleware - Middleware function
     */
    use(middleware) {
        if (typeof middleware !== 'function') {
            throw new Error('Middleware must be a function');
        }
        this.eventMiddleware.push(middleware);
    }

    /**
     * Add event filter for specific event types
     * @param {string} eventType - Event type to filter
     * @param {function} filter - Filter function
     */
    addFilter(eventType, filter) {
        if (!this.eventFilters.has(eventType)) {
            this.eventFilters.set(eventType, []);
        }
        this.eventFilters.get(eventType).push(filter);
    }

    /**
     * Start event processing loop
     */
    startProcessing() {
        if (this.isProcessing) {
            return;
        }

        this.isProcessing = true;
        this._processEventQueue();
        console.log('Event processing started');
    }

    /**
     * Stop event processing
     */
    stopProcessing() {
        this.isProcessing = false;
        console.log('Event processing stopped');
    }

    /**
     * Clear all event listeners
     */
    clearAllListeners() {
        this.listeners.clear();
        console.log('All event listeners cleared');
    }

    /**
     * Get event statistics and performance metrics
     */
    getMetrics() {
        return {
            ...this.performanceMetrics,
            queueSize: this.eventQueue.length,
            listenerCount: Array.from(this.listeners.values()).reduce((sum, arr) => sum + arr.length, 0),
            eventTypes: Array.from(this.listeners.keys()),
            isProcessing: this.isProcessing
        };
    }

    /**
     * Get event history
     * @param {number} limit - Maximum number of events to return
     */
    getEventHistory(limit = 100) {
        return this.eventHistory.slice(-limit);
    }

    /**
     * Debug method to inspect current state
     */
    debug() {
        console.group('Event Dispatcher Debug');
        console.log('Listeners:', this.listeners);
        console.log('Queue size:', this.eventQueue.length);
        console.log('Metrics:', this.getMetrics());
        console.log('History size:', this.eventHistory.length);
        console.groupEnd();
    }

    // Private methods

    _processEvent(event) {
        const startTime = performance.now();
        
        try {
            if (!this.listeners.has(event.type)) {
                return;
            }

            const listeners = this.listeners.get(event.type);
            const listenersToRemove = [];

            for (const listener of listeners) {
                if (!listener.active) {
                    continue;
                }

                // Apply filters
                if (listener.filter && !listener.filter(event)) {
                    continue;
                }

                try {
                    // Execute callback with proper context
                    const result = listener.context 
                        ? listener.callback.call(listener.context, event)
                        : listener.callback(event);

                    listener.callCount++;
                    listener.lastCalled = Date.now();

                    // Handle once listeners
                    if (listener.once) {
                        listenersToRemove.push(listener);
                    }

                    // Stop propagation if requested
                    if (event.stopPropagation && event.stopPropagation()) {
                        break;
                    }

                } catch (error) {
                    console.error(`Event listener error for ${event.type}:`, error);
                    
                    if (listener.errorHandler) {
                        try {
                            listener.errorHandler(error, event);
                        } catch (handlerError) {
                            console.error('Error handler failed:', handlerError);
                        }
                    }
                }
            }

            // Remove once listeners
            listenersToRemove.forEach(listener => {
                const index = listeners.indexOf(listener);
                if (index !== -1) {
                    listeners.splice(index, 1);
                }
            });

            event.processed = true;
            this.performanceMetrics.processedEvents++;

        } catch (error) {
            console.error('Event processing error:', error);
            this.performanceMetrics.droppedEvents++;
        }

        // Update performance metrics
        const processingTime = performance.now() - startTime;
        this._updateProcessingTime(processingTime);
        
        // Add to history
        this._addToHistory(event);
    }

    async _processEventAsync(event) {
        if (!this.listeners.has(event.type)) {
            return [];
        }

        const listeners = this.listeners.get(event.type);
        const promises = [];
        const listenersToRemove = [];

        for (const listener of listeners) {
            if (!listener.active) {
                continue;
            }

            if (listener.filter && !listener.filter(event)) {
                continue;
            }

            try {
                const result = listener.context 
                    ? listener.callback.call(listener.context, event)
                    : listener.callback(event);

                if (result && typeof result.then === 'function') {
                    promises.push(result);
                }

                listener.callCount++;
                listener.lastCalled = Date.now();

                if (listener.once) {
                    listenersToRemove.push(listener);
                }

            } catch (error) {
                console.error(`Async event listener error for ${event.type}:`, error);
                if (listener.errorHandler) {
                    listener.errorHandler(error, event);
                }
            }
        }

        // Remove once listeners
        listenersToRemove.forEach(listener => {
            const index = listeners.indexOf(listener);
            if (index !== -1) {
                listeners.splice(index, 1);
            }
        });

        event.processed = true;
        this._addToHistory(event);

        return Promise.all(promises);
    }

    _queueEvent(event) {
        // Insert event in priority order
        let insertIndex = this.eventQueue.length;
        for (let i = 0; i < this.eventQueue.length; i++) {
            if (event.priority < this.eventQueue[i].priority) {
                insertIndex = i;
                break;
            }
        }
        
        this.eventQueue.splice(insertIndex, 0, event);
        
        // Limit queue size to prevent memory issues
        if (this.eventQueue.length > 10000) {
            this.eventQueue = this.eventQueue.slice(0, 5000);
            this.performanceMetrics.droppedEvents += 5000;
        }
    }

    _processEventQueue() {
        if (!this.isProcessing) {
            return;
        }

        const batchSize = 50; // Process events in batches
        const batch = this.eventQueue.splice(0, batchSize);
        
        batch.forEach(event => this._processEvent(event));

        // Schedule next batch
        if (this.eventQueue.length > 0 || this.isProcessing) {
            requestAnimationFrame(() => this._processEventQueue());
        }
    }

    _updateProcessingTime(time) {
        const count = this.performanceMetrics.processedEvents;
        const current = this.performanceMetrics.averageProcessingTime;
        this.performanceMetrics.averageProcessingTime = ((current * (count - 1)) + time) / count;
        this.performanceMetrics.lastEventTime = Date.now();
    }

    _addToHistory(event) {
        this.eventHistory.push({
            type: event.type,
            timestamp: event.timestamp,
            id: event.id,
            processed: event.processed,
            priority: event.priority,
            source: event.source
        });

        // Limit history size
        if (this.eventHistory.length > this.maxHistorySize) {
            this.eventHistory = this.eventHistory.slice(-this.maxHistorySize * 0.8);
        }
    }

    _generateListenerId() {
        return `listener_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    _generateEventId() {
        return `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}

// Industrial Automation Event Types
const WMG_EVENTS = {
    // Safety Events (Highest Priority)
    EMERGENCY_STOP: 'EMERGENCY_STOP',
    SAFETY_ALARM: 'SAFETY_ALARM',
    COLLISION_DETECTED: 'COLLISION_DETECTED',
    SAFETY_ZONE_VIOLATION: 'SAFETY_ZONE_VIOLATION',
    
    // System Events
    SYSTEM_STARTUP: 'SYSTEM_STARTUP',
    SYSTEM_SHUTDOWN: 'SYSTEM_SHUTDOWN',
    SYSTEM_ERROR: 'SYSTEM_ERROR',
    SYSTEM_WARNING: 'SYSTEM_WARNING',
    
    // Robot Events
    ROBOT_CONNECTED: 'ROBOT_CONNECTED',
    ROBOT_DISCONNECTED: 'ROBOT_DISCONNECTED',
    ROBOT_MOVEMENT_START: 'ROBOT_MOVEMENT_START',
    ROBOT_MOVEMENT_COMPLETE: 'ROBOT_MOVEMENT_COMPLETE',
    ROBOT_HOME_COMPLETE: 'ROBOT_HOME_COMPLETE',
    ROBOT_PROGRAM_START: 'ROBOT_PROGRAM_START',
    ROBOT_PROGRAM_COMPLETE: 'ROBOT_PROGRAM_COMPLETE',
    ROBOT_JOINT_UPDATE: 'ROBOT_JOINT_UPDATE',
    
    // Training Events
    TRAINING_STARTED: 'TRAINING_STARTED',
    TRAINING_STOPPED: 'TRAINING_STOPPED',
    TRAINING_PAUSED: 'TRAINING_PAUSED',
    TRAINING_EPISODE_COMPLETE: 'TRAINING_EPISODE_COMPLETE',
    TRAINING_PROGRESS_UPDATE: 'TRAINING_PROGRESS_UPDATE',
    
    // Conveyor Events
    CONVEYOR_STARTED: 'CONVEYOR_STARTED',
    CONVEYOR_STOPPED: 'CONVEYOR_STOPPED',
    CONVEYOR_SPEED_CHANGED: 'CONVEYOR_SPEED_CHANGED',
    PART_DETECTED: 'PART_DETECTED',
    PART_PICKED: 'PART_PICKED',
    PART_PLACED: 'PART_PLACED',
    
    // Sensor Events
    SENSOR_VALUE_CHANGED: 'SENSOR_VALUE_CHANGED',
    SENSOR_THRESHOLD_EXCEEDED: 'SENSOR_THRESHOLD_EXCEEDED',
    SENSOR_FAULT: 'SENSOR_FAULT',
    
    // Performance Events
    PERFORMANCE_UPDATE: 'PERFORMANCE_UPDATE',
    CYCLE_TIME_UPDATE: 'CYCLE_TIME_UPDATE',
    QUALITY_ALERT: 'QUALITY_ALERT',
    
    // UI Events
    UI_PANEL_OPENED: 'UI_PANEL_OPENED',
    UI_PANEL_CLOSED: 'UI_PANEL_CLOSED',
    UI_SETTING_CHANGED: 'UI_SETTING_CHANGED',
    
    // Communication Events
    WEBSOCKET_CONNECTED: 'WEBSOCKET_CONNECTED',
    WEBSOCKET_DISCONNECTED: 'WEBSOCKET_DISCONNECTED',
    WEBSOCKET_ERROR: 'WEBSOCKET_ERROR',
    MESSAGE_SENT: 'MESSAGE_SENT',
    MESSAGE_RECEIVED: 'MESSAGE_RECEIVED',
    
    // Code Generation Events
    CODE_GENERATION_STARTED: 'CODE_GENERATION_STARTED',
    CODE_GENERATION_COMPLETE: 'CODE_GENERATION_COMPLETE',
    CODE_GENERATION_ERROR: 'CODE_GENERATION_ERROR'
};

// Create global event dispatcher instance
const wmgEventDispatcher = new EventDispatcher();

// Start processing immediately
wmgEventDispatcher.startProcessing();

// Add safety middleware to handle critical events
wmgEventDispatcher.use((event) => {
    // Log all safety-critical events
    if (event.priority <= 3) {
        console.warn(`CRITICAL EVENT: ${event.type}`, event.data);
    }
    
    // Auto-stop training on emergency
    if (event.type === WMG_EVENTS.EMERGENCY_STOP) {
        wmgEventDispatcher.emit(WMG_EVENTS.TRAINING_STOPPED, {
            reason: 'emergency_stop',
            timestamp: Date.now()
        }, { immediate: true });
    }
    
    return event;
});

// Export for global use
window.wmgEventDispatcher = wmgEventDispatcher;
window.WMG_EVENTS = WMG_EVENTS;

console.log('WMG Event Dispatcher system loaded and ready');