/**
 * WMG RL Digital Twin Platform - State Management System
 * University of Warwick - WMG Automation Systems Group
 * 
 * This module provides centralized state management for the entire application,
 * following Redux-style patterns for predictable state updates and efficient
 * component communication across the digital twin platform.
 */

class StateStore {
    constructor(name, initialState = {}) {
        this.name = name;
        this.state = { ...initialState };
        this.listeners = new Set();
        this.actions = new Map();
        this.middleware = [];
        this.history = [];
        this.maxHistorySize = 50;
        
        // Development tools integration
        this.enableDevTools = process?.env?.NODE_ENV === 'development';
        
        console.log(`StateStore '${name}' initialized with state:`, this.state);
    }

    /**
     * Get the current state of the store
     * @returns {Object} Current state object
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Subscribe to state changes
     * @param {Function} listener - Callback function to execute on state changes
     * @param {Array} dependencies - Optional array of state keys to watch
     * @returns {Function} Unsubscribe function
     */
    subscribe(listener, dependencies = null) {
        if (typeof listener !== 'function') {
            throw new Error('Listener must be a function');
        }

        const wrappedListener = {
            callback: listener,
            dependencies: dependencies,
            id: Symbol('listener')
        };

        this.listeners.add(wrappedListener);

        // Return unsubscribe function
        return () => {
            this.listeners.delete(wrappedListener);
        };
    }

    /**
     * Register an action for state updates
     * @param {String} actionType - Unique identifier for the action
     * @param {Function} actionHandler - Function that receives (state, payload) and returns new state
     */
    registerAction(actionType, actionHandler) {
        if (typeof actionHandler !== 'function') {
            throw new Error('Action handler must be a function');
        }

        this.actions.set(actionType, actionHandler);
        console.log(`Action '${actionType}' registered for store '${this.name}'`);
    }

    /**
     * Dispatch an action to update state
     * @param {String} actionType - The action type to execute
     * @param {*} payload - Data to pass to the action handler
     * @returns {Object} New state after action execution
     */
    dispatch(actionType, payload = null) {
        const actionHandler = this.actions.get(actionType);
        
        if (!actionHandler) {
            console.warn(`Action '${actionType}' not found in store '${this.name}'`);
            return this.state;
        }

        const previousState = { ...this.state };
        
        try {
            // Execute middleware before action
            this.executeMiddleware('before', actionType, payload, previousState);
            
            // Execute the action
            const newState = actionHandler(previousState, payload);
            
            // Validate that action returned a valid state
            if (typeof newState !== 'object' || newState === null) {
                throw new Error(`Action '${actionType}' must return a valid state object`);
            }

            // Update state
            this.state = { ...newState };
            
            // Record state change in history
            this.recordStateChange(actionType, payload, previousState, this.state);
            
            // Execute middleware after action
            this.executeMiddleware('after', actionType, payload, this.state);
            
            // Notify all listeners
            this.notifyListeners(previousState, this.state);
            
            return this.state;
            
        } catch (error) {
            console.error(`Error executing action '${actionType}' in store '${this.name}':`, error);
            return previousState;
        }
    }

    /**
     * Update state directly without actions (use sparingly)
     * @param {Object} partialState - Partial state object to merge
     */
    setState(partialState) {
        const previousState = { ...this.state };
        this.state = { ...this.state, ...partialState };
        
        this.recordStateChange('SET_STATE', partialState, previousState, this.state);
        this.notifyListeners(previousState, this.state);
        
        return this.state;
    }

    /**
     * Add middleware for action processing
     * @param {Function} middleware - Function that receives (phase, action, payload, state)
     */
    addMiddleware(middleware) {
        if (typeof middleware !== 'function') {
            throw new Error('Middleware must be a function');
        }
        
        this.middleware.push(middleware);
    }

    /**
     * Reset store to initial state
     * @param {Object} newInitialState - Optional new initial state
     */
    reset(newInitialState = null) {
        const previousState = { ...this.state };
        
        if (newInitialState) {
            this.state = { ...newInitialState };
        } else {
            // Find the initial state from history
            const initialHistory = this.history.find(entry => entry.action === 'INIT');
            this.state = initialHistory ? { ...initialHistory.previousState } : {};
        }
        
        this.recordStateChange('RESET', newInitialState, previousState, this.state);
        this.notifyListeners(previousState, this.state);
        
        console.log(`Store '${this.name}' reset to initial state`);
    }

    /**
     * Get state change history
     * @param {Number} limit - Maximum number of history entries to return
     * @returns {Array} Array of state change records
     */
    getHistory(limit = 10) {
        return this.history.slice(-limit);
    }

    /**
     * Execute middleware functions
     * @private
     */
    executeMiddleware(phase, actionType, payload, state) {
        this.middleware.forEach(middleware => {
            try {
                middleware(phase, actionType, payload, state);
            } catch (error) {
                console.error('Middleware error:', error);
            }
        });
    }

    /**
     * Record state changes for debugging and time travel
     * @private
     */
    recordStateChange(actionType, payload, previousState, newState) {
        const record = {
            timestamp: Date.now(),
            action: actionType,
            payload: payload,
            previousState: previousState,
            newState: newState,
            storeName: this.name
        };

        this.history.push(record);
        
        // Maintain history size limit
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        }

        // Development tools integration
        if (this.enableDevTools && window.__REDUX_DEVTOOLS_EXTENSION__) {
            window.__REDUX_DEVTOOLS_EXTENSION__.send(
                `${this.name}/${actionType}`,
                newState
            );
        }
    }

    /**
     * Notify all subscribers of state changes
     * @private
     */
    notifyListeners(previousState, newState) {
        this.listeners.forEach(listener => {
            try {
                // Check if listener has dependency filters
                if (listener.dependencies && Array.isArray(listener.dependencies)) {
                    // Only notify if watched dependencies changed
                    const hasChanges = listener.dependencies.some(key => 
                        previousState[key] !== newState[key]
                    );
                    
                    if (hasChanges) {
                        listener.callback(newState, previousState);
                    }
                } else {
                    // Notify for all changes
                    listener.callback(newState, previousState);
                }
            } catch (error) {
                console.error('Listener error:', error);
            }
        });
    }
}

export class StateManager {
    constructor() {
        this.stores = new Map();
        this.globalMiddleware = [];
        
        // Performance monitoring
        this.performanceMetrics = {
            totalDispatches: 0,
            averageDispatchTime: 0,
            lastDispatchTime: 0
        };
        
        console.log('StateManager initialized');
    }

    /**
     * Create a new state store
     * @param {String} storeName - Unique identifier for the store
     * @param {Object} initialState - Initial state object
     * @returns {StateStore} The created store instance
     */
    createStore(storeName, initialState = {}) {
        if (this.stores.has(storeName)) {
            console.warn(`Store '${storeName}' already exists. Returning existing store.`);
            return this.stores.get(storeName);
        }

        const store = new StateStore(storeName, initialState);
        
        // Add global middleware to the store
        this.globalMiddleware.forEach(middleware => {
            store.addMiddleware(middleware);
        });

        // Add performance tracking middleware
        store.addMiddleware(this.createPerformanceMiddleware());
        
        this.stores.set(storeName, store);
        
        console.log(`Store '${storeName}' created successfully`);
        return store;
    }

    /**
     * Get an existing store
     * @param {String} storeName - Name of the store to retrieve
     * @returns {StateStore|null} The store instance or null if not found
     */
    getStore(storeName) {
        const store = this.stores.get(storeName);
        
        if (!store) {
            console.warn(`Store '${storeName}' not found`);
            return null;
        }
        
        return store;
    }

    /**
     * Remove a store
     * @param {String} storeName - Name of the store to remove
     * @returns {Boolean} True if store was removed, false if not found
     */
    removeStore(storeName) {
        const removed = this.stores.delete(storeName);
        
        if (removed) {
            console.log(`Store '${storeName}' removed`);
        } else {
            console.warn(`Store '${storeName}' not found for removal`);
        }
        
        return removed;
    }

    /**
     * Add global middleware that applies to all stores
     * @param {Function} middleware - Middleware function
     */
    addGlobalMiddleware(middleware) {
        if (typeof middleware !== 'function') {
            throw new Error('Global middleware must be a function');
        }

        this.globalMiddleware.push(middleware);
        
        // Apply to existing stores
        this.stores.forEach(store => {
            store.addMiddleware(middleware);
        });
    }

    /**
     * Get combined state from all stores
     * @returns {Object} Object containing all store states
     */
    getCombinedState() {
        const combinedState = {};
        
        this.stores.forEach((store, storeName) => {
            combinedState[storeName] = store.getState();
        });
        
        return combinedState;
    }

    /**
     * Subscribe to changes across multiple stores
     * @param {Array} storeNames - Array of store names to watch
     * @param {Function} listener - Callback function
     * @returns {Function} Unsubscribe function
     */
    subscribeToMultiple(storeNames, listener) {
        const unsubscribeFunctions = [];
        
        storeNames.forEach(storeName => {
            const store = this.getStore(storeName);
            if (store) {
                const unsubscribe = store.subscribe((newState, previousState) => {
                    listener(storeName, newState, previousState);
                });
                unsubscribeFunctions.push(unsubscribe);
            }
        });
        
        // Return function to unsubscribe from all stores
        return () => {
            unsubscribeFunctions.forEach(unsubscribe => unsubscribe());
        };
    }

    /**
     * Dispatch action to specific store
     * @param {String} storeName - Target store name
     * @param {String} actionType - Action type to dispatch
     * @param {*} payload - Action payload
     * @returns {Object|null} New state or null if store not found
     */
    dispatch(storeName, actionType, payload) {
        const store = this.getStore(storeName);
        
        if (!store) {
            return null;
        }
        
        return store.dispatch(actionType, payload);
    }

    /**
     * Reset all stores to their initial states
     */
    resetAllStores() {
        this.stores.forEach(store => {
            store.reset();
        });
        
        console.log('All stores reset to initial state');
    }

    /**
     * Get performance metrics
     * @returns {Object} Performance statistics
     */
    getPerformanceMetrics() {
        return { ...this.performanceMetrics };
    }

    /**
     * Export current state for persistence
     * @returns {Object} Serializable state object
     */
    exportState() {
        const exportData = {
            timestamp: Date.now(),
            version: '1.0.0',
            stores: {}
        };
        
        this.stores.forEach((store, storeName) => {
            exportData.stores[storeName] = {
                state: store.getState(),
                history: store.getHistory()
            };
        });
        
        return exportData;
    }

    /**
     * Import state from exported data
     * @param {Object} exportedData - Previously exported state data
     */
    importState(exportedData) {
        if (!exportedData || !exportedData.stores) {
            throw new Error('Invalid export data format');
        }
        
        Object.entries(exportedData.stores).forEach(([storeName, storeData]) => {
            const store = this.getStore(storeName);
            if (store && storeData.state) {
                store.setState(storeData.state);
                console.log(`State imported for store '${storeName}'`);
            }
        });
    }

    /**
     * Create performance monitoring middleware
     * @private
     * @returns {Function} Middleware function
     */
    createPerformanceMiddleware() {
        return (phase, actionType, payload, state) => {
            if (phase === 'before') {
                this.performanceMetrics.lastDispatchTime = performance.now();
            } else if (phase === 'after') {
                const duration = performance.now() - this.performanceMetrics.lastDispatchTime;
                this.performanceMetrics.totalDispatches++;
                
                // Calculate rolling average
                const currentAverage = this.performanceMetrics.averageDispatchTime;
                const newAverage = (currentAverage * (this.performanceMetrics.totalDispatches - 1) + duration) / this.performanceMetrics.totalDispatches;
                this.performanceMetrics.averageDispatchTime = newAverage;
            }
        };
    }

    /**
     * Create logging middleware for debugging
     * @param {String} logLevel - Log level ('info', 'debug', 'warn')
     * @returns {Function} Middleware function
     */
    createLoggingMiddleware(logLevel = 'info') {
        return (phase, actionType, payload, state) => {
            if (phase === 'before' && console[logLevel]) {
                console[logLevel](`Action dispatched: ${actionType}`, {
                    payload,
                    timestamp: new Date().toISOString()
                });
            }
        };
    }

    /**
     * Destroy the state manager and cleanup resources
     */
    destroy() {
        this.stores.clear();
        this.globalMiddleware.length = 0;
        this.performanceMetrics = {
            totalDispatches: 0,
            averageDispatchTime: 0,
            lastDispatchTime: 0
        };
        
        console.log('StateManager destroyed');
    }
}

// Export default instance for convenience
const stateManager = new StateManager();
export default stateManager;