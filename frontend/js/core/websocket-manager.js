/**
 * WMG RL Digital Twin Platform - WebSocket Communication Manager
 * University of Warwick - WMG Automation Systems Group
 * 
 * This module manages real-time bidirectional communication between the frontend
 * digital twin interface and the backend reinforcement learning system through
 * WebSocket connections with automatic reconnection and message queuing.
 */

export class WebSocketManager {
    constructor(url = 'ws://localhost:8000/ws') {
        this.url = url;
        this.websocket = null;
        this.connectionState = 'disconnected'; // disconnected, connecting, connected, error
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 3000;
        this.reconnectMultiplier = 1.5;
        this.maxReconnectDelay = 30000;
        
        // Message handling
        this.messageHandlers = new Map();
        this.messageQueue = [];
        this.maxQueueSize = 100;
        
        // Connection monitoring
        this.heartbeatInterval = null;
        this.heartbeatTimeout = null;
        this.heartbeatDelay = 30000; // 30 seconds
        this.heartbeatTimeoutDuration = 10000; // 10 seconds
        
        // Statistics
        this.statistics = {
            totalConnections: 0,
            totalReconnections: 0,
            messagesSent: 0,
            messagesReceived: 0,
            lastConnectedAt: null,
            lastDisconnectedAt: null,
            connectionUptime: 0
        };
        
        // Event listeners
        this.eventListeners = {
            connect: new Set(),
            disconnect: new Set(),
            error: new Set(),
            message: new Set(),
            reconnect: new Set()
        };
        
        console.log('WebSocketManager initialized with URL:', url);
    }

    /**
     * Establish WebSocket connection to the backend
     * @returns {Promise<void>} Resolves when connection is established
     */
    async connect() {
        if (this.connectionState === 'connected' || this.connectionState === 'connecting') {
            console.log('WebSocket already connected or connecting');
            return;
        }

        this.connectionState = 'connecting';
        console.log('Attempting to connect to WebSocket server...');

        try {
            this.websocket = new WebSocket(this.url);
            this.setupWebSocketEventHandlers();
            
            // Return promise that resolves when connection opens
            return new Promise((resolve, reject) => {
                const connectTimeout = setTimeout(() => {
                    reject(new Error('WebSocket connection timeout'));
                }, 10000);

                this.websocket.addEventListener('open', () => {
                    clearTimeout(connectTimeout);
                    resolve();
                }, { once: true });

                this.websocket.addEventListener('error', (error) => {
                    clearTimeout(connectTimeout);
                    reject(error);
                }, { once: true });
            });

        } catch (error) {
            this.connectionState = 'error';
            console.error('Failed to create WebSocket connection:', error);
            this.handleConnectionError(error);
            throw error;
        }
    }

    /**
     * Disconnect from WebSocket server
     * @param {Number} code - WebSocket close code
     * @param {String} reason - Reason for disconnection
     */
    disconnect(code = 1000, reason = 'Manual disconnect') {
        console.log('Disconnecting WebSocket:', reason);
        
        this.stopHeartbeat();
        this.reconnectAttempts = this.maxReconnectAttempts; // Prevent auto-reconnect
        
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.close(code, reason);
        }
        
        this.connectionState = 'disconnected';
        this.statistics.lastDisconnectedAt = Date.now();
        this.updateConnectionUptime();
    }

    /**
     * Send message to backend server
     * @param {Object} message - Message object to send
     * @returns {Boolean} True if message was sent, false if queued
     */
    send(message) {
        const messageData = {
            ...message,
            timestamp: Date.now(),
            id: this.generateMessageId()
        };

        if (this.connectionState === 'connected' && this.websocket.readyState === WebSocket.OPEN) {
            try {
                this.websocket.send(JSON.stringify(messageData));
                this.statistics.messagesSent++;
                console.log('Message sent:', messageData);
                return true;
            } catch (error) {
                console.error('Error sending message:', error);
                this.queueMessage(messageData);
                return false;
            }
        } else {
            console.log('WebSocket not ready, queueing message');
            this.queueMessage(messageData);
            return false;
        }
    }

    /**
     * Register message handler for specific message types
     * @param {String} messageType - Type of message to handle
     * @param {Function} handler - Handler function
     * @returns {Function} Unregister function
     */
    onMessage(messageType, handler) {
        if (typeof handler !== 'function') {
            throw new Error('Message handler must be a function');
        }

        if (!this.messageHandlers.has(messageType)) {
            this.messageHandlers.set(messageType, new Set());
        }

        this.messageHandlers.get(messageType).add(handler);
        console.log(`Message handler registered for type: ${messageType}`);

        // Return unregister function
        return () => {
            const handlers = this.messageHandlers.get(messageType);
            if (handlers) {
                handlers.delete(handler);
                if (handlers.size === 0) {
                    this.messageHandlers.delete(messageType);
                }
            }
        };
    }

    /**
     * Register event listener for connection events
     * @param {String} eventType - Event type (connect, disconnect, error, message, reconnect)
     * @param {Function} listener - Event listener function
     * @returns {Function} Unregister function
     */
    addEventListener(eventType, listener) {
        if (!this.eventListeners[eventType]) {
            throw new Error(`Invalid event type: ${eventType}`);
        }

        this.eventListeners[eventType].add(listener);

        return () => {
            this.eventListeners[eventType].delete(listener);
        };
    }

    /**
     * Get current connection status
     * @returns {String} Connection state
     */
    getConnectionState() {
        return this.connectionState;
    }

    /**
     * Check if WebSocket is currently connected
     * @returns {Boolean} True if connected
     */
    isConnected() {
        return this.connectionState === 'connected' && 
               this.websocket && 
               this.websocket.readyState === WebSocket.OPEN;
    }

    /**
     * Get connection statistics
     * @returns {Object} Statistics object
     */
    getStatistics() {
        this.updateConnectionUptime();
        return { ...this.statistics };
    }

    /**
     * Setup WebSocket event handlers
     * @private
     */
    setupWebSocketEventHandlers() {
        if (!this.websocket) return;

        this.websocket.addEventListener('open', (event) => {
            this.handleConnectionOpen(event);
        });

        this.websocket.addEventListener('message', (event) => {
            this.handleMessage(event);
        });

        this.websocket.addEventListener('close', (event) => {
            this.handleConnectionClose(event);
        });

        this.websocket.addEventListener('error', (event) => {
            this.handleConnectionError(event);
        });
    }

    /**
     * Handle WebSocket connection open
     * @private
     */
    handleConnectionOpen(event) {
        console.log('WebSocket connection established');
        
        this.connectionState = 'connected';
        this.reconnectAttempts = 0;
        this.statistics.totalConnections++;
        this.statistics.lastConnectedAt = Date.now();
        
        // Send queued messages
        this.processMessageQueue();
        
        // Start heartbeat
        this.startHeartbeat();
        
        // Notify event listeners
        this.eventListeners.connect.forEach(listener => {
            try {
                listener(event);
            } catch (error) {
                console.error('Error in connect event listener:', error);
            }
        });
    }

    /**
     * Handle incoming WebSocket messages
     * @private
     */
    handleMessage(event) {
        try {
            const messageData = JSON.parse(event.data);
            this.statistics.messagesReceived++;
            
            console.log('Message received:', messageData);

            // Handle heartbeat response
            if (messageData.type === 'pong') {
                this.handleHeartbeatResponse();
                return;
            }

            // Route message to appropriate handlers
            const messageType = messageData.type;
            if (this.messageHandlers.has(messageType)) {
                this.messageHandlers.get(messageType).forEach(handler => {
                    try {
                        handler(messageData);
                    } catch (error) {
                        console.error(`Error in message handler for type ${messageType}:`, error);
                    }
                });
            } else {
                console.log(`No handlers registered for message type: ${messageType}`);
            }

            // Notify general message listeners
            this.eventListeners.message.forEach(listener => {
                try {
                    listener(messageData);
                } catch (error) {
                    console.error('Error in message event listener:', error);
                }
            });

        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    /**
     * Handle WebSocket connection close
     * @private
     */
    handleConnectionClose(event) {
        console.log('WebSocket connection closed:', event.code, event.reason);
        
        this.connectionState = 'disconnected';
        this.statistics.lastDisconnectedAt = Date.now();
        this.updateConnectionUptime();
        
        this.stopHeartbeat();

        // Notify event listeners
        this.eventListeners.disconnect.forEach(listener => {
            try {
                listener(event);
            } catch (error) {
                console.error('Error in disconnect event listener:', error);
            }
        });

        // Attempt reconnection if not manually disconnected
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket connection errors
     * @private
     */
    handleConnectionError(error) {
        console.error('WebSocket connection error:', error);
        
        this.connectionState = 'error';
        
        // Notify event listeners
        this.eventListeners.error.forEach(listener => {
            try {
                listener(error);
            } catch (listenerError) {
                console.error('Error in error event listener:', listenerError);
            }
        });
    }

    /**
     * Schedule automatic reconnection
     * @private
     */
    scheduleReconnect() {
        this.reconnectAttempts++;
        
        const delay = Math.min(
            this.reconnectDelay * Math.pow(this.reconnectMultiplier, this.reconnectAttempts - 1),
            this.maxReconnectDelay
        );

        console.log(`Scheduling reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);

        setTimeout(async () => {
            if (this.connectionState === 'disconnected' || this.connectionState === 'error') {
                try {
                    this.statistics.totalReconnections++;
                    
                    // Notify reconnect listeners
                    this.eventListeners.reconnect.forEach(listener => {
                        try {
                            listener(this.reconnectAttempts);
                        } catch (error) {
                            console.error('Error in reconnect event listener:', error);
                        }
                    });
                    
                    await this.connect();
                    console.log('Reconnection successful');
                } catch (error) {
                    console.error('Reconnection failed:', error);
                    
                    if (this.reconnectAttempts < this.maxReconnectAttempts) {
                        this.scheduleReconnect();
                    } else {
                        console.error('Maximum reconnection attempts reached');
                    }
                }
            }
        }, delay);
    }

    /**
     * Queue message for later sending
     * @private
     */
    queueMessage(message) {
        if (this.messageQueue.length >= this.maxQueueSize) {
            // Remove oldest message
            this.messageQueue.shift();
            console.warn('Message queue full, removed oldest message');
        }

        this.messageQueue.push(message);
        console.log(`Message queued. Queue size: ${this.messageQueue.length}`);
    }

    /**
     * Process queued messages when connection is restored
     * @private
     */
    processMessageQueue() {
        if (this.messageQueue.length === 0) return;

        console.log(`Processing ${this.messageQueue.length} queued messages`);

        const messages = [...this.messageQueue];
        this.messageQueue.length = 0;

        messages.forEach(message => {
            this.send(message);
        });
    }

    /**
     * Start heartbeat mechanism
     * @private
     */
    startHeartbeat() {
        this.stopHeartbeat(); // Clear any existing heartbeat

        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected()) {
                this.send({ action: 'ping' });
                
                // Set timeout for heartbeat response
                this.heartbeatTimeout = setTimeout(() => {
                    console.warn('Heartbeat timeout, connection may be lost');
                    if (this.websocket) {
                        this.websocket.close(1001, 'Heartbeat timeout');
                    }
                }, this.heartbeatTimeoutDuration);
            }
        }, this.heartbeatDelay);
    }

    /**
     * Stop heartbeat mechanism
     * @private
     */
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }

        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }

    /**
     * Handle heartbeat response from server
     * @private
     */
    handleHeartbeatResponse() {
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }

    /**
     * Generate unique message ID
     * @private
     * @returns {String} Unique message identifier
     */
    generateMessageId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Update connection uptime statistics
     * @private
     */
    updateConnectionUptime() {
        if (this.statistics.lastConnectedAt && this.connectionState === 'connected') {
            this.statistics.connectionUptime = Date.now() - this.statistics.lastConnectedAt;
        }
    }

    /**
     * Reset all statistics
     */
    resetStatistics() {
        this.statistics = {
            totalConnections: 0,
            totalReconnections: 0,
            messagesSent: 0,
            messagesReceived: 0,
            lastConnectedAt: null,
            lastDisconnectedAt: null,
            connectionUptime: 0
        };
        
        console.log('WebSocket statistics reset');
    }

    /**
     * Get detailed connection information
     * @returns {Object} Connection information
     */
    getConnectionInfo() {
        return {
            url: this.url,
            state: this.connectionState,
            readyState: this.websocket ? this.websocket.readyState : null,
            reconnectAttempts: this.reconnectAttempts,
            maxReconnectAttempts: this.maxReconnectAttempts,
            queuedMessages: this.messageQueue.length,
            registeredHandlers: Object.fromEntries(
                Array.from(this.messageHandlers.entries()).map(([type, handlers]) => [type, handlers.size])
            ),
            statistics: this.getStatistics()
        };
    }

    /**
     * Cleanup and destroy the WebSocket manager
     */
    destroy() {
        console.log('Destroying WebSocketManager');
        
        this.disconnect();
        this.stopHeartbeat();
        
        // Clear all handlers and listeners
        this.messageHandlers.clear();
        Object.values(this.eventListeners).forEach(listenerSet => listenerSet.clear());
        
        // Clear message queue
        this.messageQueue.length = 0;
        
        this.websocket = null;
    }
}

// Export default instance for convenience
export default WebSocketManager;