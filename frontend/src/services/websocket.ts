import type {
  WebSocketConfig,
  WebSocketMessage,
  RiskUpdateMessage,
  ErrorMessage,
} from '../types/websocket';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export class RiskWebSocketClient {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private isIntentionallyClosed = false;

  constructor(config: WebSocketConfig) {
    this.config = config;
    this.connect();
  }

  private connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      // Build WebSocket URL with query parameters
      const params = new URLSearchParams({
        start_time: this.config.startTime.toISOString(),
        end_time: this.config.endTime.toISOString(),
      });

      if (this.config.bbox) {
        params.append('min_lat', this.config.bbox[0].toString());
        params.append('min_lng', this.config.bbox[1].toString());
        params.append('max_lat', this.config.bbox[2].toString());
        params.append('max_lng', this.config.bbox[3].toString());
      }

      const wsUrl = API_BASE_URL.replace('http', 'ws') + `/realtime/risk-updates?${params.toString()}`;
      
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.config.onConnect?.();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        // Only log error if WebSocket is not intentionally closed
        if (!this.isIntentionallyClosed) {
          console.error('WebSocket error:', error);
          this.config.onError?.(new Error('WebSocket connection error'));
        }
      };

      this.ws.onclose = () => {
        if (!this.isIntentionallyClosed) {
          console.log('WebSocket disconnected');
          this.config.onDisconnect?.();

          // Attempt to reconnect if not intentionally closed
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        } else {
          // Intentionally closed, just log debug message
          console.debug('WebSocket intentionally closed');
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      this.config.onError?.(error as Error);
      this.scheduleReconnect();
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'risk_update':
        this.config.onRiskUpdate((message as RiskUpdateMessage).data);
        break;
      case 'heartbeat':
        // Heartbeat received, connection is alive
        break;
      case 'error':
        const errorMessage = (message as ErrorMessage).data.message;
        console.error('WebSocket error message:', errorMessage);
        this.config.onError?.(new Error(errorMessage));
        break;
      default:
        console.warn('Unknown message type:', message);
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    );

    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  public updateConfig(config: Partial<WebSocketConfig>): void {
    this.config = { ...this.config, ...config };
    
    // Reconnect with new config
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.disconnect();
      this.connect();
    }
  }

  public disconnect(): void {
    this.isIntentionallyClosed = true;
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.ws) {
      // Only close if WebSocket is in a state that allows closing
      // CONNECTING (0), OPEN (1), CLOSING (2) - avoid closing if already CLOSED (3)
      if (this.ws.readyState === WebSocket.CONNECTING || 
          this.ws.readyState === WebSocket.OPEN || 
          this.ws.readyState === WebSocket.CLOSING) {
        try {
          this.ws.close();
        } catch (error) {
          // Ignore errors during close (e.g., already closed)
          console.debug('WebSocket close error (ignored):', error);
        }
      }
      this.ws = null;
    }
  }

  public isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

