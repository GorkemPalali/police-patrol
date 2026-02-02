export interface RiskUpdateMessage {
  type: 'risk_update';
  data: RiskMapData;
  timestamp: string;
}

export interface HeartbeatMessage {
  type: 'heartbeat';
  timestamp: string;
}

export interface ErrorMessage {
  type: 'error';
  data: {
    message: string;
  };
  timestamp: string;
}

export type WebSocketMessage = RiskUpdateMessage | HeartbeatMessage | ErrorMessage;

export interface RiskMapData {
  cells: RiskCell[];
  time_window: {
    start: string;
    end: string;
  };
  bbox?: [number, number, number, number];
  grid_size_m: number;
  use_hex: boolean;
}

export interface RiskCell {
  id: string;
  lat: number;
  lng: number;
  risk_score: number;
  confidence: number;
}

export interface WebSocketConfig {
  startTime: Date;
  endTime: Date;
  bbox?: [number, number, number, number];
  onRiskUpdate: (data: RiskMapData) => void;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}




