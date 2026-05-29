export type PredictResponse = {
  ticker: string;
  predicted_close: number;
  last_close: number;
  delta_pct: number;
  signal: string;
  scaler_path: string;
};

export type OHLCVRow = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type OHLCVResponse = {
  ticker: string;
  rows: OHLCVRow[];
};
