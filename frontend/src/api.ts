import type { OHLCVResponse, PredictResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchPrediction(ticker: string): Promise<PredictResponse> {
  return requestJson<PredictResponse>(`/predict/${encodeURIComponent(ticker)}`);
}

export async function fetchOhlcv(ticker: string, limit: number): Promise<OHLCVResponse> {
  return requestJson<OHLCVResponse>(`/ohlcv/${encodeURIComponent(ticker)}?limit=${limit}`);
}
