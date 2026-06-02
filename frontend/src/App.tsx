import { useMemo, useState } from "react";
import { fetchOhlcv, fetchPrediction } from "./api";
import type { OHLCVRow, OHLCVResponse, PredictResponse } from "./types";

const RECENT_DAYS_LIMIT = 21;
const SUGGESTED_TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "AMD"];

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2
  }).format(value);
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2
  }).format(value);
}

type CandleDatum = {
  x: number;
  openY: number;
  closeY: number;
  highY: number;
  lowY: number;
  isUp: boolean;
  isPrediction: boolean;
};

function pickStep(range: number): number {
  const bases = [1, 2, 5, 10];
  const exponent = Math.floor(Math.log10(range / 4 || 1));
  const baseScale = Math.pow(10, exponent);
  for (const base of bases) {
    const step = base * baseScale;
    if (range / step <= 5) {
      return step;
    }
  }
  return 10 * baseScale;
}

function buildTicks(min: number, max: number): number[] {
  const range = max - min || 1;
  const step = pickStep(range);
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= end + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
}

function buildCandles(rows: OHLCVRow[], width: number, height: number, padding: number) {
  const lows = rows.map((row) => row.low);
  const highs = rows.map((row) => row.high);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const range = max - min || 1;
  const count = Math.max(rows.length, 1);
  const xStep = (width - padding * 2) / count;
  const candleWidth = Math.min(18, xStep * 0.6);

  const candles: CandleDatum[] = rows.map((row, index) => {
    const centerX = padding + index * xStep + xStep / 2;
    const highY = padding + (1 - (row.high - min) / range) * (height - padding * 2);
    const lowY = padding + (1 - (row.low - min) / range) * (height - padding * 2);
    const openY = padding + (1 - (row.open - min) / range) * (height - padding * 2);
    const closeY = padding + (1 - (row.close - min) / range) * (height - padding * 2);
    const isUp = row.close >= row.open;
    const isPrediction = row.date === "Prediction";
    return { x: centerX, highY, lowY, openY, closeY, isUp, isPrediction };
  });

  return { candles, min, max, candleWidth };
}

export default function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [ohlcv, setOhlcv] = useState<OHLCVResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chart = useMemo(() => {
    if (!ohlcv) {
      return null;
    }
    const rows = [...ohlcv.rows];
    if (prediction) {
      rows.push({
        date: "Prediction",
        open: prediction.predicted_open,
        high: prediction.predicted_high,
        low: prediction.predicted_low,
        close: prediction.predicted_close,
        volume: 0
      });
    }
    const width = 860;
    const height = 280;
    const padding = 24;
    const { candles, min, max, candleWidth } = buildCandles(rows, width, height, padding);
    const startDate = ohlcv.rows[0]?.date ?? "";
    const endDate = ohlcv.rows[ohlcv.rows.length - 1]?.date ?? "";
    const lastClose = ohlcv.rows[ohlcv.rows.length - 1]?.close ?? 0;
    const ticks = buildTicks(min, max);
    return { min, max, candles, candleWidth, startDate, endDate, lastClose, ticks };
  }, [ohlcv, prediction]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const [predictionResult, ohlcvResult] = await Promise.all([
        fetchPrediction(ticker.trim()),
        fetchOhlcv(ticker.trim(), RECENT_DAYS_LIMIT)
      ]);
      setPrediction(predictionResult);
      setOhlcv(ohlcvResult);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setPrediction(null);
      setOhlcv(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <p className="eyebrow">STOCK DASHBOARD</p>
          <h1 className="title">Check a Stock, Instantly</h1>
          <p className="lede">
            Enter a ticker to view the last year of closing prices and the model's next-day
            OHLC prediction.
          </p>
        </div>
        <div className="rule" aria-hidden="true" />
      </header>

      <form className="controls" onSubmit={handleSubmit}>
        <label className="field">
          <span className="field-label">Ticker</span>
          <input
            value={ticker}
            onChange={(event) => setTicker(event.target.value.toUpperCase())}
            placeholder="AAPL"
            list="ticker-list"
            className="input"
          />
          <datalist id="ticker-list">
            {SUGGESTED_TICKERS.map((symbol) => (
              <option key={symbol} value={symbol} />
            ))}
          </datalist>
        </label>
        <button className="button" type="submit" disabled={loading}>
          {loading ? "LOADING" : "VIEW"}
        </button>
        <div className="hint">Data comes from the local API service.</div>
      </form>

      {error ? <p className="error">{error}</p> : null}

      <section className="card">
        <div className="card-header">
          <h2 className="card-title">Candles (21 Days + Prediction)</h2>
          <span className="card-meta">{chart ? `${chart.startDate} to ${chart.endDate}` : "Awaiting data"}</span>
        </div>
        {chart ? (
          <div className="chart">
            <div className="chart-shell">
              <div className="chart-axis">
                {chart.ticks
                  .slice()
                  .reverse()
                  .map((tick) => (
                    <span key={tick} className="axis-value">
                      {formatCurrency(tick)}
                    </span>
                  ))}
              </div>
              <svg viewBox="0 0 860 280" className="chart-svg" aria-label="Candlestick chart">
                {[0.25, 0.5, 0.75].map((value) => (
                  <line
                    key={value}
                    x1={20}
                    x2={840}
                    y1={20 + value * 240}
                    y2={20 + value * 240}
                    className="chart-grid-line"
                  />
                ))}
                {chart.candles.map((candle, index) => {
                  const bodyTop = Math.min(candle.openY, candle.closeY);
                  const bodyHeight = Math.max(2, Math.abs(candle.openY - candle.closeY));
                  const directionClass = candle.isUp ? "candle-up" : "candle-down";
                  const predictionClass = candle.isPrediction ? "candle-pred" : "";
                  return (
                    <g key={`${candle.x}-${index}`} className={`candle ${directionClass} ${predictionClass}`}>
                      <line
                        x1={candle.x}
                        x2={candle.x}
                        y1={candle.highY}
                        y2={candle.lowY}
                        className="candle-wick"
                      />
                      <rect
                        x={candle.x - chart.candleWidth / 2}
                        y={bodyTop}
                        width={chart.candleWidth}
                        height={bodyHeight}
                        className="candle-body"
                      />
                    </g>
                  );
                })}
              </svg>
            </div>
            <div className="chart-footer">
              <span className="mono">Low: {formatCurrency(chart.min)}</span>
              <span className="mono">Latest: {formatCurrency(chart.lastClose)}</span>
              <span className="mono">High: {formatCurrency(chart.max)}</span>
              <span className="mono">Prediction: next candle</span>
            </div>
          </div>
        ) : (
          <div className="empty">Enter a ticker to load the chart.</div>
        )}
      </section>

      <section className="card inverted">
        <div className="card-header">
          <h2 className="card-title">Prediction for Tomorrow</h2>
          <span className="card-meta">Model output</span>
        </div>
        {prediction ? (
          <div className="signal">
            <div>
              <p className="signal-label">Ticker</p>
              <p className="signal-value">{prediction.ticker}</p>
            </div>
            <div>
              <p className="signal-label">Predicted open</p>
              <p className="signal-value">{formatCurrency(prediction.predicted_open)}</p>
            </div>
            <div>
              <p className="signal-label">Predicted high</p>
              <p className="signal-value">{formatCurrency(prediction.predicted_high)}</p>
            </div>
            <div>
              <p className="signal-label">Predicted low</p>
              <p className="signal-value">{formatCurrency(prediction.predicted_low)}</p>
            </div>
            <div>
              <p className="signal-label">Predicted close</p>
              <p className="signal-value">{formatCurrency(prediction.predicted_close)}</p>
            </div>
            <div>
              <p className="signal-label">Signal</p>
              <p className={`signal-value signal-${prediction.signal.toLowerCase()}`}>
                {prediction.signal}
              </p>
            </div>
            <div>
              <p className="signal-label">Change</p>
              <p className="signal-value">
                {prediction.delta_pct > 0 ? "+" : ""}
                {formatNumber(prediction.delta_pct)}%
              </p>
            </div>
          </div>
        ) : (
          <div className="empty">Enter a ticker to see the prediction.</div>
        )}
      </section>
    </div>
  );
}
