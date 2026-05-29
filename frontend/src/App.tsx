import { useMemo, useState } from "react";
import { fetchOhlcv, fetchPrediction } from "./api";
import type { OHLCVRow, OHLCVResponse, PredictResponse } from "./types";

const ONE_YEAR_LIMIT = 252;
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

type ChartPoint = { x: number; y: number };

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

function buildLinePoints(rows: OHLCVRow[], width: number, height: number, padding: number) {
  const values = rows.map((row) => row.close);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const xStep = (width - padding * 2) / Math.max(values.length - 1, 1);
  const points = values.map((value, index) => {
    const x = padding + index * xStep;
    const y = padding + (1 - (value - min) / range) * (height - padding * 2);
    return { x, y };
  });
  return { points, min, max };
}

function buildLinePath(points: ChartPoint[]): string {
  return points
    .map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`)
    .map((point, index) => (index === 0 ? `M ${point}` : `L ${point}`))
    .join(" ");
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
    const rows = ohlcv.rows;
    const width = 860;
    const height = 280;
    const padding = 24;
    const { points, min, max } = buildLinePoints(rows, width, height, padding);
    const path = buildLinePath(points);
    const lastPoint = points[points.length - 1];
    const startDate = rows[0]?.date ?? "";
    const endDate = rows[rows.length - 1]?.date ?? "";
    const lastClose = rows[rows.length - 1]?.close ?? 0;
    const ticks = buildTicks(min, max);
    return { min, max, path, startDate, endDate, lastClose, lastPoint, ticks };
  }, [ohlcv]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const [predictionResult, ohlcvResult] = await Promise.all([
        fetchPrediction(ticker.trim()),
        fetchOhlcv(ticker.trim(), ONE_YEAR_LIMIT)
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
            prediction.
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
          <h2 className="card-title">Closing Prices (1 Year)</h2>
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
              <svg viewBox="0 0 860 280" className="chart-svg" aria-label="Closing price chart">
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
                <path d={chart.path} className="chart-line" fill="none" />
                {chart.lastPoint ? (
                  <circle
                    cx={chart.lastPoint.x}
                    cy={chart.lastPoint.y}
                    r={4}
                    className="chart-dot"
                  />
                ) : null}
              </svg>
            </div>
            <div className="chart-footer">
              <span className="mono">Low: {formatCurrency(chart.min)}</span>
              <span className="mono">Latest: {formatCurrency(chart.lastClose)}</span>
              <span className="mono">High: {formatCurrency(chart.max)}</span>
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
              <p className="signal-label">Predicted close</p>
              <p className="signal-value">{formatCurrency(prediction.predicted_close)}</p>
            </div>
            <div>
              <p className="signal-label">Signal</p>
              <p className="signal-value">{prediction.signal}</p>
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
