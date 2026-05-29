import { useMemo, useState } from "react";
import { fetchOhlcv, fetchPrediction } from "./api";
import type { OHLCVRow, OHLCVResponse, PredictResponse } from "./types";

const DEFAULT_LIMIT = 60;

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

function buildSparkline(rows: OHLCVRow[]): string {
  if (rows.length === 0) {
    return "";
  }
  const values = rows.map((row) => row.close);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const width = 180;
  const height = 48;
  const step = width / Math.max(values.length - 1, 1);
  return values
    .map((value, idx) => {
      const x = idx * step;
      const y = height - ((value - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [limit, setLimit] = useState(DEFAULT_LIMIT);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [ohlcv, setOhlcv] = useState<OHLCVResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sparkline = useMemo(() => {
    return ohlcv ? buildSparkline(ohlcv.rows) : "";
  }, [ohlcv]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const [predictionResult, ohlcvResult] = await Promise.all([
        fetchPrediction(ticker.trim()),
        fetchOhlcv(ticker.trim(), limit)
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
    <div className="page">
      <header className="hero">
        <div className="hero-grid">
          <div>
            <p className="eyebrow">STOCK PREDICTION STUDIO</p>
            <h1 className="hero-title">
              MARKET
              <span className="hero-title-break">DISCIPLINE</span>
            </h1>
            <p className="hero-lede">
              A monochrome command center for probabilistic next-close signals built on
              a recurrent LSTM model and two years of daily market history.
            </p>
          </div>
          <div className="hero-rule">
            <div className="rule-block" />
            <div className="rule-line" />
          </div>
        </div>
        <div className="hero-meta">
          <div>
            <span className="meta-label">MODEL</span>
            <span className="meta-value">LSTM REGRESSOR</span>
          </div>
          <div>
            <span className="meta-label">FEATURES</span>
            <span className="meta-value">OPEN, HIGH, LOW, CLOSE, VOLUME</span>
          </div>
          <div>
            <span className="meta-label">SIGNAL</span>
            <span className="meta-value">BUY, HOLD, SELL</span>
          </div>
        </div>
      </header>

      <section className="section">
        <div className="section-header">
          <h2 className="section-title">Prediction Interface</h2>
          <p className="section-subtitle">Call the FastAPI service for a live prediction.</p>
        </div>
        <div className="panel-grid">
          <form className="panel" onSubmit={handleSubmit}>
            <div className="panel-header">
              <h3>Request</h3>
              <span className="panel-tag">/predict and /ohlcv</span>
            </div>
            <label className="field">
              <span className="field-label">Ticker symbol</span>
              <input
                value={ticker}
                onChange={(event) => setTicker(event.target.value.toUpperCase())}
                placeholder="AAPL"
                className="input"
              />
            </label>
            <label className="field">
              <span className="field-label">History rows</span>
              <input
                type="number"
                min={10}
                max={200}
                value={limit}
                onChange={(event) => setLimit(Number(event.target.value))}
                className="input"
              />
            </label>
            <button className="button button-primary" type="submit" disabled={loading}>
              {loading ? "RUNNING" : "RUN PREDICTION"}
            </button>
            {error ? <p className="error">{error}</p> : null}
            <div className="note">
              Base URL is read from <span className="mono">VITE_API_BASE</span>.
            </div>
          </form>

          <div className="panel inverted">
            <div className="panel-header">
              <h3>Signal</h3>
              <span className="panel-tag">Latest close vs predicted close</span>
            </div>
            {prediction ? (
              <div className="signal-grid">
                <div>
                  <p className="signal-label">TICKER</p>
                  <p className="signal-value">{prediction.ticker}</p>
                </div>
                <div>
                  <p className="signal-label">LAST CLOSE</p>
                  <p className="signal-value">{formatCurrency(prediction.last_close)}</p>
                </div>
                <div>
                  <p className="signal-label">PREDICTED CLOSE</p>
                  <p className="signal-value">{formatCurrency(prediction.predicted_close)}</p>
                </div>
                <div>
                  <p className="signal-label">DELTA</p>
                  <p className="signal-value">
                    {prediction.delta_pct > 0 ? "+" : ""}
                    {formatNumber(prediction.delta_pct)}%
                  </p>
                </div>
                <div>
                  <p className="signal-label">SIGNAL</p>
                  <p className="signal-pill">{prediction.signal}</p>
                </div>
              </div>
            ) : (
              <p className="panel-empty">Submit a ticker to load the signal.</p>
            )}
          </div>
        </div>
      </section>

      <div className="section-rule" />

      <section className="section">
        <div className="section-header">
          <h2 className="section-title">Market Tape</h2>
          <p className="section-subtitle">Most recent OHLCV history with a close-price sparkline.</p>
        </div>
        <div className="tape">
          <div className="tape-header">
            <div>
              <p className="meta-label">SERIES</p>
              <p className="meta-value">CLOSE PRICE</p>
            </div>
            {ohlcv ? (
              <svg className="sparkline" viewBox="0 0 180 48" aria-hidden="true">
                <polyline points={sparkline} fill="none" stroke="currentColor" strokeWidth="2" />
              </svg>
            ) : (
              <div className="sparkline-placeholder">NO DATA</div>
            )}
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Open</th>
                  <th>High</th>
                  <th>Low</th>
                  <th>Close</th>
                  <th>Volume</th>
                </tr>
              </thead>
              <tbody>
                {ohlcv ? (
                  ohlcv.rows.slice(-20).map((row) => (
                    <tr key={`${row.date}-${row.close}`}>
                      <td className="mono">{row.date}</td>
                      <td>{formatCurrency(row.open)}</td>
                      <td>{formatCurrency(row.high)}</td>
                      <td>{formatCurrency(row.low)}</td>
                      <td>{formatCurrency(row.close)}</td>
                      <td>{formatNumber(row.volume)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} className="table-empty">
                      Run a request to populate the tape.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <div className="section-rule" />

      <section className="section">
        <div className="section-header">
          <h2 className="section-title">System Protocol</h2>
          <p className="section-subtitle">Operational details and API endpoints.</p>
        </div>
        <div className="protocol-grid">
          <div className="protocol-card">
            <p className="protocol-title">GET /predict/{ticker}</p>
            <p className="protocol-copy">Returns next close, delta percent, and BUY/HOLD/SELL signal.</p>
          </div>
          <div className="protocol-card">
            <p className="protocol-title">GET /ohlcv/{ticker}?limit=60</p>
            <p className="protocol-copy">Returns recent OHLCV rows for market tape display.</p>
          </div>
          <div className="protocol-card inverted">
            <p className="protocol-title">LOCAL RUNTIME</p>
            <p className="protocol-copy">Run API: uvicorn api:app --reload</p>
          </div>
        </div>
      </section>

      <footer className="footer">
        <p className="footer-title">Stock Prediction Studio</p>
        <p className="footer-copy">Minimalist monochrome interface for focused market decisions.</p>
      </footer>
    </div>
  );
}
