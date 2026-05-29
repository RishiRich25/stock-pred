from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import FEATURES, predict_next, _safe_ticker_name


app = FastAPI(title="Stock Prediction API")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
	allow_credentials=True,
	allow_methods=["*"] ,
	allow_headers=["*"] ,
)


class PredictResponse(BaseModel):
	ticker: str
	predicted_close: float
	last_close: float
	delta_pct: float
	signal: str
	scaler_path: str


class OHLCVRow(BaseModel):
	date: str
	open: float
	high: float
	low: float
	close: float
	volume: float


class OHLCVResponse(BaseModel):
	ticker: str
	rows: List[OHLCVRow]


def _load_ohlcv_rows(ticker: str, tickers_dir: str | Path = "data/tickers", limit: int = 60) -> List[OHLCVRow]:
	safe_ticker = _safe_ticker_name(ticker)
	csv_path = Path(tickers_dir) / f"{safe_ticker}.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Ticker data not found: {csv_path}")

	df = pd.read_csv(csv_path)
	df.columns = [c.strip().lower() for c in df.columns]
	missing = [c for c in FEATURES if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns {missing} in {csv_path}")

	if "date" in df.columns:
		df = df.sort_values("date")
	else:
		df = df.reset_index().rename(columns={"index": "date"})

	rows = []
	for _, row in df.tail(limit).iterrows():
		rows.append(
			OHLCVRow(
				date=str(row["date"]),
				open=float(row["open"]),
				high=float(row["high"]),
				low=float(row["low"]),
				close=float(row["close"]),
				volume=float(row["volume"]),
			)
		)
	return rows


@app.get("/predict/{ticker}", response_model=PredictResponse)
def predict_ticker(ticker: str) -> PredictResponse:
	try:
		result = predict_next(ticker)
		return PredictResponse(**result)
	except FileNotFoundError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/ohlcv/{ticker}", response_model=OHLCVResponse)
def get_ohlcv(ticker: str, limit: int = 60) -> OHLCVResponse:
	try:
		rows = _load_ohlcv_rows(ticker, limit=limit)
		return OHLCVResponse(ticker=ticker, rows=rows)
	except FileNotFoundError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
