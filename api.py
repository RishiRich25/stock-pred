from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import load_recent_market_data, predict_next


app = FastAPI(title="Stock Prediction API")
app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		"http://localhost:5173",
		"http://127.0.0.1:5173",
		"http://localhost:5174",
		"http://127.0.0.1:5174",
	],
	allow_credentials=True,
	allow_methods=["*"] ,
	allow_headers=["*"] ,
)


class PredictResponse(BaseModel):
	ticker: str
	predicted_open: float
	predicted_high: float
	predicted_low: float
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



def _load_ohlcv_rows(ticker: str, limit: int = 21) -> List[OHLCVRow]:
	df = load_recent_market_data(ticker, limit=limit)

	rows = []
	for _, row in df.iterrows():
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
def get_ohlcv(ticker: str, limit: int = 21) -> OHLCVResponse:
	try:
		rows = _load_ohlcv_rows(ticker, limit=limit)
		return OHLCVResponse(ticker=ticker, rows=rows)
	except FileNotFoundError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
