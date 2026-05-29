from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURES = ["open", "high", "low", "close", "volume"]


@dataclass
class ZScoreScaler:
	mean: np.ndarray
	std: np.ndarray

	def transform(self, values: np.ndarray) -> np.ndarray:
		return (values - self.mean) / (self.std + 1e-8)

	def inverse_transform(self, values: np.ndarray) -> np.ndarray:
		return (values * (self.std + 1e-8)) + self.mean


class LSTMRegressor(nn.Module):
	def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out, _ = self.lstm(x)
		last = out[:, -1, :]
		return self.fc(last)


class TickerDataset(Dataset):
	def __init__(
		self,
		sequences: np.ndarray,
		targets: np.ndarray,
	) -> None:
		self.sequences = torch.tensor(sequences, dtype=torch.float32)
		self.targets = torch.tensor(targets, dtype=torch.float32)

	def __len__(self) -> int:
		return len(self.sequences)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.sequences[idx], self.targets[idx]


def _safe_ticker_name(ticker: str) -> str:
	return ticker.replace("/", "-").replace(".", "-")


def _load_ticker_dataframe(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	df.columns = [c.strip().lower() for c in df.columns]
	missing = [c for c in FEATURES if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns {missing} in {csv_path}")
	return df


def _build_sequences(values: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
	sequences: List[np.ndarray] = []
	targets: List[float] = []
	for i in range(len(values) - seq_len):
		sequences.append(values[i : i + seq_len])
		targets.append(values[i + seq_len, FEATURES.index("close")])
	return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32).reshape(-1, 1)


def _compute_scalers(values: np.ndarray) -> ZScoreScaler:
	mean = values.mean(axis=0)
	std = values.std(axis=0)
	return ZScoreScaler(mean=mean, std=std)


def _prepare_ticker_data(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray, ZScoreScaler]:
	values = df[FEATURES].to_numpy(dtype=np.float32)
	scaler = _compute_scalers(values)
	normalized = scaler.transform(values)
	sequences, targets = _build_sequences(normalized, seq_len)
	return sequences, targets, scaler


def _iter_ticker_csvs(tickers_dir: Path) -> Iterable[Path]:
	for path in sorted(tickers_dir.glob("*.csv")):
		yield path


def _compute_mae_r2(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
	mae = float(np.mean(np.abs(preds - targets)))
	ss_res = float(np.sum((targets - preds) ** 2))
	ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
	r2 = 0.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
	return mae, r2


def train_model(
	tickers_dir: str | Path = "data/tickers",
	model_path: str | Path = "models/lstm.pt",
	seq_len: int = 60,
	hidden_size: int = 64,
	num_layers: int = 2,
	dropout: float = 0.1,
	epochs: int = 25,
	batch_size: int = 64,
	lr: float = 1e-3,
	device: str | None = None,
) -> Dict[str, float]:
	tickers_dir = Path(tickers_dir)
	model_path = Path(model_path)
	model_path.parent.mkdir(parents=True, exist_ok=True)

	sequences_all: List[np.ndarray] = []
	targets_all: List[np.ndarray] = []
	for csv_path in _iter_ticker_csvs(tickers_dir):
		df = _load_ticker_dataframe(csv_path)
		sequences, targets, _ = _prepare_ticker_data(df, seq_len)
		if len(sequences) == 0:
			continue
		sequences_all.append(sequences)
		targets_all.append(targets)

	if not sequences_all:
		raise ValueError("No ticker data found to train on.")

	sequences_np = np.concatenate(sequences_all, axis=0)
	targets_np = np.concatenate(targets_all, axis=0)

	split_idx = int(len(sequences_np) * 0.8)
	train_ds = TickerDataset(sequences_np[:split_idx], targets_np[:split_idx])
	val_ds = TickerDataset(sequences_np[split_idx:], targets_np[split_idx:])

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

	if device is None:
		device = "cuda"
	if device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA device requested but not available.")
	model = LSTMRegressor(len(FEATURES), hidden_size, num_layers, dropout=dropout).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	loss_fn = nn.MSELoss()

	best_val = float("inf")
	best_mae = float("inf")
	best_r2 = float("-inf")
	for epoch in range(1, epochs + 1):
		model.train()
		for batch_x, batch_y in train_loader:
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			optimizer.zero_grad()
			pred = model(batch_x)
			loss = loss_fn(pred, batch_y)
			loss.backward()
			optimizer.step()

		model.eval()
		val_loss = 0.0
		val_preds: List[np.ndarray] = []
		val_targets: List[np.ndarray] = []
		with torch.no_grad():
			for batch_x, batch_y in val_loader:
				batch_x = batch_x.to(device)
				batch_y = batch_y.to(device)
				pred = model(batch_x)
				val_loss += loss_fn(pred, batch_y).item() * len(batch_x)
				val_preds.append(pred.cpu().numpy())
				val_targets.append(batch_y.cpu().numpy())
		val_loss /= len(val_loader.dataset)
		preds_np = np.concatenate(val_preds, axis=0)
		targets_np = np.concatenate(val_targets, axis=0)
		val_mae, val_r2 = _compute_mae_r2(preds_np, targets_np)
		print(
			f"Epoch {epoch}/{epochs} | "
			f"val_mse={val_loss:.6f} | val_mae={val_mae:.6f} | val_r2={val_r2:.6f}"
		)
		if val_loss < best_val:
			best_val = val_loss
			best_mae = val_mae
			best_r2 = val_r2
			torch.save(model.state_dict(), model_path)

	return {"val_mse": best_val, "val_mae": best_mae, "val_r2": best_r2}


def _load_model(model_path: Path, hidden_size: int, num_layers: int, dropout: float, device: str) -> LSTMRegressor:
	model = LSTMRegressor(len(FEATURES), hidden_size, num_layers, dropout=dropout)
	state = torch.load(model_path, map_location=device)
	model.load_state_dict(state)
	model.to(device)
	model.eval()
	return model


def _save_scaler(scaler: ZScoreScaler, path: Path, ticker: str) -> None:
	payload = {
		"ticker": ticker,
		"mean": scaler.mean.tolist(),
		"std": scaler.std.tolist(),
		"features": FEATURES,
		"saved_at": datetime.utcnow().isoformat() + "Z",
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2))


def _load_scaler(path: Path) -> ZScoreScaler:
	payload = json.loads(path.read_text())
	return ZScoreScaler(mean=np.array(payload["mean"], dtype=np.float32), std=np.array(payload["std"], dtype=np.float32))


def predict_next(
	ticker: str,
	tickers_dir: str | Path = "data/tickers",
	model_path: str | Path = "models/lstm.pt",
	scalers_dir: str | Path = "models/scalers",
	seq_len: int = 90,
	hidden_size: int = 64,
	num_layers: int = 2,
	dropout: float = 0.1,
	threshold: float = 1.5,
	device: str | None = None,
) -> Dict[str, float | str]:
	tickers_dir = Path(tickers_dir)
	model_path = Path(model_path)
	scalers_dir = Path(scalers_dir)
	safe_ticker = _safe_ticker_name(ticker)
	csv_path = tickers_dir / f"{safe_ticker}.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Ticker data not found: {csv_path}")

	df = _load_ticker_dataframe(csv_path)
	if len(df) <= seq_len + 1:
		raise ValueError("Not enough history to build a prediction window.")

	sequences, _, scaler = _prepare_ticker_data(df, seq_len)
	last_sequence = sequences[-1:]

	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model = _load_model(model_path, hidden_size, num_layers, dropout, device)
	with torch.no_grad():
		pred_norm = model(torch.tensor(last_sequence, dtype=torch.float32).to(device)).cpu().numpy()

	close_idx = FEATURES.index("close")
	pred_close = scaler.inverse_transform(
		np.concatenate(
			[np.zeros((1, close_idx), dtype=np.float32), pred_norm, np.zeros((1, len(FEATURES) - close_idx - 1), dtype=np.float32)],
			axis=1,
		)
	)[0, close_idx]

	last_close = float(df["close"].iloc[-1])
	delta_pct = ((pred_close - last_close) / last_close) * 100.0
	if delta_pct > threshold:
		signal = "BUY"
	elif delta_pct < -threshold:
		signal = "SELL"
	else:
		signal = "HOLD"

	scaler_path = scalers_dir / f"{safe_ticker}.json"
	_save_scaler(scaler, scaler_path, ticker)

	return {
		"ticker": ticker,
		"predicted_close": float(pred_close),
		"last_close": last_close,
		"delta_pct": float(delta_pct),
		"signal": signal,
		"scaler_path": str(scaler_path),
	}


def load_prediction_scaler(ticker: str, scalers_dir: str | Path = "models/scalers") -> ZScoreScaler:
	safe_ticker = _safe_ticker_name(ticker)
	scaler_path = Path(scalers_dir) / f"{safe_ticker}.json"
	if not scaler_path.exists():
		raise FileNotFoundError(f"Scaler not found: {scaler_path}")
	return _load_scaler(scaler_path)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train LSTM model and optionally predict next close.")
	parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
	parser.add_argument("--predict", type=str, default="", help="Ticker to predict after training.")
	parser.add_argument("--seq-len", type=int, default=60, help="Sequence length for training/prediction.")
	parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
	parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size.")
	parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
	parser.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout.")
	parser.add_argument("--threshold", type=float, default=1.5, help="Signal threshold percentage.")
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	metrics = train_model(
		epochs=args.epochs,
		seq_len=args.seq_len,
		batch_size=args.batch_size,
		hidden_size=args.hidden_size,
		num_layers=args.num_layers,
		dropout=args.dropout,
	)
	print("Training complete:")
	print(metrics)
	if args.predict:
		result = predict_next(
			args.predict,
			seq_len=args.seq_len,
			hidden_size=args.hidden_size,
			num_layers=args.num_layers,
			dropout=args.dropout,
			threshold=args.threshold,
		)
		print("Prediction:")
		print(result)
