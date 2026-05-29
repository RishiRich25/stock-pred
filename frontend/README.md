# Stock Prediction Frontend

Minimalist monochrome React UI for the FastAPI stock prediction service.

## Setup

1. Install dependencies:
   npm install
2. Start dev server:
   npm run dev
3. Configure API base URL (optional):
   Copy .env.example to .env and set VITE_API_BASE.

## API Expectations

- GET /predict/{ticker}
- GET /ohlcv/{ticker}?limit=60
