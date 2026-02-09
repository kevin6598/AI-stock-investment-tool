"use client";

import { useState, useEffect } from "react";
import TickerSearch from "@/components/TickerSearch";
import PredictionCard from "@/components/PredictionCard";
import { api, RetailPrediction, HealthResponse } from "@/lib/api";

const WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM"];

export default function Dashboard() {
  const [predictions, setPredictions] = useState<RetailPrediction[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const h = await api.getHealth();
        setHealth(h);
      } catch {
        // API not running
      }

      const results: RetailPrediction[] = [];
      for (const ticker of WATCHLIST.slice(0, 4)) {
        try {
          const res = await api.predict(ticker, "1M");
          results.push(res.prediction);
        } catch {
          // Skip failed predictions
        }
      }
      setPredictions(results);
      setLoading(false);
    }
    load();
  }, []);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-gray-400 mt-1">
            AI-powered stock predictions with multi-modal deep learning
          </p>
        </div>
        <TickerSearch />
      </div>

      {/* Status bar */}
      <div className="flex items-center gap-4 mb-6 text-xs">
        <div className="flex items-center gap-1.5">
          <div
            className={`w-2 h-2 rounded-full ${
              health?.status === "healthy" ? "bg-up" : "bg-down"
            }`}
          />
          <span className="text-gray-400">
            API: {health?.status || "offline"}
          </span>
        </div>
        {health?.model_loaded && (
          <>
            <span className="text-gray-600">|</span>
            <span className="text-gray-400">
              Model: {health.model_info?.model_type || "unknown"}
            </span>
            <span className="text-gray-600">|</span>
            <span className="text-gray-400">
              Tickers: {health.model_info?.trained_tickers.length || 0}
            </span>
          </>
        )}
      </div>

      {/* Watchlist */}
      <div className="mb-6">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Watchlist</h2>
        <div className="flex flex-wrap gap-2">
          {WATCHLIST.map((t) => (
            <a
              key={t}
              href={`/symbol/${t}`}
              className="px-3 py-1.5 bg-card rounded-lg text-sm hover:bg-gray-700
                         transition-colors border border-gray-700"
            >
              {t}
            </a>
          ))}
        </div>
      </div>

      {/* Predictions grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="bg-card rounded-xl p-6 border border-gray-700 animate-pulse"
            >
              <div className="h-6 bg-gray-700 rounded w-1/4 mb-4" />
              <div className="h-4 bg-gray-700 rounded w-1/2 mb-2" />
              <div className="h-4 bg-gray-700 rounded w-2/3" />
            </div>
          ))}
        </div>
      ) : predictions.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {predictions.map((p) => (
            <a key={p.ticker} href={`/symbol/${p.ticker}`}>
              <PredictionCard prediction={p} />
            </a>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-gray-400">
          <p className="text-lg mb-2">No predictions available</p>
          <p className="text-sm">
            Make sure the FastAPI backend is running on port 8000
          </p>
        </div>
      )}
    </div>
  );
}
