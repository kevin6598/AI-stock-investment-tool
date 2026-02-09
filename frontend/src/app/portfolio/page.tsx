"use client";

import { useState } from "react";
import TickerSearch from "@/components/TickerSearch";
import { api, RetailPrediction } from "@/lib/api";
import { formatReturn, directionColor } from "@/lib/utils";

interface Holding {
  ticker: string;
  shares: number;
  prediction?: RetailPrediction;
}

export default function PortfolioPage() {
  const [holdings, setHoldings] = useState<Holding[]>([
    { ticker: "AAPL", shares: 50 },
    { ticker: "MSFT", shares: 30 },
    { ticker: "GOOGL", shares: 20 },
    { ticker: "NVDA", shares: 15 },
  ]);
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  async function analyzePortfolio() {
    setLoading(true);
    const updated = [...holdings];

    for (let i = 0; i < updated.length; i++) {
      try {
        const res = await api.predict(updated[i].ticker, "1M");
        updated[i].prediction = res.prediction;
      } catch {
        // Skip
      }
    }

    setHoldings(updated);
    setAnalyzed(true);
    setLoading(false);
  }

  const totalConfidence = analyzed
    ? holdings.reduce((sum, h) => sum + (h.prediction?.confidence || 0), 0) /
      holdings.length
    : 0;

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Portfolio</h1>
          <p className="text-sm text-gray-400 mt-1">
            AI-optimized allocation and forecast
          </p>
        </div>
        <button
          onClick={analyzePortfolio}
          disabled={loading}
          className="bg-accent text-gray-900 px-4 py-2 rounded-lg text-sm font-medium
                     hover:bg-cyan-300 transition-colors disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Run AI Analysis"}
        </button>
      </div>

      {analyzed && (
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-card rounded-xl p-4 border border-gray-700 text-center">
            <div className="text-xs text-gray-400">Holdings</div>
            <div className="text-2xl font-bold">{holdings.length}</div>
          </div>
          <div className="bg-card rounded-xl p-4 border border-gray-700 text-center">
            <div className="text-xs text-gray-400">Avg Confidence</div>
            <div className="text-2xl font-bold text-accent">
              {(totalConfidence * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-card rounded-xl p-4 border border-gray-700 text-center">
            <div className="text-xs text-gray-400">Bullish / Bearish</div>
            <div className="text-2xl font-bold">
              <span className="text-up">
                {holdings.filter((h) => h.prediction?.direction === "UP").length}
              </span>
              {" / "}
              <span className="text-down">
                {holdings.filter((h) => h.prediction?.direction === "DOWN").length}
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="bg-card rounded-xl border border-gray-700 overflow-hidden">
        <table className="data-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Shares</th>
              <th>AI Signal</th>
              <th>Expected Return</th>
              <th>Confidence</th>
              <th>Risk</th>
            </tr>
          </thead>
          <tbody>
            {holdings.map((h) => (
              <tr key={h.ticker}>
                <td>
                  <a
                    href={`/symbol/${h.ticker}`}
                    className="text-accent hover:underline font-medium"
                  >
                    {h.ticker}
                  </a>
                </td>
                <td className="font-mono">{h.shares}</td>
                <td>
                  {h.prediction ? (
                    <span
                      className={`signal-badge ${
                        h.prediction.direction === "UP"
                          ? "signal-up"
                          : h.prediction.direction === "DOWN"
                          ? "signal-down"
                          : "signal-hold"
                      }`}
                    >
                      {h.prediction.direction}
                    </span>
                  ) : (
                    <span className="text-gray-500">--</span>
                  )}
                </td>
                <td>
                  {h.prediction ? (
                    <span
                      className={`font-mono ${directionColor(
                        h.prediction.direction
                      )}`}
                    >
                      {formatReturn(h.prediction.point_estimate)}
                    </span>
                  ) : (
                    "--"
                  )}
                </td>
                <td>
                  {h.prediction ? (
                    <span className="font-mono">
                      {(h.prediction.confidence * 100).toFixed(1)}%
                    </span>
                  ) : (
                    "--"
                  )}
                </td>
                <td>
                  {h.prediction ? (
                    <div className="w-16">
                      <div className="gauge-track">
                        <div
                          className="gauge-fill bg-rose-500"
                          style={{
                            width: `${h.prediction.risk_score * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  ) : (
                    "--"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
