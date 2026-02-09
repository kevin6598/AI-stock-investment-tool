"use client";

import { useState, useEffect } from "react";
import { api, RetailPrediction } from "@/lib/api";
import { formatReturn, directionColor } from "@/lib/utils";

const INDICES = ["SPY", "QQQ", "DIA", "IWM"];
const SECTORS = [
  { ticker: "XLK", name: "Technology" },
  { ticker: "XLF", name: "Financials" },
  { ticker: "XLE", name: "Energy" },
  { ticker: "XLV", name: "Healthcare" },
  { ticker: "XLI", name: "Industrials" },
  { ticker: "XLP", name: "Consumer Staples" },
  { ticker: "XLY", name: "Consumer Disc." },
  { ticker: "XLU", name: "Utilities" },
];

export default function MarketPage() {
  const [indexPreds, setIndexPreds] = useState<RetailPrediction[]>([]);
  const [sectorPreds, setSectorPreds] = useState<
    { name: string; prediction: RetailPrediction }[]
  >([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      // Index predictions
      const idx: RetailPrediction[] = [];
      for (const t of INDICES) {
        try {
          const res = await api.predict(t, "1M");
          idx.push(res.prediction);
        } catch {
          // skip
        }
      }
      setIndexPreds(idx);

      // Sector predictions
      const sec: { name: string; prediction: RetailPrediction }[] = [];
      for (const s of SECTORS) {
        try {
          const res = await api.predict(s.ticker, "1M");
          sec.push({ name: s.name, prediction: res.prediction });
        } catch {
          // skip
        }
      }
      setSectorPreds(sec);
      setLoading(false);
    }
    load();
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Market Overview</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
        AI probability forecasts for major indices and sectors
      </p>

      {/* Index probabilities */}
      <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
        Index Probabilities (1M)
      </h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {loading
          ? [1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700 animate-pulse h-24"
              />
            ))
          : indexPreds.map((p) => (
              <div
                key={p.ticker}
                className="bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold">{p.ticker}</span>
                  <span
                    className={`signal-badge ${
                      p.direction === "UP"
                        ? "signal-up"
                        : p.direction === "DOWN"
                        ? "signal-down"
                        : "signal-hold"
                    }`}
                  >
                    {p.direction}
                  </span>
                </div>
                <div className={`text-lg font-mono ${directionColor(p.direction)}`}>
                  P(Up) = {(p.p_up * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {formatReturn(p.point_estimate)} expected
                </div>
              </div>
            ))}
      </div>

      {/* Sector heatmap */}
      <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
        Sector Forecast Heatmap
      </h2>
      {loading ? (
        <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 animate-pulse h-48" />
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {sectorPreds.map((s) => {
            const p = s.prediction;
            const intensity = Math.min(Math.abs(p.point_estimate) * 500, 1);
            const bg =
              p.point_estimate >= 0
                ? `rgba(52, 211, 153, ${intensity * 0.3})`
                : `rgba(244, 63, 94, ${intensity * 0.3})`;
            return (
              <div
                key={p.ticker}
                className="rounded-lg p-3 border border-gray-200 dark:border-gray-700 text-center"
                style={{ backgroundColor: bg }}
              >
                <div className="text-xs text-gray-500 dark:text-gray-400">{s.name}</div>
                <div className="font-bold text-sm">{p.ticker}</div>
                <div
                  className={`text-sm font-mono ${directionColor(p.direction)}`}
                >
                  {formatReturn(p.point_estimate)}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
