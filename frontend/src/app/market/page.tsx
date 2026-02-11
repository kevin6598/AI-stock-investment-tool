"use client";

import { useState, useEffect } from "react";
import { api, RetailPrediction, Top10Stock, Top10Response } from "@/lib/api";
import { formatReturn, directionColor, formatPercent } from "@/lib/utils";

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

const MARKET_TABS = ["US", "KR", "ALL"] as const;
type MarketTab = (typeof MARKET_TABS)[number];

export default function MarketPage() {
  const [indexPreds, setIndexPreds] = useState<RetailPrediction[]>([]);
  const [sectorPreds, setSectorPreds] = useState<
    { name: string; prediction: RetailPrediction }[]
  >([]);
  const [loading, setLoading] = useState(true);

  // Top 10 state
  const [activeMarketTab, setActiveMarketTab] = useState<MarketTab>("US");
  const [top10Data, setTop10Data] = useState<Record<string, Top10Response>>({});
  const [top10Loading, setTop10Loading] = useState(false);

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

  // Load Top 10 when tab changes
  useEffect(() => {
    if (top10Data[activeMarketTab]) return;
    setTop10Loading(true);
    api
      .getTop10(activeMarketTab, "1M")
      .then((res) => {
        setTop10Data((prev) => ({ ...prev, [activeMarketTab]: res }));
      })
      .catch(() => {})
      .finally(() => setTop10Loading(false));
  }, [activeMarketTab, top10Data]);

  const currentTop10 = top10Data[activeMarketTab];

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
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-8">
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

      {/* Top 10 Picks */}
      <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
        Top 10 AI Picks
      </h2>

      {/* Market tabs */}
      <div className="flex gap-2 mb-4">
        {MARKET_TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveMarketTab(tab)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activeMarketTab === tab
                ? "bg-blue-600 text-white"
                : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Top 10 cards */}
      {top10Loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700 animate-pulse h-28"
            />
          ))}
        </div>
      ) : currentTop10 && currentTop10.stocks.length > 0 ? (
        <>
          <div className="text-xs text-gray-400 mb-3">
            {currentTop10.total_candidates} candidates analyzed | Pass rate:{" "}
            {(currentTop10.pass_rate * 100).toFixed(1)}%
            {currentTop10.model_version && (
              <> | Model: {currentTop10.model_version}</>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {currentTop10.stocks.map((stock) => (
              <Top10Card key={stock.ticker} stock={stock} />
            ))}
          </div>
        </>
      ) : (
        <div className="text-center text-gray-400 py-8 bg-white dark:bg-card rounded-xl border border-gray-200 dark:border-gray-700">
          No picks available for {activeMarketTab} market
        </div>
      )}
    </div>
  );
}

function Top10Card({ stock }: { stock: Top10Stock }) {
  const isUp = stock.direction === "UP";
  const borderColor = isUp
    ? "border-l-emerald-500"
    : "border-l-rose-500";

  return (
    <div
      className={`bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700 border-l-4 ${borderColor}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="bg-gray-100 dark:bg-gray-800 text-xs font-bold px-2 py-0.5 rounded">
            #{stock.rank}
          </span>
          <span className="font-bold">{stock.ticker}</span>
          <span
            className={`text-xs font-medium ${
              isUp ? "text-emerald-500" : "text-rose-500"
            }`}
          >
            {isUp ? "\u2191" : "\u2193"} {stock.direction}
          </span>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-400">Score</div>
          <div className="font-mono text-sm font-bold">{stock.score.toFixed(3)}</div>
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-4 gap-2 text-xs mb-2">
        <div>
          <div className="text-gray-400">P(Up)</div>
          <div className={`font-mono ${directionColor(stock.direction)}`}>
            {(stock.p_up * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-gray-400">Return</div>
          <div className={`font-mono ${directionColor(stock.direction)}`}>
            {formatReturn(stock.expected_return)}
          </div>
        </div>
        <div>
          <div className="text-gray-400">Confidence</div>
          <div className="font-mono">{(stock.confidence * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-gray-400">Risk</div>
          <div className="font-mono">{(stock.risk_score * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* Allocation bar */}
      <div className="flex items-center gap-2">
        <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded-full h-1.5">
          <div
            className={`h-1.5 rounded-full ${isUp ? "bg-emerald-500" : "bg-rose-500"}`}
            style={{ width: `${Math.min(stock.allocation_weight * 100 * 5, 100)}%` }}
          />
        </div>
        <span className="text-xs text-gray-400 font-mono">
          {(stock.allocation_weight * 100).toFixed(1)}%
        </span>
      </div>

      {/* Reasons */}
      {stock.reasons.length > 0 && (
        <div className="mt-2 text-xs text-gray-400">
          {stock.reasons[0]}
        </div>
      )}
    </div>
  );
}
