"use client";

import { useState, useEffect } from "react";
import {
  api,
  RetailPrediction,
  EarlyWarningResponse,
} from "@/lib/api";
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

export default function MarketRegimePage() {
  const [indexPreds, setIndexPreds] = useState<RetailPrediction[]>([]);
  const [sectorPreds, setSectorPreds] = useState<
    { name: string; prediction: RetailPrediction }[]
  >([]);
  const [warning, setWarning] = useState<EarlyWarningResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      // Load early warning and market data in parallel
      const warningPromise = api.getEarlyWarning().catch(() => null);

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

      const w = await warningPromise;
      if (w) setWarning(w);

      setLoading(false);
    }
    load();
  }, []);

  // Extract volatility signal from early warning
  const volSignal = warning?.signals.find(
    (s) => s.name === "volatility_regime"
  );
  const contagionSignal = warning?.signals.find(
    (s) => s.name === "cross_market_contagion"
  );
  const sectorSignal = warning?.signals.find(
    (s) => s.name === "sector_concentration"
  );

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Market Regime</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
        Market conditions that affect strategy structural health
      </p>

      {/* Regime Indicators from Early Warning */}
      {warning && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <RegimeCard
            label="Volatility Regime"
            signal={volSignal}
            fallbackDetail="VIX data not available"
          />
          <RegimeCard
            label="Cross-Market Contagion"
            signal={contagionSignal}
            fallbackDetail="No contagion data"
          />
          <RegimeCard
            label="Sector Concentration"
            signal={sectorSignal}
            fallbackDetail="No sector data"
          />
        </div>
      )}

      {/* Index context */}
      <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
        Index Context (1M)
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
                <div
                  className={`text-lg font-mono ${directionColor(p.direction)}`}
                >
                  {formatReturn(p.point_estimate)}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Regime: {p.regime}
                </div>
              </div>
            ))}
      </div>

      {/* Sector heatmap */}
      <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
        Sector Landscape
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
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {s.name}
                </div>
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

      {/* Strategy Impact Note */}
      <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
          How Market Regime Affects the Strategy
        </h2>
        <div className="space-y-3 text-xs text-gray-400 leading-relaxed">
          <p>
            The surviving strategy (63d horizon, momentum d4 + 52w-low d0) is a
            contrarian value play. It performs best in <span className="text-accent">normal-to-moderate
            volatility</span> regimes where mean reversion operates reliably.
          </p>
          <p>
            <span className="text-rose-400 font-medium">Danger conditions:</span>{" "}
            VIX {">"} 2x its 252-day MA, synchronized global sell-offs (US + KR both
            down {">"} 3%), or sector concentration above 60% in signal stocks.
          </p>
          <p>
            The early warning system monitors these conditions and automatically
            adjusts exposure recommendations. No manual regime classification needed.
          </p>
        </div>
      </div>
    </div>
  );
}

function RegimeCard({
  label,
  signal,
  fallbackDetail,
}: {
  label: string;
  signal?: { score: number; detail: string; raw_value: number | null };
  fallbackDetail: string;
}) {
  const score = signal?.score ?? 0;
  const pct = Math.round(score * 100);
  let barColor = "bg-emerald-400";
  let status = "Normal";
  if (score >= 0.6) {
    barColor = "bg-rose-500";
    status = "Alert";
  } else if (score >= 0.3) {
    barColor = "bg-yellow-400";
    status = "Elevated";
  }

  return (
    <div className="bg-white dark:bg-card rounded-xl p-5 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium">{label}</h3>
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${
            score >= 0.6
              ? "bg-rose-500/10 text-rose-500"
              : score >= 0.3
              ? "bg-yellow-400/10 text-yellow-400"
              : "bg-emerald-400/10 text-emerald-400"
          }`}
        >
          {status}
        </span>
      </div>
      <div className="mb-2">
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
      <p className="text-xs text-gray-400 leading-relaxed">
        {signal?.detail || fallbackDetail}
      </p>
    </div>
  );
}
