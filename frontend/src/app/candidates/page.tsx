"use client";

import { useState, useEffect } from "react";
import TickerSearch from "@/components/TickerSearch";
import {
  api,
  StrategyCandidateStock,
  StrategyCandidatesResponse,
  ExposureGuidanceResponse,
} from "@/lib/api";
import { formatReturn, directionColor } from "@/lib/utils";

export default function CandidatesPage() {
  const [data, setData] = useState<StrategyCandidatesResponse | null>(null);
  const [exposure, setExposure] = useState<ExposureGuidanceResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const results = await Promise.allSettled([
        api.getStrategyCandidates("3M"),
        api.getExposureGuidance(),
      ]);
      if (results[0].status === "fulfilled") setData(results[0].value);
      if (results[1].status === "fulfilled") setExposure(results[1].value);
      setLoading(false);
    }
    load();
  }, []);

  const mult = exposure?.exposure_multiplier ?? 1;

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Signal Candidates</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Stocks matching the strategy signal, filtered by early warning exposure
          </p>
        </div>
        <TickerSearch />
      </div>

      {/* Strategy header banner */}
      {data && (
        <div className="bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-xl p-4 border border-indigo-200 dark:border-indigo-800 mb-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Active Strategy</div>
              <div className="font-mono text-sm font-bold text-indigo-600 dark:text-indigo-400">
                {data.strategy_name}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 font-mono">
                {data.strategy_id}
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500">Signal Match</div>
              <div className="font-mono text-lg font-bold">
                {data.signal_matches}
                <span className="text-xs text-gray-400 font-normal">
                  /{data.universe_size}
                </span>
              </div>
              <div className="text-xs text-gray-400">
                {data.universe_size > 0
                  ? ((data.signal_matches / data.universe_size) * 100).toFixed(1)
                  : "0.0"}
                % pass rate
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Exposure context */}
      {exposure && (
        <div className="bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700 mb-6 flex items-center justify-between">
          <div className="text-sm">
            <span className="text-gray-500">Current exposure multiplier: </span>
            <span className="font-mono font-bold text-accent">
              {(mult * 100).toFixed(0)}%
            </span>
            <span className="text-gray-500 ml-3">
              ({exposure.warning_level})
            </span>
          </div>
          <div className="text-xs text-gray-500">
            Weights below are pre-adjustment. Multiply by {(mult * 100).toFixed(0)}% for actual allocation.
          </div>
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700 animate-pulse h-28"
            />
          ))}
        </div>
      ) : data && data.stocks.length > 0 ? (
        <>
          <div className="text-xs text-gray-400 mb-3">
            {data.signal_matches} signal matches | ML pass rate:{" "}
            {(data.pass_rate * 100).toFixed(1)}% | Horizon: {data.horizon}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.stocks.map((stock) => (
              <CandidateCard
                key={stock.ticker}
                stock={stock}
                exposureMult={mult}
              />
            ))}
          </div>
        </>
      ) : (
        <div className="text-center text-gray-400 py-12 bg-white dark:bg-card rounded-xl border border-gray-200 dark:border-gray-700">
          <p className="text-lg mb-2">No candidates available</p>
          <p className="text-sm">
            The strategy signal is not currently producing actionable candidates,
            or the API is not running.
          </p>
        </div>
      )}
    </div>
  );
}

function CandidateCard({
  stock,
  exposureMult,
}: {
  stock: StrategyCandidateStock;
  exposureMult: number;
}) {
  const isUp = stock.direction === "UP";
  const borderColor = isUp ? "border-l-emerald-500" : "border-l-rose-500";
  const adjustedWeight = stock.allocation_weight * exposureMult;

  return (
    <div
      className={`bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700 border-l-4 ${borderColor}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="bg-gray-100 dark:bg-gray-800 text-xs font-bold px-2 py-0.5 rounded">
            #{stock.rank}
          </span>
          <a
            href={`/symbol/${stock.ticker}`}
            className="font-bold text-accent hover:underline"
          >
            {stock.ticker}
          </a>
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
          <div className="font-mono text-sm font-bold">
            {stock.score.toFixed(3)}
          </div>
        </div>
      </div>

      {/* Strategy signal features */}
      {stock.mom_60d != null && stock.high_52w_pct != null && (
        <div className="flex gap-3 mb-2 text-xs">
          <div className="bg-indigo-50 dark:bg-indigo-900/30 rounded px-2 py-1">
            <span className="text-gray-500">mom_60d </span>
            <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400">
              d{stock.mom_60d_decile}
            </span>
            <span className="text-gray-400 ml-1">
              ({(stock.mom_60d * 100).toFixed(1)}%)
            </span>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/30 rounded px-2 py-1">
            <span className="text-gray-500">52w_pct </span>
            <span className="font-mono font-bold text-purple-600 dark:text-purple-400">
              d{stock.high_52w_pct_decile}
            </span>
            <span className="text-gray-400 ml-1">
              ({(stock.high_52w_pct * 100).toFixed(1)}%)
            </span>
          </div>
        </div>
      )}

      {/* Metrics */}
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

      {/* Allocation bars */}
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-500 w-12">Raw</span>
          <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded-full h-1.5">
            <div
              className={`h-1.5 rounded-full ${isUp ? "bg-emerald-500/50" : "bg-rose-500/50"}`}
              style={{
                width: `${Math.min(stock.allocation_weight * 100 * 5, 100)}%`,
              }}
            />
          </div>
          <span className="text-[10px] text-gray-400 font-mono w-12 text-right">
            {(stock.allocation_weight * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-accent w-12">Adj.</span>
          <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded-full h-1.5">
            <div
              className={`h-1.5 rounded-full ${isUp ? "bg-emerald-500" : "bg-rose-500"}`}
              style={{
                width: `${Math.min(adjustedWeight * 100 * 5, 100)}%`,
              }}
            />
          </div>
          <span className="text-[10px] text-accent font-mono w-12 text-right">
            {(adjustedWeight * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}
