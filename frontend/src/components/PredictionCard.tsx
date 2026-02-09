"use client";

import { RetailPrediction } from "@/lib/api";
import { formatReturn, directionColor, confidenceLabel, riskLabel } from "@/lib/utils";

interface Props {
  prediction: RetailPrediction;
}

export default function PredictionCard({ prediction }: Props) {
  const p = prediction;
  const dirClass = directionColor(p.direction);

  return (
    <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold">{p.ticker}</h2>
          <span className="text-sm text-gray-500 dark:text-gray-400">{p.horizon} forecast</span>
        </div>
        <div className="text-right">
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
          {p.is_zero_shot && (
            <span className="ml-2 text-xs text-yellow-400">(zero-shot)</span>
          )}
        </div>
      </div>

      {/* Main signal */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Expected Return</div>
          <div className={`text-xl font-mono font-bold ${dirClass}`}>
            {formatReturn(p.point_estimate)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">P(Up)</div>
          <div className="text-xl font-mono font-bold text-accent">
            {(p.p_up * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Regime</div>
          <div className="text-sm font-medium capitalize">{p.regime}</div>
        </div>
      </div>

      {/* Confidence & Risk gauges */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500 dark:text-gray-400">Confidence</span>
            <span>{confidenceLabel(p.confidence)}</span>
          </div>
          <div className="gauge-track">
            <div
              className="gauge-fill bg-accent"
              style={{ width: `${p.confidence * 100}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500 dark:text-gray-400">Risk</span>
            <span>{riskLabel(p.risk_score)}</span>
          </div>
          <div className="gauge-track">
            <div
              className="gauge-fill bg-rose-500"
              style={{ width: `${p.risk_score * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Quantile forecast */}
      <div>
        <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">Return Distribution</div>
        <div className="flex items-end justify-between h-16 gap-1">
          {[
            { label: "5%", value: p.quantiles.q05 },
            { label: "10%", value: p.quantiles.q10 },
            { label: "25%", value: p.quantiles.q25 },
            { label: "50%", value: p.quantiles.q50 },
            { label: "75%", value: p.quantiles.q75 },
            { label: "90%", value: p.quantiles.q90 },
            { label: "95%", value: p.quantiles.q95 },
          ].map((q) => (
            <div key={q.label} className="flex-1 text-center">
              <div
                className={`text-[10px] font-mono ${
                  q.value >= 0 ? "text-up" : "text-down"
                }`}
              >
                {formatReturn(q.value)}
              </div>
              <div className="text-[9px] text-gray-400 dark:text-gray-500">{q.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
