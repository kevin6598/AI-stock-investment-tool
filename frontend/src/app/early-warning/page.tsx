"use client";

import { useState, useEffect } from "react";
import { api, EarlyWarningResponse, WarningSignal } from "@/lib/api";

const LEVEL_COLORS: Record<string, string> = {
  HEALTHY: "text-emerald-400",
  CAUTION: "text-yellow-400",
  WARNING: "text-orange-400",
  DANGER: "text-rose-400",
  CRITICAL: "text-red-600",
};

export default function EarlyWarningPage() {
  const [data, setData] = useState<EarlyWarningResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getEarlyWarning()
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500 dark:text-gray-400">
          Computing early warning signals...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-700 dark:text-red-400">Error: {error}</p>
        <p className="text-sm text-red-500 mt-2">
          Make sure the FastAPI backend is running on port 8000.
        </p>
      </div>
    );
  }

  if (!data) return null;

  const levelColor = LEVEL_COLORS[data.level] || "text-gray-400";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Early Warning System</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          6-dimensional failure precursor detection for strategy structural health
        </p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 text-center">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
            Warning Score
          </div>
          <div className={`text-4xl font-bold font-mono ${levelColor}`}>
            {(data.warning_score * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 text-center">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
            Level
          </div>
          <div className={`text-4xl font-bold ${levelColor}`}>
            {data.level}
          </div>
        </div>
        <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 text-center">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
            Exposure Multiplier
          </div>
          <div className="text-4xl font-bold font-mono text-accent">
            {(data.exposure_multiplier * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Signal Details */}
      <div className="space-y-4">
        <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400">
          Signal Breakdown
        </h2>
        {data.signals.map((signal) => (
          <SignalDetailCard key={signal.name} signal={signal} />
        ))}
      </div>

      <div className="mt-6 text-xs text-gray-500 dark:text-gray-400">
        Computed at: {data.timestamp} | Strategy: {data.strategy_id}
      </div>
    </div>
  );
}

function SignalDetailCard({ signal }: { signal: WarningSignal }) {
  const pct = Math.round(signal.score * 100);
  let barColor = "bg-emerald-400";
  let statusLabel = "Normal";
  if (signal.score >= 0.6) {
    barColor = "bg-rose-500";
    statusLabel = "Alert";
  } else if (signal.score >= 0.3) {
    barColor = "bg-yellow-400";
    statusLabel = "Elevated";
  }

  const displayName = signal.name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (l) => l.toUpperCase());

  return (
    <div className="bg-white dark:bg-card rounded-xl p-5 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <h3 className="font-medium">{displayName}</h3>
          <span
            className={`text-xs px-2 py-0.5 rounded-full ${
              signal.score >= 0.6
                ? "bg-rose-500/10 text-rose-500"
                : signal.score >= 0.3
                ? "bg-yellow-400/10 text-yellow-400"
                : "bg-emerald-400/10 text-emerald-400"
            }`}
          >
            {statusLabel}
          </span>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500">
            Weight: {(signal.weight * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500">
            Contribution: {(signal.weighted_contribution * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="mb-3">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-gray-500">Signal Score</span>
          <span className="font-mono">{pct}%</span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full transition-all ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <p className="text-xs text-gray-400 leading-relaxed">{signal.detail}</p>

      {signal.raw_value !== null && signal.raw_value !== 0 && (
        <div className="mt-2 text-xs text-gray-500">
          Raw value: <span className="font-mono">{signal.raw_value.toFixed(4)}</span>
        </div>
      )}
    </div>
  );
}
