"use client";

import { useState, useEffect } from "react";
import {
  api,
  ExposureGuidanceResponse,
  EarlyWarningResponse,
  StrategyStatusResponse,
} from "@/lib/api";

const LEVEL_COLORS: Record<string, string> = {
  HEALTHY: "text-emerald-400",
  CAUTION: "text-yellow-400",
  WARNING: "text-orange-400",
  DANGER: "text-rose-400",
  CRITICAL: "text-red-600",
};

export default function ExposurePage() {
  const [exposure, setExposure] = useState<ExposureGuidanceResponse | null>(null);
  const [warning, setWarning] = useState<EarlyWarningResponse | null>(null);
  const [strategy, setStrategy] = useState<StrategyStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const results = await Promise.allSettled([
        api.getExposureGuidance(),
        api.getEarlyWarning(),
        api.getStrategyStatus(),
      ]);
      if (results[0].status === "fulfilled") setExposure(results[0].value);
      if (results[1].status === "fulfilled") setWarning(results[1].value);
      if (results[2].status === "fulfilled") setStrategy(results[2].value);
      setLoading(false);
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500 dark:text-gray-400">Loading exposure guidance...</div>
      </div>
    );
  }

  const level = exposure?.warning_level || "HEALTHY";
  const levelColor = LEVEL_COLORS[level] || "text-gray-400";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Capital Exposure Guidance</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          How much capital to allocate based on strategy health
        </p>
      </div>

      {exposure && (
        <>
          {/* Main Exposure Display */}
          <div className="bg-white dark:bg-card rounded-xl p-8 border border-gray-200 dark:border-gray-700 mb-6 text-center">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
              Recommended Exposure
            </div>
            <div className={`text-6xl font-bold font-mono ${levelColor}`}>
              {(exposure.exposure_multiplier * 100).toFixed(0)}%
            </div>
            <div className={`text-lg font-medium mt-2 ${levelColor}`}>
              {exposure.recommended_action}
            </div>
            <div className="text-sm text-gray-400 mt-4 max-w-lg mx-auto leading-relaxed">
              {exposure.position_guidance}
            </div>
          </div>

          {/* Exposure Schedule Visualization */}
          <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
            <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">
              Exposure Schedule
            </h2>
            <div className="space-y-3">
              {[
                { range: "0% - 20%", level: "HEALTHY", exposure: "100%", color: "bg-emerald-400" },
                { range: "20% - 40%", level: "CAUTION", exposure: "75%", color: "bg-yellow-400" },
                { range: "40% - 60%", level: "WARNING", exposure: "50%", color: "bg-orange-400" },
                { range: "60% - 80%", level: "DANGER", exposure: "25%", color: "bg-rose-400" },
                { range: "80% - 100%", level: "CRITICAL", exposure: "0%", color: "bg-red-600" },
              ].map((row) => {
                const isActive = row.level === level;
                return (
                  <div
                    key={row.level}
                    className={`flex items-center gap-4 p-3 rounded-lg transition-colors ${
                      isActive
                        ? "bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600"
                        : ""
                    }`}
                  >
                    <div className={`w-3 h-3 rounded-full ${row.color}`} />
                    <div className="flex-1 text-sm">
                      <span className="font-medium">{row.level}</span>
                      <span className="text-gray-500 ml-2">
                        Warning Score: {row.range}
                      </span>
                    </div>
                    <div className="text-sm font-mono font-medium">
                      {row.exposure} exposure
                    </div>
                    {isActive && (
                      <span className="text-xs text-accent font-medium">CURRENT</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Position Sizing Example */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">
                Position Sizing Example
              </h2>
              <div className="space-y-3 text-sm">
                {[10000, 50000, 100000, 500000].map((capital) => {
                  const adjusted = capital * (exposure.exposure_multiplier);
                  const cash = capital - adjusted;
                  return (
                    <div key={capital} className="flex justify-between">
                      <span className="text-gray-500">
                        ${capital.toLocaleString()} portfolio
                      </span>
                      <span>
                        <span className="font-mono text-accent">
                          ${adjusted.toLocaleString()}
                        </span>
                        {" invested, "}
                        <span className="font-mono text-gray-500">
                          ${cash.toLocaleString()}
                        </span>
                        {" cash"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">
                Strategy Context
              </h2>
              {strategy && (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Horizon</span>
                    <span className="font-mono">{strategy.horizon_days} days</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Backtest Sharpe</span>
                    <span className="font-mono">{strategy.backtest.sharpe.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Win Rate</span>
                    <span className="font-mono">
                      {(strategy.backtest.win_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Pipeline Trust</span>
                    <span className="font-mono">
                      {(strategy.governance.trust_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="border-t border-gray-200 dark:border-gray-700 pt-2 mt-2">
                    <div className="text-xs text-gray-400">
                      Exposure = Warning-adjusted allocation. Cash portion earns
                      risk-free rate. Re-entry when warning score drops below 40%.
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 text-xs text-gray-500 dark:text-gray-400">
            Computed at: {exposure.timestamp}
          </div>
        </>
      )}
    </div>
  );
}
