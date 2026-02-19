"use client";

import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import { api, HealthResponse, StrategyStatusResponse } from "@/lib/api";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [strategy, setStrategy] = useState<StrategyStatusResponse | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    api.getHealth().then(setHealth).catch(() => {});
    api.getStrategyStatus().then(setStrategy).catch(() => {});
  }, []);

  if (!mounted) return null;

  return (
    <div className="max-w-2xl">
      <h1 className="text-2xl font-bold mb-2">Settings</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
        System configuration and status
      </p>

      {/* Theme */}
      <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
        <h2 className="text-sm font-medium mb-4">Appearance</h2>
        <div className="flex gap-3">
          {["dark", "light", "system"].map((t) => (
            <button
              key={t}
              onClick={() => setTheme(t)}
              className={`px-4 py-2 rounded-lg text-sm capitalize transition-colors ${
                theme === t
                  ? "bg-accent text-gray-900 font-medium"
                  : "bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Strategy info */}
      <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
        <h2 className="text-sm font-medium mb-4">Strategy Information</h2>
        {strategy ? (
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Strategy ID</span>
              <span className="font-mono text-xs">{strategy.strategy_id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Version</span>
              <span className="font-mono">{strategy.version}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Market</span>
              <span className="font-mono">{strategy.market}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Horizon</span>
              <span className="font-mono">{strategy.horizon_days} days</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Type</span>
              <span className="font-mono">{strategy.type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Trust Score</span>
              <span className="font-mono">
                {(strategy.governance.trust_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Unable to load strategy data. Start the FastAPI backend first.
          </p>
        )}
      </div>

      {/* Model info */}
      <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
        <h2 className="text-sm font-medium mb-4">Model Information</h2>
        {health?.model_info ? (
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Type</span>
              <span className="font-mono">{health.model_info.model_type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Version</span>
              <span className="font-mono">
                {health.model_info.version || "N/A"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Features</span>
              <span className="font-mono">{health.model_info.n_features}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Trained Tickers</span>
              <span className="font-mono">
                {health.model_info.trained_tickers.length}
              </span>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No model loaded.
          </p>
        )}
      </div>

      {/* API Status */}
      <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
        <h2 className="text-sm font-medium mb-4">API Status</h2>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">Status</span>
            <span className="flex items-center gap-1.5">
              <div
                className={`w-2 h-2 rounded-full ${
                  health?.status === "healthy" ? "bg-up" : "bg-down"
                }`}
              />
              {health?.status || "Offline"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">Uptime</span>
            <span className="font-mono">
              {health
                ? `${Math.floor(health.uptime_seconds / 60)}m`
                : "N/A"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">Endpoint</span>
            <span className="font-mono text-xs">
              {process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}
            </span>
          </div>
        </div>
      </div>

      {/* About */}
      <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <h2 className="text-sm font-medium mb-4">About</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Strategy Governance Platform v2.0. Built on a V4 quantitative
          pipeline that filters 37,322 candidates through 16 stages to
          identify 1 surviving strategy. The platform monitors strategy
          structural health through 6-dimensional early warning detection
          and translates warning scores into capital exposure guidance.
        </p>
      </div>
    </div>
  );
}
