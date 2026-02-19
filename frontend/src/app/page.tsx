"use client";

import { useState, useEffect } from "react";
import {
  api,
  StrategyStatusResponse,
  EarlyWarningResponse,
  ExposureGuidanceResponse,
  HealthResponse,
} from "@/lib/api";

const LEVEL_COLORS: Record<string, string> = {
  HEALTHY: "text-emerald-400",
  CAUTION: "text-yellow-400",
  WARNING: "text-orange-400",
  DANGER: "text-rose-400",
  CRITICAL: "text-red-600",
};

const LEVEL_BG: Record<string, string> = {
  HEALTHY: "bg-emerald-400/10 border-emerald-400/30",
  CAUTION: "bg-yellow-400/10 border-yellow-400/30",
  WARNING: "bg-orange-400/10 border-orange-400/30",
  DANGER: "bg-rose-400/10 border-rose-400/30",
  CRITICAL: "bg-red-600/10 border-red-600/30",
};

export default function Dashboard() {
  const [strategy, setStrategy] = useState<StrategyStatusResponse | null>(null);
  const [warning, setWarning] = useState<EarlyWarningResponse | null>(null);
  const [exposure, setExposure] = useState<ExposureGuidanceResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const results = await Promise.allSettled([
        api.getStrategyStatus(),
        api.getEarlyWarning(),
        api.getExposureGuidance(),
        api.getHealth(),
      ]);

      if (results[0].status === "fulfilled") setStrategy(results[0].value);
      if (results[1].status === "fulfilled") setWarning(results[1].value);
      if (results[2].status === "fulfilled") setExposure(results[2].value);
      if (results[3].status === "fulfilled") setHealth(results[3].value);
      setLoading(false);
    }
    load();
  }, []);

  const warningLevel = warning?.level || "HEALTHY";
  const levelColor = LEVEL_COLORS[warningLevel] || "text-gray-400";
  const levelBg = LEVEL_BG[warningLevel] || "";

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Strategy Governance Dashboard</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Survivability monitoring for the sole surviving strategy
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div
              className={`w-2 h-2 rounded-full ${
                health?.status === "healthy" ? "bg-up" : "bg-down"
              }`}
            />
            <span className="text-xs text-gray-500 dark:text-gray-400">
              API: {health?.status || "offline"}
            </span>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 animate-pulse h-32"
            />
          ))}
        </div>
      ) : (
        <>
          {/* Warning Level Banner */}
          <div
            className={`rounded-xl p-6 border mb-6 ${levelBg}`}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  System Status
                </div>
                <div className={`text-3xl font-bold ${levelColor}`}>
                  {warningLevel}
                </div>
                <div className="text-sm text-gray-400 mt-1">
                  {exposure?.recommended_action || "Loading..."}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  Warning Score
                </div>
                <div className={`text-4xl font-bold font-mono ${levelColor}`}>
                  {warning ? (warning.warning_score * 100).toFixed(0) : "--"}
                  <span className="text-lg">%</span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Exposure: {warning ? `${(warning.exposure_multiplier * 100).toFixed(0)}%` : "--"}
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Strategy Identity */}
            <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">
                Strategy Identity
              </h2>
              {strategy ? (
                <div className="space-y-3">
                  <div>
                    <div className="text-xs text-gray-500">ID</div>
                    <div className="text-xs font-mono break-all">{strategy.strategy_id}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="text-xs text-gray-500">Market</div>
                      <div className="font-medium">{strategy.market}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Horizon</div>
                      <div className="font-medium">{strategy.horizon_days}d</div>
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Signal</div>
                    <div className="text-sm">
                      <span className="font-mono text-accent">{strategy.signal.feature_1}</span>
                      {" d"}{strategy.signal.feature_1_decile}
                      {" "}{strategy.signal.logic}{" "}
                      <span className="font-mono text-accent">{strategy.signal.feature_2}</span>
                      {" d"}{strategy.signal.feature_2_decile}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Thesis</div>
                    <div className="text-xs text-gray-400 leading-relaxed">
                      {strategy.thesis}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-gray-500">Unable to load strategy data</p>
              )}
            </div>

            {/* Backtest Performance */}
            <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">
                Validated Performance
              </h2>
              {strategy ? (
                <div className="grid grid-cols-2 gap-3">
                  <MetricItem
                    label="Sharpe"
                    value={strategy.backtest.sharpe.toFixed(2)}
                  />
                  <MetricItem
                    label="Beta-Neutral Sharpe"
                    value={strategy.backtest.beta_neutral_sharpe.toFixed(2)}
                    highlight
                  />
                  <MetricItem
                    label="Total Return"
                    value={`${(strategy.backtest.total_return * 100).toFixed(1)}%`}
                  />
                  <MetricItem
                    label="Precision (Buy)"
                    value={`${(strategy.backtest.precision_buy * 100).toFixed(1)}%`}
                  />
                  <MetricItem
                    label="Win Rate"
                    value={`${(strategy.backtest.win_rate * 100).toFixed(1)}%`}
                  />
                  <MetricItem
                    label="Win Folds"
                    value={strategy.backtest.win_folds}
                  />
                  <MetricItem
                    label="CVaR"
                    value={strategy.backtest.cvar.toFixed(4)}
                  />
                  <MetricItem
                    label="Turnover"
                    value={strategy.backtest.turnover.toFixed(1)}
                  />
                </div>
              ) : (
                <p className="text-sm text-gray-500">Unable to load backtest data</p>
              )}
            </div>

            {/* Governance / Trust */}
            <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">
                Trust Assessment
              </h2>
              {strategy ? (
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-500">Pipeline Trust Score</span>
                      <span className="font-mono">
                        {(strategy.governance.trust_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="gauge-track">
                      <div
                        className="gauge-fill bg-accent"
                        style={{
                          width: `${strategy.governance.trust_score * 100}%`,
                        }}
                      />
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Level: {strategy.governance.trust_level}
                    </div>
                  </div>

                  <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
                    <div className="text-xs text-gray-500 mb-2">Pipeline Recommendation</div>
                    <div className="text-sm font-medium text-yellow-400">
                      {strategy.governance.recommendation}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Single strategy = monoculture risk. Trust score penalized.
                    </div>
                  </div>

                  {exposure && (
                    <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
                      <div className="text-xs text-gray-500 mb-2">Live Guidance</div>
                      <div className="text-xs text-gray-400 leading-relaxed">
                        {exposure.position_guidance}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-gray-500">Unable to load governance data</p>
              )}
            </div>
          </div>

          {/* Early Warning Signals Summary */}
          {warning && warning.signals.length > 0 && (
            <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Early Warning Signals
                </h2>
                <a
                  href="/early-warning"
                  className="text-xs text-accent hover:underline"
                >
                  View Details
                </a>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                {warning.signals.map((signal) => (
                  <SignalCard key={signal.name} signal={signal} />
                ))}
              </div>
              <div className="mt-4 text-xs text-gray-500">
                Last updated: {warning.timestamp}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function MetricItem({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
      <div
        className={`font-mono text-sm font-medium ${
          highlight ? "text-accent" : ""
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function SignalCard({ signal }: { signal: { name: string; score: number; weight: number; detail: string } }) {
  const pct = Math.round(signal.score * 100);
  let barColor = "bg-emerald-400";
  if (signal.score >= 0.6) barColor = "bg-rose-500";
  else if (signal.score >= 0.3) barColor = "bg-yellow-400";

  const displayName = signal.name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (l) => l.toUpperCase());

  return (
    <div className="text-center">
      <div className="text-[10px] text-gray-500 dark:text-gray-400 mb-1 truncate">
        {displayName}
      </div>
      <div className="gauge-track mb-1">
        <div className={`gauge-fill ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="text-xs font-mono">{pct}%</div>
    </div>
  );
}
