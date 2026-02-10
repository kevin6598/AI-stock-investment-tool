"use client";

import { useEffect, useState } from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { api } from "@/lib/api";

interface DiagnosticsData {
  status: string;
  source: string | null;
  diagnostics: {
    model_type?: string;
    version_id?: string;
    horizon?: string;
    created_at?: string;
    metrics?: Record<string, number>;
    overfitting_score?: number | null;
    stress_test?: Record<string, number>;
    ic_mean?: number | null;
    icir?: number | null;
    sharpe?: number | null;
    hit_ratio?: number | null;
    brier_score?: number | null;
    calibration_error?: number | null;
  } | null;
  message?: string;
}

function OverfittingMeter({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  let color = "bg-green-500";
  let label = "Good";
  if (score > 0.6) {
    color = "bg-red-500";
    label = "Concerning";
  } else if (score > 0.3) {
    color = "bg-yellow-500";
    label = "Moderate";
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-gray-600 dark:text-gray-400">Overfitting Score</span>
        <span className="font-medium">{pct}% - {label}</span>
      </div>
      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
        <div
          className={`h-3 rounded-full ${color} transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  format,
}: {
  label: string;
  value: number | null | undefined;
  format?: string;
}) {
  const display =
    value == null
      ? "N/A"
      : format === "pct"
      ? `${(value * 100).toFixed(1)}%`
      : format === "fixed4"
      ? value.toFixed(4)
      : value.toFixed(2);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
      <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide">
        {label}
      </div>
      <div className="text-2xl font-bold mt-1 text-gray-900 dark:text-white">
        {display}
      </div>
    </div>
  );
}

export default function ModelPage() {
  const [data, setData] = useState<DiagnosticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getDiagnostics()
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
        <div className="text-gray-500 dark:text-gray-400">Loading diagnostics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-700 dark:text-red-400">Error: {error}</p>
      </div>
    );
  }

  if (!data || !data.diagnostics) {
    return (
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <p className="text-yellow-700 dark:text-yellow-400">
          {data?.message || "No model diagnostics available. Train a model first."}
        </p>
      </div>
    );
  }

  const d = data.diagnostics;
  const m = d.metrics || {};

  // Radar chart data
  const radarData = [
    { metric: "IC", value: Math.min((d.ic_mean || 0) * 10, 1), fullMark: 1 },
    { metric: "Sharpe", value: Math.min(Math.max((d.sharpe || 0) / 3, 0), 1), fullMark: 1 },
    { metric: "Hit Ratio", value: d.hit_ratio || 0, fullMark: 1 },
    {
      metric: "Calibration",
      value: Math.max(1 - (d.calibration_error || 1), 0),
      fullMark: 1,
    },
    {
      metric: "Stability",
      value: Math.max(1 - (d.overfitting_score || 0.5), 0),
      fullMark: 1,
    },
  ];

  // Stress test table data
  const stressEntries = Object.entries(d.stress_test || {}).map(([name, sharpe]) => ({
    scenario: name.replace(/_/g, " "),
    sharpe: typeof sharpe === "number" ? sharpe : 0,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Model Diagnostics
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          {d.model_type || "Unknown"} | {d.version_id || "N/A"} | {d.horizon || "1M"} |
          Created: {d.created_at || "N/A"}
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="IC Mean" value={d.ic_mean} format="fixed4" />
        <MetricCard label="ICIR" value={d.icir} />
        <MetricCard label="Sharpe" value={d.sharpe} />
        <MetricCard label="Hit Ratio" value={d.hit_ratio} format="pct" />
        <MetricCard label="Brier Score" value={d.brier_score} format="fixed4" />
        <MetricCard label="Cal. Error (ECE)" value={d.calibration_error} format="fixed4" />
      </div>

      {/* Overfitting Meter */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <OverfittingMeter score={d.overfitting_score ?? 0.5} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Radar Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            Robustness Profile
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#9CA3AF", fontSize: 12 }} />
              <PolarRadiusAxis angle={90} domain={[0, 1]} tick={false} />
              <Radar
                name="Model"
                dataKey="value"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.3}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Stress Test Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            Stress Test Results
          </h2>
          {stressEntries.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stressEntries} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" tick={{ fill: "#9CA3AF", fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="scenario"
                  width={120}
                  tick={{ fill: "#9CA3AF", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1F2937",
                    border: "1px solid #374151",
                    borderRadius: "8px",
                    color: "#F3F4F6",
                  }}
                />
                <Bar dataKey="sharpe" fill="#F59E0B" name="Stress Sharpe" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              No stress test data available.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
