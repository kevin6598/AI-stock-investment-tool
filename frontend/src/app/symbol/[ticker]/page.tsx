"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import PredictionCard from "@/components/PredictionCard";
import SentimentPanel from "@/components/SentimentPanel";
import {
  api,
  RetailPrediction,
  SentimentResponse,
  IndicatorResponse,
} from "@/lib/api";
import { formatReturn } from "@/lib/utils";

type Horizon = "1M" | "3M" | "6M";

export default function SymbolPage() {
  const params = useParams();
  const ticker = (params.ticker as string)?.toUpperCase() || "";

  const [horizon, setHorizon] = useState<Horizon>("1M");
  const [prediction, setPrediction] = useState<RetailPrediction | null>(null);
  const [sentiment, setSentiment] = useState<SentimentResponse | null>(null);
  const [indicators, setIndicators] = useState<IndicatorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setError(null);

    Promise.all([
      api.predict(ticker, horizon).catch(() => null),
      api.getSentiment(ticker).catch(() => null),
      api.getIndicators(ticker, "1y", "moving_averages,momentum,volatility").catch(() => null),
    ]).then(([predRes, sentRes, indRes]) => {
      if (predRes) setPrediction(predRes.prediction);
      else setError("Prediction failed. Is the API running?");
      setSentiment(sentRes);
      setIndicators(indRes);
      setLoading(false);
    });
  }, [ticker, horizon]);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">{ticker}</h1>
          <p className="text-sm text-gray-400">Symbol Analysis</p>
        </div>
        <div className="flex gap-2">
          {(["1M", "3M", "6M"] as Horizon[]).map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                horizon === h
                  ? "bg-accent text-gray-900 font-medium"
                  : "bg-card border border-gray-700 hover:bg-gray-700"
              }`}
            >
              {h}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg p-4 mb-6 text-rose-400 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-card rounded-xl p-6 border border-gray-700 animate-pulse h-64" />
          <div className="bg-card rounded-xl p-6 border border-gray-700 animate-pulse h-64" />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Main prediction + indicators */}
          <div className="lg:col-span-2 space-y-6">
            {prediction && <PredictionCard prediction={prediction} />}

            {/* Technical indicators table */}
            {indicators && indicators.indicators.length > 0 && (
              <div className="bg-card rounded-xl p-6 border border-gray-700">
                <h3 className="text-sm font-medium text-gray-400 mb-4">
                  Technical Indicators (latest values)
                </h3>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Indicator</th>
                      <th>Value</th>
                      <th>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {indicators.indicators.map((ind) => {
                      const latest = ind.values[ind.values.length - 1];
                      return (
                        <tr key={ind.name}>
                          <td className="font-mono text-xs">{ind.name}</td>
                          <td className="font-mono text-xs">
                            {latest?.value?.toFixed(4) ?? "N/A"}
                          </td>
                          <td className="text-xs text-gray-400">
                            {latest?.date ?? "N/A"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Advanced mode */}
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              {showAdvanced ? "Hide" : "Show"} Advanced Details
            </button>

            {showAdvanced && prediction && (
              <div className="bg-card rounded-xl p-6 border border-gray-700">
                <h3 className="text-sm font-medium text-gray-400 mb-4">
                  Advanced Details
                </h3>
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <span className="text-gray-400">Hold Signal:</span>{" "}
                    <span className="font-mono">
                      {prediction.hold_signal.toFixed(4)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Zero-Shot:</span>{" "}
                    <span>{prediction.is_zero_shot ? "Yes" : "No"}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Q05 Return:</span>{" "}
                    <span className="font-mono text-down">
                      {formatReturn(prediction.quantiles.q05)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Q95 Return:</span>{" "}
                    <span className="font-mono text-up">
                      {formatReturn(prediction.quantiles.q95)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right: Sentiment */}
          <div>
            <SentimentPanel data={sentiment} loading={loading} />
          </div>
        </div>
      )}
    </div>
  );
}
