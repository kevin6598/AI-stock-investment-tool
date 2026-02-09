"use client";

import { SentimentResponse } from "@/lib/api";

interface Props {
  data: SentimentResponse | null;
  loading?: boolean;
}

export default function SentimentPanel({ data, loading }: Props) {
  if (loading) {
    return (
      <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700 animate-pulse">
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4" />
        <div className="space-y-2">
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full" />
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-2/3" />
        </div>
      </div>
    );
  }

  if (!data) return null;

  const s = data.sentiment;
  const sentimentColor =
    s.sentiment_mean > 0.1
      ? "text-up"
      : s.sentiment_mean < -0.1
      ? "text-down"
      : "text-gray-400";

  return (
    <div className="bg-white dark:bg-card rounded-xl p-6 border border-gray-200 dark:border-gray-700">
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">
        Sentiment Analysis
      </h3>

      {/* Main score */}
      <div className="flex items-center gap-4 mb-4">
        <div className={`text-3xl font-bold font-mono ${sentimentColor}`}>
          {s.sentiment_weighted >= 0 ? "+" : ""}
          {s.sentiment_weighted.toFixed(3)}
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400">
          <div>{s.news_volume} articles analyzed</div>
          <div>Momentum: {s.sentiment_momentum.toFixed(3)}</div>
        </div>
      </div>

      {/* Sentiment breakdown */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="text-center">
          <div className="text-up text-sm font-mono">
            {(s.positive_ratio * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-gray-400 dark:text-gray-500">Positive</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500 dark:text-gray-400 text-sm font-mono">
            {((1 - s.positive_ratio - s.negative_ratio) * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-gray-400 dark:text-gray-500">Neutral</div>
        </div>
        <div className="text-center">
          <div className="text-down text-sm font-mono">
            {(s.negative_ratio * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-gray-400 dark:text-gray-500">Negative</div>
        </div>
      </div>

      {/* Event & Macro impact */}
      {(s.event_direction !== 0 || s.macro_impact !== 0) && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-3 space-y-1">
          {s.event_direction !== 0 && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500 dark:text-gray-400">Event Signal</span>
              <span
                className={
                  s.event_direction > 0 ? "text-up" : "text-down"
                }
              >
                {s.event_direction > 0 ? "Bullish" : "Bearish"} (
                {s.event_magnitude.toFixed(2)})
              </span>
            </div>
          )}
          {s.macro_impact !== 0 && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500 dark:text-gray-400">Macro Impact</span>
              <span
                className={s.macro_impact > 0 ? "text-up" : "text-down"}
              >
                {s.macro_impact > 0 ? "+" : ""}
                {s.macro_impact.toFixed(3)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Keywords */}
      {Object.keys(data.keywords).length > 0 && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-3 mt-3">
          <div className="text-[10px] text-gray-400 dark:text-gray-500 mb-2">
            Keyword Themes
          </div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(data.keywords)
              .filter(([, v]) => v > 0)
              .sort(([, a], [, b]) => b - a)
              .map(([key, value]) => (
                <span
                  key={key}
                  className="text-[10px] bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full"
                >
                  {key.replace("kw_", "")}:{" "}
                  {(value * 100).toFixed(0)}%
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
