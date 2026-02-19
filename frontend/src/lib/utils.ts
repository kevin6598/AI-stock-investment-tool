import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatPercent(value: number, decimals: number = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatReturn(value: number): string {
  const pct = (value * 100).toFixed(2);
  return value >= 0 ? `+${pct}%` : `${pct}%`;
}

export function directionColor(direction: string): string {
  switch (direction) {
    case "UP":
      return "text-up";
    case "DOWN":
      return "text-down";
    default:
      return "text-gray-400";
  }
}

export function confidenceLabel(confidence: number): string {
  if (confidence >= 0.8) return "Very High";
  if (confidence >= 0.6) return "High";
  if (confidence >= 0.4) return "Moderate";
  if (confidence >= 0.2) return "Low";
  return "Very Low";
}

export function riskLabel(risk: number): string {
  if (risk >= 0.8) return "Very High";
  if (risk >= 0.6) return "High";
  if (risk >= 0.4) return "Moderate";
  if (risk >= 0.2) return "Low";
  return "Very Low";
}

export function warningLevelColor(level: string): string {
  switch (level) {
    case "HEALTHY":
      return "text-emerald-400";
    case "CAUTION":
      return "text-yellow-400";
    case "WARNING":
      return "text-orange-400";
    case "DANGER":
      return "text-rose-400";
    case "CRITICAL":
      return "text-red-600";
    default:
      return "text-gray-400";
  }
}
