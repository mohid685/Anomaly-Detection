"use client";

import { PredictionResult } from "@/lib/api";

export default function ResultPanel({ result }: { result: PredictionResult | null }) {
  if (!result) {
    return (
      <div className="flex h-72 items-center justify-center border border-surface-border bg-surface-panel">
        <div className="text-sm text-neutral-600 font-mono">Awaiting inspection</div>
      </div>
    );
  }

  const statusColor = result.is_defective ? "text-status-alert" : "text-status-ok";
  const statusLabel = result.is_defective ? "DEFECT DETECTED" : "NO DEFECT DETECTED";
  const marginPct = (((result.score - result.threshold) / result.threshold) * 100).toFixed(1);

  return (
    <div className="border border-surface-border bg-surface-panel">
      <div className="border-b border-surface-border p-4">
        <img
          src={`data:image/png;base64,${result.heatmap_base64}`}
          alt="Anomaly heatmap overlay"
          className="mx-auto max-h-64 object-contain"
        />
      </div>
      <div className="grid grid-cols-3 divide-x divide-surface-border font-mono text-sm">
        <div className="p-4">
          <div className="text-xs text-neutral-500 uppercase tracking-wide">Verdict</div>
          <div className={`mt-1 font-semibold ${statusColor}`}>{statusLabel}</div>
        </div>
        <div className="p-4">
          <div className="text-xs text-neutral-500 uppercase tracking-wide">Anomaly Score</div>
          <div className="mt-1 text-neutral-200">{result.score.toFixed(4)}</div>
        </div>
        <div className="p-4">
          <div className="text-xs text-neutral-500 uppercase tracking-wide">Threshold</div>
          <div className="mt-1 text-neutral-200">{result.threshold.toFixed(4)}</div>
        </div>
      </div>
      <div className="border-t border-surface-border p-4 text-xs text-neutral-500 font-mono">
        Deviation from threshold: {marginPct}%
      </div>
    </div>
  );
}