"use client";

import { useTheme } from "./ThemeProvider";
import { CLASS_LABELS, type DemoResponse } from "@/lib/types";

export function ScanHistory({
  history,
  selectedIndex,
  onSelect,
}: {
  history: DemoResponse[];
  selectedIndex: number;
  onSelect: (idx: number) => void;
}) {
  const { theme } = useTheme();
  const isLight = theme === "light";

  if (history.length === 0) return null;

  const cyanText = isLight ? "#0284c7" : "#22d3ee";
  const amberText = isLight ? "#d97706" : "#fbbf24";
  const mutedText = isLight ? "#94a3b8" : "#64748b";
  const borderColor = isLight ? "#e2e8f0" : "#1f2533";
  const selectedBg = isLight ? "rgba(2, 132, 199, 0.06)" : "rgba(34, 211, 238, 0.06)";
  const hoverBg = isLight ? "rgba(0,0,0,0.02)" : "rgba(255,255,255,0.02)";

  return (
    <div className="space-y-1">
      <div
        className="mb-2 font-mono text-[10px] uppercase tracking-widest"
        style={{ color: mutedText }}
      >
        Scan History ({history.length})
      </div>
      {history.map((scan, idx) => {
        const isSelected = idx === selectedIndex;
        const isMatch = scan.predicted_class === scan.true_class;
        const conf = (scan.confidence * 100).toFixed(1);
        const ilm = (scan.ilmenite_fraction * 100).toFixed(1);

        return (
          <button
            key={idx}
            onClick={() => onSelect(idx)}
            className="flex w-full items-center justify-between gap-3 rounded px-3 py-2 text-left font-mono transition-colors"
            style={{
              background: isSelected ? selectedBg : "transparent",
              borderLeft: isSelected
                ? `2px solid ${cyanText}`
                : "2px solid transparent",
            }}
            onMouseEnter={(e) => {
              if (!isSelected)
                (e.currentTarget as HTMLElement).style.background = hoverBg;
            }}
            onMouseLeave={(e) => {
              if (!isSelected)
                (e.currentTarget as HTMLElement).style.background = "transparent";
            }}
          >
            <div className="flex items-center gap-2">
              <span
                className="text-[10px] uppercase tracking-wider"
                style={{ color: mutedText }}
              >
                #{idx + 1}
              </span>
              <span
                className="text-xs uppercase tracking-wider"
                style={{ color: isSelected ? cyanText : (isLight ? "#0f172a" : "#e2e8f0") }}
              >
                {CLASS_LABELS[scan.predicted_class] ?? scan.predicted_class}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span
                className="text-[10px]"
                style={{ color: isMatch ? cyanText : amberText }}
              >
                {conf}%
              </span>
              <span
                className="text-[10px]"
                style={{ color: mutedText }}
              >
                FeTiO₃ {ilm}%
              </span>
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{
                  backgroundColor: isMatch ? (isLight ? "#16a34a" : "#4ade80") : amberText,
                }}
              />
            </div>
          </button>
        );
      })}
    </div>
  );
}
