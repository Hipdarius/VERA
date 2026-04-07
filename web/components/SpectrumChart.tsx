"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useTheme } from "./ThemeProvider";
import type { EndmembersResponse } from "@/lib/types";

const ENDMEMBER_COLORS: Record<string, { dark: string; light: string }> = {
  olivine: { dark: "#4ade80", light: "#16a34a" },
  pyroxene: { dark: "#fb923c", light: "#ea580c" },
  anorthite: { dark: "#60a5fa", light: "#2563eb" },
  ilmenite: { dark: "#94a3b8", light: "#64748b" },
};

function getEndmemberColor(name: string, isLight: boolean): string {
  const key = Object.keys(ENDMEMBER_COLORS).find((k) =>
    name.toLowerCase().includes(k)
  );
  if (!key) return isLight ? "#94a3b8" : "#64748b";
  return isLight ? ENDMEMBER_COLORS[key].light : ENDMEMBER_COLORS[key].dark;
}

export function SpectrumChart({
  wavelengths,
  spectrum,
  endmembers,
}: {
  wavelengths: number[] | null;
  spectrum: number[] | null;
  endmembers: EndmembersResponse | null;
}) {
  const { theme } = useTheme();
  const isLight = theme === "light";

  if (!wavelengths || !spectrum) {
    return (
      <div
        className="flex h-72 items-center justify-center font-mono text-xs uppercase tracking-widest"
        style={{ color: isLight ? "#94a3b8" : "#64748b" }}
      >
        awaiting acquisition\u2026
      </div>
    );
  }

  // Endmember names from the dict-shaped response
  const endmemberNames = endmembers ? Object.keys(endmembers.endmembers) : [];

  // Build data array with spectrum + endmember columns
  const data = wavelengths.map((nm, i) => {
    const row: Record<string, number> = {
      nm: Math.round(nm),
      reflectance: spectrum[i],
    };
    if (endmembers) {
      for (const name of endmemberNames) {
        const emSpec = endmembers.endmembers[name];
        // Find the closest wavelength index in the endmember data
        const emIdx = endmembers.wavelengths_nm.findIndex(
          (w) => Math.abs(w - nm) < 2
        );
        if (emIdx !== -1 && emIdx < emSpec.length) {
          row[name] = emSpec[emIdx];
        }
      }
    }
    return row;
  });

  const gridColor = isLight ? "#e2e8f0" : "#1f2533";
  const axisStroke = isLight ? "#cbd5e1" : "#475569";
  const tickFill = isLight ? "#64748b" : "#94a3b8";
  const labelFill = isLight ? "#94a3b8" : "#64748b";
  const tooltipBg = isLight ? "#ffffff" : "#0a0d14";
  const tooltipBorder = isLight ? "1px solid #e2e8f0" : "1px solid #22d3ee44";
  const tooltipColor = isLight ? "#0f172a" : "#e2e8f0";

  return (
    <div>
      <div className="h-72 w-full">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
            <defs>
              <linearGradient id="specGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor={isLight ? "#0284c7" : "#22d3ee"} stopOpacity={0.9} />
                <stop offset="60%" stopColor={isLight ? "#0284c7" : "#22d3ee"} stopOpacity={0.9} />
                <stop offset="100%" stopColor={isLight ? "#d97706" : "#fbbf24"} stopOpacity={0.9} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={gridColor} strokeDasharray="3 6" />
            <XAxis
              dataKey="nm"
              stroke={axisStroke}
              tick={{ fill: tickFill, fontSize: 10, fontFamily: "ui-monospace" }}
              label={{
                value: "wavelength (nm)",
                position: "insideBottom",
                offset: -2,
                fill: labelFill,
                fontSize: 10,
              }}
            />
            <YAxis
              stroke={axisStroke}
              tick={{ fill: tickFill, fontSize: 10, fontFamily: "ui-monospace" }}
              domain={[0, "dataMax + 0.1"]}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={{
                background: tooltipBg,
                border: tooltipBorder,
                borderRadius: 8,
                fontFamily: "ui-monospace",
                fontSize: 11,
                color: tooltipColor,
              }}
              formatter={(v: number) => [v.toFixed(4), "reflectance"]}
              labelFormatter={(v) => `${v} nm`}
            />
            {/* Endmember lines rendered first (behind main spectrum) */}
            {endmemberNames.map((name) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={getEndmemberColor(name, isLight)}
                strokeWidth={1.5}
                strokeDasharray="6 4"
                strokeOpacity={0.3}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
            {/* Main spectrum line on top */}
            <Line
              type="monotone"
              dataKey="reflectance"
              stroke="url(#specGrad)"
              strokeWidth={2}
              dot={false}
              isAnimationActive
              animationDuration={650}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Endmember legend */}
      {endmemberNames.length > 0 && (
        <div
          className="mt-2 flex flex-wrap gap-4 font-mono text-[10px] uppercase tracking-widest"
          style={{ color: isLight ? "#94a3b8" : "#64748b" }}
        >
          {endmemberNames.map((name) => {
            const color = getEndmemberColor(name, isLight);
            return (
              <span key={name} className="flex items-center gap-1.5">
                <span
                  style={{
                    display: "inline-block",
                    width: 16,
                    height: 2,
                    backgroundColor: color,
                    opacity: 0.5,
                    borderTop: `1px dashed ${color}`,
                  }}
                />
                {name}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
