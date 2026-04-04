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

export function SpectrumChart({
  wavelengths,
  spectrum,
}: {
  wavelengths: number[] | null;
  spectrum: number[] | null;
}) {
  if (!wavelengths || !spectrum) {
    return (
      <div className="flex h-72 items-center justify-center font-mono text-xs uppercase tracking-widest text-slate-500">
        awaiting acquisition…
      </div>
    );
  }

  const data = wavelengths.map((nm, i) => ({
    nm: Math.round(nm),
    reflectance: spectrum[i],
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
          <defs>
            <linearGradient id="specGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.9} />
              <stop offset="60%" stopColor="#22d3ee" stopOpacity={0.9} />
              <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.9} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#1f2533" strokeDasharray="3 6" />
          <XAxis
            dataKey="nm"
            stroke="#475569"
            tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "ui-monospace" }}
            label={{
              value: "wavelength (nm)",
              position: "insideBottom",
              offset: -2,
              fill: "#64748b",
              fontSize: 10,
            }}
          />
          <YAxis
            stroke="#475569"
            tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "ui-monospace" }}
            domain={[0, "dataMax + 0.1"]}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip
            contentStyle={{
              background: "#0a0d14",
              border: "1px solid #22d3ee44",
              borderRadius: 8,
              fontFamily: "ui-monospace",
              fontSize: 11,
              color: "#e2e8f0",
            }}
            formatter={(v: number) => [v.toFixed(4), "reflectance"]}
            labelFormatter={(v) => `${v} nm`}
          />
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
  );
}
