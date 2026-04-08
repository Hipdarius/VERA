"use client";

import { motion } from "framer-motion";
import { useTheme } from "./ThemeProvider";

export function ScanButton({
  onClick,
  isScanning,
  disabled,
}: {
  onClick: () => void;
  isScanning: boolean;
  disabled?: boolean;
}) {
  const { theme } = useTheme();
  const isLight = theme === "light";

  return (
    <motion.button
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      onClick={onClick}
      disabled={disabled || isScanning}
      className="group relative overflow-hidden rounded-lg border px-6 py-3 font-mono text-sm uppercase tracking-[0.25em] transition-all disabled:cursor-not-allowed disabled:opacity-40"
      style={{
        borderColor: isLight ? "rgba(2, 132, 199, 0.4)" : "rgba(56, 189, 248, 0.4)",
        background: isLight ? "rgba(2, 132, 199, 0.08)" : "rgba(56, 189, 248, 0.1)",
        color: isLight ? "#0284c7" : "#38bdf8",
        boxShadow: isLight ? "0 1px 3px rgba(2, 132, 199, 0.15)" : "0 0 24px rgba(56, 189, 248, 0.35)",
      }}
    >
      <span className="relative z-10 flex items-center gap-3">
        <span
          className={`inline-block h-2 w-2 rounded-full ${isScanning ? "animate-pulse" : ""}`}
          style={{ backgroundColor: isLight ? "#0284c7" : "#38bdf8" }}
        />
        {isScanning ? "Scanning\u2026" : "Initiate Scan"}
      </span>

      {isScanning && (
        <span className="pointer-events-none absolute inset-x-0 top-0 h-full">
          <span
            className="absolute inset-x-0 h-[2px] animate-scan-line"
            style={{
              backgroundImage: isLight
                ? "linear-gradient(to right, transparent, #0284c7, transparent)"
                : "linear-gradient(to right, transparent, #38bdf8, transparent)",
            }}
          />
        </span>
      )}
    </motion.button>
  );
}
