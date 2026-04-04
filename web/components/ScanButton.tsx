"use client";

import { motion } from "framer-motion";

export function ScanButton({
  onClick,
  isScanning,
  disabled,
}: {
  onClick: () => void;
  isScanning: boolean;
  disabled?: boolean;
}) {
  return (
    <motion.button
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      onClick={onClick}
      disabled={disabled || isScanning}
      className={`group relative overflow-hidden rounded-lg border border-cyan-glow/40 bg-cyan-glow/10 px-6 py-3 font-mono text-sm uppercase tracking-[0.25em] text-cyan-glow shadow-glow-cyan transition-all
        disabled:cursor-not-allowed disabled:opacity-40
        hover:bg-cyan-glow/20 hover:shadow-glow-cyan`}
    >
      <span className="relative z-10 flex items-center gap-3">
        <span
          className={`inline-block h-2 w-2 rounded-full bg-cyan-glow ${
            isScanning ? "animate-pulse" : ""
          }`}
        />
        {isScanning ? "Scanning…" : "Initiate Scan"}
      </span>

      {/* The scanline overlay only appears while a scan is running. */}
      {isScanning && (
        <span className="pointer-events-none absolute inset-x-0 top-0 h-full">
          <span className="absolute inset-x-0 h-[2px] animate-scan-line bg-gradient-to-r from-transparent via-cyan-glow to-transparent" />
        </span>
      )}
    </motion.button>
  );
}
