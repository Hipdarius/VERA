"use client";

import { motion } from "framer-motion";

export function MissionPanel({
  title,
  children,
  delay = 0,
}: {
  title: string;
  children: React.ReactNode;
  delay?: number;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="panel"
    >
      <div className="panel-header">
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-cyan-glow shadow-glow-cyan" />
        {title}
      </div>
      <div className="panel-body">{children}</div>
    </motion.section>
  );
}
