import { Holding } from "./types";

export const sampleHoldings: Holding[] = [
  { id: "wm-aggr", ticker: "WM-AGGR", shares: 1, purchasePrice: 102370.1, currentPrice: 121312.79, tag: "Aggressive WM" },
  { id: "wm-cons", ticker: "WM-CONS", shares: 1, purchasePrice: 166900, currentPrice: 185872, tag: "Conservative WM" },
  { id: "self", ticker: "SELF", shares: 1, purchasePrice: 32920, currentPrice: 35780, tag: "Self-Managed" },
  { id: "spy", ticker: "SPY", shares: 1, purchasePrice: 629.17, currentPrice: 690.62, tag: "SPY" }
];

export const defaultBenchmarks = ["SPY", "QQQ", "^GSPC", "^IXIC"];
