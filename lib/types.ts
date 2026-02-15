export type PortfolioTag = "Aggressive WM" | "Conservative WM" | "Self-Managed" | "SPY" | "User";

export type Holding = {
  id: string;
  ticker: string;
  shares: number;
  purchasePrice: number;
  currentPrice?: number;
  tag: PortfolioTag;
};

export type PricePoint = {
  date: string;
  value: number;
};
