"use client";

import { useState } from "react";
import Papa from "papaparse";
import { Holding, PortfolioTag } from "../lib/types";

const tags: PortfolioTag[] = ["User", "Aggressive WM", "Conservative WM", "Self-Managed", "SPY"];

type Props = {
  onAdd: (holding: Holding) => void;
  onBulkAdd: (holdings: Holding[]) => void;
};

export function PortfolioForm({ onAdd, onBulkAdd }: Props) {
  const [ticker, setTicker] = useState("");
  const [shares, setShares] = useState("1");
  const [price, setPrice] = useState("");
  const [tag, setTag] = useState<PortfolioTag>("User");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!ticker || !price) return;
    onAdd({
      id: `${ticker}-${Date.now()}`,
      ticker: ticker.toUpperCase(),
      shares: Number(shares),
      purchasePrice: Number(price),
      tag
    });
    setTicker("");
    setShares("1");
    setPrice("");
  }

  function handleCSV(file: File) {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const rows = results.data as Record<string, string>[];
        const parsed = rows
          .filter((row) => row.ticker || row.symbol)
          .map((row) => {
            const symbol = (row.ticker || row.symbol || "").toUpperCase();
            return {
              id: `${symbol}-${Math.random().toString(36).slice(2)}`,
              ticker: symbol,
              shares: Number(row.shares || row.quantity || "0"),
              purchasePrice: Number(row.purchase_price || row.cost_basis || row.price || "0"),
              tag: (row.tag as PortfolioTag) || "User"
            } as Holding;
          });
        onBulkAdd(parsed);
      }
    });
  }

  return (
    <div className="grid gap-4 md:grid-cols-[1.2fr_1fr]">
      <form onSubmit={handleSubmit} className="card p-4">
        <h3 className="font-heading text-lg mb-3">Manual Entry</h3>
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <label>Ticker</label>
            <input value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="AAPL" />
          </div>
          <div>
            <label>Shares</label>
            <input value={shares} onChange={(e) => setShares(e.target.value)} type="number" min="0" step="0.01" />
          </div>
          <div>
            <label>Purchase Price</label>
            <input value={price} onChange={(e) => setPrice(e.target.value)} type="number" min="0" step="0.01" />
          </div>
          <div>
            <label>Portfolio Tag</label>
            <select value={tag} onChange={(e) => setTag(e.target.value as PortfolioTag)}>
              {tags.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>
        <button className="btn btn-active mt-4" type="submit">Add Holding</button>
      </form>

      <div className="card p-4">
        <h3 className="font-heading text-lg mb-3">CSV Upload</h3>
        <p className="text-sm text-[#a9b2bf] mb-3">
          Columns supported: ticker/symbol, shares/quantity, purchase_price/cost_basis/price, tag (optional)
        </p>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleCSV(file);
          }}
        />
      </div>
    </div>
  );
}
