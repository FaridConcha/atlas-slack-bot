"use client";

import { useMemo, useState } from "react";
import { Holding } from "../lib/types";
import clsx from "clsx";

type Props = {
  holdings: Holding[];
  totalValue: number;
};

type SortKey = "ticker" | "value" | "gain";

export function HoldingsTable({ holdings, totalValue }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("value");
  const [direction, setDirection] = useState<"asc" | "desc">("desc");
  const fmt = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  const sorted = useMemo(() => {
    const list = [...holdings].map((h) => {
      const current = h.currentPrice ?? h.purchasePrice ?? 0;
      const value = current * h.shares;
      const cost = h.purchasePrice * h.shares;
      const gain = value - cost;
      const gainPct = cost ? (gain / cost) * 100 : 0;
      const weight = totalValue ? (value / totalValue) * 100 : 0;
      return { ...h, current, cost, value, gain, gainPct, weight };
    });

    list.sort((a, b) => {
      const mul = direction === "asc" ? 1 : -1;
      if (sortKey === "ticker") return a.ticker.localeCompare(b.ticker) * mul;
      if (sortKey === "gain") return (a.gain - b.gain) * mul;
      return (a.value - b.value) * mul;
    });

    return list;
  }, [holdings, sortKey, direction]);

  function toggleSort(next: SortKey) {
    if (next === sortKey) {
      setDirection(direction === "asc" ? "desc" : "asc");
    } else {
      setSortKey(next);
      setDirection("desc");
    }
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-[0.18em] text-[#a9b2bf]">
            <th className="py-3" onClick={() => toggleSort("ticker")}>Ticker</th>
            <th className="py-3">Shares</th>
            <th className="py-3">Purchase Price</th>
            <th className="py-3">Current Price</th>
            <th className="py-3" onClick={() => toggleSort("value")}>Value</th>
            <th className="py-3" onClick={() => toggleSort("gain")}>Unrealized P/L</th>
            <th className="py-3">P/L %</th>
            <th className="py-3">Cost Basis</th>
            <th className="py-3">Weight</th>
            <th className="py-3">Tag</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((h) => (
            <tr key={h.id} className="border-t border-[#2a333b]">
              <td className="py-3 font-semibold">{h.ticker}</td>
              <td className="py-3">{h.shares}</td>
              <td className="py-3">${fmt.format(h.purchasePrice)}</td>
              <td className="py-3">${fmt.format(h.current)}</td>
              <td className="py-3">${fmt.format(h.value)}</td>
              <td className={clsx("py-3", h.gain >= 0 ? "text-greenBrand" : "text-red-400")}>
                {h.gain >= 0 ? "+" : ""}${fmt.format(h.gain)}
              </td>
              <td className={clsx("py-3", h.gainPct >= 0 ? "text-greenBrand" : "text-red-400")}>
                {h.gainPct >= 0 ? "+" : ""}{h.gainPct.toFixed(2)}%
              </td>
              <td className="py-3">${fmt.format(h.cost)}</td>
              <td className="py-3">{h.weight.toFixed(2)}%</td>
              <td className="py-3">{h.tag}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
