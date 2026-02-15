"use client";

import { useMemo, useState, useEffect } from "react";
import { PortfolioForm } from "../components/PortfolioForm";
import { ChartPanel } from "../components/ChartPanel";
import { HoldingsTable } from "../components/HoldingsTable";
import { Holding, PortfolioTag } from "../lib/types";
import { sampleHoldings } from "../lib/sample-data";

const portfolioFilters: PortfolioTag[] = [
  "Aggressive WM",
  "Conservative WM",
  "Self-Managed",
  "SPY"
];

const benchmarkTickers = ["SPY", "QQQ", "^IXIC"];

export default function Home() {
  const [holdings, setHoldings] = useState<Holding[]>(sampleHoldings);
  const [filter, setFilter] = useState<PortfolioTag | "All">("All");
  const [dateRange, setDateRange] = useState("6m");
  const [granularity, setGranularity] = useState<"daily" | "weekly" | "monthly">("weekly");
  const [viewMode, setViewMode] = useState<"dollar" | "percent">("percent");
  const [priceHistory, setPriceHistory] = useState<Record<string, Record<string, number>>>({});
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const fmt = useMemo(
    () => new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
    []
  );

  const latestPrices = useMemo(() => {
    const map: Record<string, number> = {};
    Object.entries(priceHistory).forEach(([ticker, series]) => {
      const dates = Object.keys(series).sort();
      const latest = dates[dates.length - 1];
      if (latest) map[ticker] = series[latest];
    });
    return map;
  }, [priceHistory]);

  const displayHoldings = useMemo(() => {
    return holdings.map((h) => ({
      ...h,
      currentPrice: latestPrices[h.ticker] ?? h.currentPrice ?? h.purchasePrice
    }));
  }, [holdings, latestPrices]);

  const filteredHoldings = useMemo(() => {
    if (filter === "All") return displayHoldings;
    return displayHoldings.filter((h) => h.tag === filter);
  }, [displayHoldings, filter]);

  const totals = useMemo(() => {
    return filteredHoldings.reduce(
      (acc, h) => {
        const current = latestPrices[h.ticker] ?? h.currentPrice ?? h.purchasePrice;
        const value = current * h.shares;
        const cost = h.purchasePrice * h.shares;
        acc.value += value;
        acc.cost += cost;
        return acc;
      },
      { value: 0, cost: 0 }
    );
  }, [filteredHoldings, latestPrices]);

  const totalGain = totals.value - totals.cost;
  const totalGainPct = totals.cost ? (totalGain / totals.cost) * 100 : 0;

  const allocation = useMemo(() => {
    const map = new Map<string, number>();
    filteredHoldings.forEach((h) => {
      const current = latestPrices[h.ticker] ?? h.currentPrice ?? h.purchasePrice;
      const value = current * h.shares;
      map.set(h.ticker, (map.get(h.ticker) ?? 0) + value);
    });
    const labels = Array.from(map.keys());
    const values = Array.from(map.values());
    return { labels, values };
  }, [filteredHoldings]);

  const topTickers = useMemo(() => {
    return [...filteredHoldings]
      .map((h) => {
        const current = h.currentPrice ?? h.purchasePrice;
        return { ticker: h.ticker, value: current * h.shares };
      })
      .sort((a, b) => b.value - a.value)
      .slice(0, 5)
      .map((t) => t.ticker);
  }, [filteredHoldings]);

  const holdingsByTag = useMemo(() => {
    const map: Record<string, Holding[]> = {};
    displayHoldings.forEach((h) => {
      map[h.tag] = map[h.tag] ?? [];
      map[h.tag].push(h);
    });
    return map;
  }, [displayHoldings]);

  function getContributors(tag: PortfolioTag) {
    const list = holdingsByTag[tag] ?? [];
    const computed = list
      .map((h) => {
        const current = h.currentPrice ?? h.purchasePrice;
        const value = current * h.shares;
        const cost = h.purchasePrice * h.shares;
        return { ticker: h.ticker, gain: value - cost };
      })
      .sort((a, b) => b.gain - a.gain);
    return {
      top: computed.slice(0, 5),
      bottom: computed.slice(-5).reverse()
    };
  }

  const contributorAgg = getContributors("Aggressive WM");
  const contributorCons = getContributors("Conservative WM");
  const contributorSelf = getContributors("Self-Managed");

  function handleAdd(holding: Holding) {
    setHoldings((prev) => [...prev, holding]);
  }

  function handleBulkAdd(list: Holding[]) {
    setHoldings((prev) => [...prev, ...list]);
  }

  function isFetchableTicker(ticker: string) {
    return /^[A-Z\\.\\^]{1,6}$/.test(ticker) && !ticker.startsWith("WM-") && ticker !== "SELF";
  }

  useEffect(() => {
    async function loadHistory() {
      const tickers = Array.from(new Set([...holdings.map((h) => h.ticker), ...benchmarkTickers]))
        .filter(isFetchableTicker);
      const missing = tickers.filter((t) => !priceHistory[t]);
      if (missing.length === 0) return;

      setLoadingHistory(true);
      try {
        const responses = await Promise.all(
          missing.map(async (ticker) => {
            const res = await fetch(`/api/history?symbol=${ticker}`);
            const json = await res.json();
            return { ticker, json };
          })
        );

        const next: Record<string, Record<string, number>> = { ...priceHistory };
        responses.forEach(({ ticker, json }) => {
          const series = json["Time Series (Daily)"] || {};
          const mapped: Record<string, number> = {};
          Object.keys(series).forEach((date) => {
            const close = Number(series[date]["5. adjusted close"] || series[date]["4. close"]);
            if (Number.isFinite(close)) mapped[date] = close;
          });
          next[ticker] = mapped;
        });
        setPriceHistory(next);
      } finally {
        setLoadingHistory(false);
      }
    }

    loadHistory();
  }, [holdings, priceHistory]);

  function applyDateRange(dates: string[]) {
    const sorted = [...dates].sort();
    const cutoffDays: Record<string, number> = { "1m": 31, "3m": 92, "6m": 184, "1y": 366, "5y": 1825 };
    const maxDays = cutoffDays[dateRange] ?? 184;
    const latest = sorted[sorted.length - 1];
    if (!latest) return sorted;
    const latestDate = new Date(latest);
    return sorted.filter((d) => {
      const diff = (latestDate.getTime() - new Date(d).getTime()) / (1000 * 60 * 60 * 24);
      return diff <= maxDays;
    });
  }

  function groupByGranularity(series: { date: string; value: number }[]) {
    if (granularity === "daily") return series;
    const bucketed = new Map<string, { date: string; value: number }>();
    series.forEach((point) => {
      const date = new Date(point.date);
      let key = point.date;
      if (granularity === "weekly") {
        const week = new Date(date);
        week.setDate(date.getDate() - date.getDay());
        key = week.toISOString().slice(0, 10);
      } else if (granularity === "monthly") {
        key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-01`;
      }
      bucketed.set(key, point);
    });
    return Array.from(bucketed.values()).sort((a, b) => a.date.localeCompare(b.date));
  }

  function buildSeriesForTag(tag: PortfolioTag) {
    const list = holdingsByTag[tag] ?? [];
    const valid = list.filter((h) => priceHistory[h.ticker]);
    if (valid.length === 0) return [];
    const dates = applyDateRange(
      Array.from(
        new Set(
          valid.flatMap((h) => Object.keys(priceHistory[h.ticker] ?? {}))
        )
      )
    );
    const series = dates.map((date) => {
      const value = valid.reduce((sum, h) => {
        const price = priceHistory[h.ticker]?.[date];
        if (!price) return sum;
        return sum + price * h.shares;
      }, 0);
      return { date, value };
    });
    const compact = groupByGranularity(series.filter((p) => p.value > 0));
    if (viewMode === "percent" && compact.length > 0) {
      const base = compact[0].value;
      return compact.map((p) => ({ ...p, value: (p.value / base) * 100 }));
    }
    return compact;
  }

  function buildSeriesForTicker(ticker: string) {
    const seriesMap = priceHistory[ticker];
    if (!seriesMap) return [];
    const dates = applyDateRange(Object.keys(seriesMap));
    const series = dates.map((date) => ({
      date,
      value: seriesMap[date]
    }));
    const compact = groupByGranularity(series);
    if (viewMode === "percent" && compact.length > 0) {
      const base = compact[0].value;
      return compact.map((p) => ({ ...p, value: (p.value / base) * 100 }));
    }
    return compact;
  }

  const seriesAgg = buildSeriesForTag("Aggressive WM");
  const seriesCons = buildSeriesForTag("Conservative WM");
  const seriesSelf = buildSeriesForTag("Self-Managed");
  const seriesSpy = buildSeriesForTag("SPY");
  const seriesQQQ = buildSeriesForTicker("QQQ");
  const seriesIXIC = buildSeriesForTicker("^IXIC");

  const comparisonLabels = useMemo(() => {
    const set = new Set<string>();
    [seriesAgg, seriesCons, seriesSelf, seriesSpy, seriesQQQ, seriesIXIC].forEach((s) =>
      s.forEach((p) => set.add(p.date))
    );
    return Array.from(set).sort();
  }, [seriesAgg, seriesCons, seriesSelf, seriesSpy, seriesQQQ, seriesIXIC]);

  function alignSeries(series: { date: string; value: number }[]) {
    const map = new Map(series.map((p) => [p.date, p.value]));
    return comparisonLabels.map((d) => map.get(d) ?? null);
  }

  function calcMetrics(series: { date: string; value: number }[], bench: { date: string; value: number }[]) {
    const aligned = alignSeries(series);
    const alignedBench = alignSeries(bench);
    const returns: number[] = [];
    const benchReturns: number[] = [];
    for (let i = 1; i < aligned.length; i++) {
      const prev = aligned[i - 1];
      const curr = aligned[i];
      const prevB = alignedBench[i - 1];
      const currB = alignedBench[i];
      if (prev != null && curr != null && prevB != null && currB != null) {
        returns.push((curr - prev) / prev);
        benchReturns.push((currB - prevB) / prevB);
      }
    }
    if (returns.length < 2) return { sharpe: null, beta: null, vol: null };
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const meanB = benchReturns.reduce((a, b) => a + b, 0) / benchReturns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (returns.length - 1);
    const varianceB = benchReturns.reduce((a, b) => a + Math.pow(b - meanB, 2), 0) / (benchReturns.length - 1);
    const cov = returns.reduce((a, b, i) => a + (b - mean) * (benchReturns[i] - meanB), 0) / (returns.length - 1);
    const periods = granularity === "daily" ? 252 : granularity === "weekly" ? 52 : 12;
    const vol = Math.sqrt(variance) * Math.sqrt(periods);
    const sharpe = vol ? (mean * Math.sqrt(periods)) / vol : null;
    const beta = varianceB ? cov / varianceB : null;
    return { sharpe, beta, vol };
  }

  const metricsAgg = calcMetrics(seriesAgg, seriesSpy);
  const metricsCons = calcMetrics(seriesCons, seriesSpy);
  const metricsSelf = calcMetrics(seriesSelf, seriesSpy);
  const metricsFiltered = filter === "All" ? null : calcMetrics(buildSeriesForTag(filter as PortfolioTag), seriesSpy);

  useEffect(() => {
    setSelectedTickers(topTickers);
  }, [filter, topTickers]);

  function downloadCSV() {
    if (comparisonLabels.length === 0) return;
    const rows = [
      ["date", "Aggressive WM", "Conservative WM", "Self-Managed", "SPY", "QQQ", "NASDAQ"],
      ...comparisonLabels.map((d, i) => [
        d,
        alignSeries(seriesAgg)[i] ?? "",
        alignSeries(seriesCons)[i] ?? "",
        alignSeries(seriesSelf)[i] ?? "",
        alignSeries(seriesSpy)[i] ?? "",
        alignSeries(seriesQQQ)[i] ?? "",
        alignSeries(seriesIXIC)[i] ?? ""
      ])
    ];
    const csv = rows.map((r) => r.join(",")).join("\\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "portfolio-comparison.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="min-h-screen">
      <div className="border-b border-[#2a333b] bg-[#1b2126]">
        <div className="mx-auto max-w-6xl px-6 py-6">
          <h1 className="font-heading text-3xl">Portfolio Analysis Dashboard</h1>
          <p className="text-sm text-[#a9b2bf] mt-2">
            Build and analyze portfolios, compare against benchmarks, and explore scenarios with interactive charts.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <button className={`btn ${filter === "All" ? "btn-active" : ""}`} onClick={() => setFilter("All")}>
              All
            </button>
            {portfolioFilters.map((p) => (
              <button key={p} className={`btn ${filter === p ? "btn-active" : ""}`} onClick={() => setFilter(p)}>
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-6 py-8 space-y-8">
        <section className="grid gap-4 md:grid-cols-3">
          <div className="card p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-[#a9b2bf]">Portfolio Value</p>
            <h2 className="text-2xl font-heading mt-2">${fmt.format(totals.value)}</h2>
            <p className="text-sm text-[#a9b2bf]">Filtered: {filter}</p>
          </div>
          <div className="card p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-[#a9b2bf]">Unrealized P/L</p>
            <h2 className={`text-2xl font-heading mt-2 ${totalGain >= 0 ? "text-greenBrand" : "text-red-400"}`}>
              {totalGain >= 0 ? "+" : ""}${fmt.format(totalGain)}
            </h2>
            <p className="text-sm text-[#a9b2bf]">{totalGainPct.toFixed(2)}%</p>
          </div>
          <div className="card p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-[#a9b2bf]">Date Range</p>
            <select value={dateRange} onChange={(e) => setDateRange(e.target.value)}>
              <option value="1m">1M</option>
              <option value="3m">3M</option>
              <option value="6m">6M</option>
              <option value="1y">1Y</option>
              <option value="5y">5Y</option>
            </select>
            <p className="text-sm text-[#a9b2bf] mt-2">Updates charts and performance.</p>
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="font-heading text-xl">Portfolio Context</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="card p-4">
              <h3 className="font-heading text-lg">Aggressive WM</h3>
              <p className="text-sm text-[#a9b2bf]">
                Value/cyclical tilt with strong financials, industrials, and materials exposure. This mix outperformed when
                rates stayed elevated and cyclicals led. Biggest contributors: FBT, FTXO, RDVY, GLW, FTI, and B.
              </p>
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-goldBrand">Why it worked</summary>
                <p className="text-sm text-[#a9b2bf] mt-2">
                  The last 6 months favored cyclicals and banks. Stock selection in financials and industrials
                  added alpha over SPY, while income ETFs stabilized volatility.
                </p>
              </details>
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-goldBrand">Top Contributors</summary>
                <ul className="text-sm text-[#a9b2bf] mt-2 list-disc pl-5">
                  {contributorAgg.top.length === 0 ? (
                    <li>No holdings loaded for this portfolio.</li>
                  ) : (
                    contributorAgg.top.map((c) => (
                      <li key={c.ticker}>{c.ticker}: +${fmt.format(c.gain)}</li>
                    ))
                  )}
                </ul>
              </details>
            </div>
            <div className="card p-4">
              <h3 className="font-heading text-lg">Conservative WM</h3>
              <p className="text-sm text-[#a9b2bf]">
                Income‑first portfolio combining bonds, hedged equity, and defensive growth. Designed for capital
                preservation with steady yield and modest upside.
              </p>
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-goldBrand">2026 sensitivity</summary>
                <p className="text-sm text-[#a9b2bf] mt-2">
                  If rates fall, bond prices should support returns. If markets weaken, hedged equity limits drawdown.
                </p>
              </details>
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-goldBrand">Key Contributors</summary>
                <ul className="text-sm text-[#a9b2bf] mt-2 list-disc pl-5">
                  {contributorCons.top.length === 0 ? (
                    <li>No holdings loaded for this portfolio.</li>
                  ) : (
                    contributorCons.top.map((c) => (
                      <li key={c.ticker}>{c.ticker}: +${fmt.format(c.gain)}</li>
                    ))
                  )}
                </ul>
              </details>
            </div>
            <div className="card p-4">
              <h3 className="font-heading text-lg">Self‑Managed</h3>
              <p className="text-sm text-[#a9b2bf]">
                Tech‑heavy and concentrated, with a large GOOGL weight. This setup can win in tech‑led rallies but
                lagged during value/cyclical leadership.
              </p>
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-goldBrand">Key risks</summary>
                <p className="text-sm text-[#a9b2bf] mt-2">
                  Concentration increases volatility. Underweights in energy/materials/industrials reduce performance
                  during inflation and rate‑sensitive regimes.
                </p>
              </details>
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-goldBrand">Top Winners / Losers</summary>
                <ul className="text-sm text-[#a9b2bf] mt-2 list-disc pl-5">
                  {contributorSelf.top.length === 0 ? (
                    <li>No holdings loaded for this portfolio.</li>
                  ) : (
                    <>
                      {contributorSelf.top.map((c) => (
                        <li key={c.ticker}>{c.ticker}: +${fmt.format(c.gain)}</li>
                      ))}
                      {contributorSelf.bottom.map((c) => (
                        <li key={`${c.ticker}-down`}>{c.ticker}: -${fmt.format(Math.abs(c.gain))}</li>
                      ))}
                    </>
                  )}
                </ul>
              </details>
            </div>
            <div className="card p-4">
              <h3 className="font-heading text-lg">SPY (Benchmark)</h3>
              <p className="text-sm text-[#a9b2bf]">
                Broad market proxy with heavy mega‑cap tech exposure. SPY leads in tech‑dominant cycles and provides
                a baseline for comparing alpha and risk.
              </p>
              <details className="mt-3">
                <summary className="cursor-pointer text-sm text-goldBrand">Use case</summary>
                <p className="text-sm text-[#a9b2bf] mt-2">
                  Best for passive exposure; use it to measure whether active tilts improve returns or reduce drawdowns.
                </p>
              </details>
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="font-heading text-xl">Add Holdings</h2>
          <PortfolioForm onAdd={handleAdd} onBulkAdd={handleBulkAdd} />
        </section>

        <section className="space-y-4">
          <h2 className="font-heading text-xl">Interactive Charts</h2>
          <div className="flex flex-wrap gap-2">
            <button className={`btn ${viewMode === "dollar" ? "btn-active" : ""}`} onClick={() => setViewMode("dollar")}>Dollar</button>
            <button className={`btn ${viewMode === "percent" ? "btn-active" : ""}`} onClick={() => setViewMode("percent")}>Percent Growth</button>
            <select value={granularity} onChange={(e) => setGranularity(e.target.value as "daily" | "weekly" | "monthly")}>
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
            <button className="btn" onClick={downloadCSV}>Download CSV</button>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="card p-4">
              <ChartPanel
                title="Allocation"
                type="pie"
                labels={allocation.labels}
                values={allocation.values}
              />
            </div>
            <div className="card p-4">
              <ChartPanel
                title="Holdings Value"
                labels={allocation.labels}
                values={allocation.values}
              />
            </div>
            <div className="card p-4">
              <ChartPanel
                title={`Performance (${dateRange})`}
                type="line"
                labels={["Start", "Mid", "End"]}
                values={[totals.cost, (totals.cost + totals.value) / 2, totals.value]}
              />
            </div>
          </div>
          <div className="card p-4">
            <h3 className="font-heading text-lg mb-3">Portfolio vs SPY (Line Comparison)</h3>
            <ChartPanel
              title="Comparison"
              type="line"
              labels={comparisonLabels}
              values={[]}
              datasets={[
                { label: "Aggressive WM", data: alignSeries(seriesAgg), color: "rgba(242, 204, 87, 0.9)" },
                { label: "Conservative WM", data: alignSeries(seriesCons), color: "rgba(143, 202, 233, 0.9)" },
                { label: "Self-Managed", data: alignSeries(seriesSelf), color: "rgba(86, 183, 132, 0.9)" },
                { label: "SPY", data: alignSeries(seriesSpy), color: "rgba(230, 230, 237, 0.8)" },
                seriesQQQ.length > 0 ? { label: "QQQ", data: alignSeries(seriesQQQ), color: "rgba(105, 118, 132, 0.9)" } : null,
                seriesIXIC.length > 0 ? { label: "NASDAQ", data: alignSeries(seriesIXIC), color: "rgba(34, 40, 43, 0.9)" } : null
              ].filter(Boolean) as { label: string; data: number[]; color: string }[]}
            />
            <p className="text-sm text-[#a9b2bf] mt-2">
              {loadingHistory ? "Loading real daily data..." : "Showing real historical series (adjusted close)."}
            </p>
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="font-heading text-xl">Risk Metrics (vs SPY)</h2>
          <div className="grid gap-4 md:grid-cols-4">
            {[{ label: "Aggressive WM", m: metricsAgg }, { label: "Conservative WM", m: metricsCons }, { label: "Self-Managed", m: metricsSelf }, { label: "Filtered", m: metricsFiltered }].map((item) => (
              <div key={item.label} className="card p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-[#a9b2bf]">{item.label}</p>
                <div className="mt-2 text-sm text-[#a9b2bf]">Volatility</div>
                <div className="text-lg font-heading">{item.m?.vol != null ? `${(item.m.vol * 100).toFixed(2)}%` : "N/A"}</div>
                <div className="mt-2 text-sm text-[#a9b2bf]">Sharpe</div>
                <div className="text-lg font-heading">{item.m?.sharpe != null ? item.m.sharpe.toFixed(2) : "N/A"}</div>
                <div className="mt-2 text-sm text-[#a9b2bf]">Beta</div>
                <div className="text-lg font-heading">{item.m?.beta != null ? item.m.beta.toFixed(2) : "N/A"}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-[#a9b2bf]">Metrics use {granularity} returns, annualized. Risk‑free rate assumed 0.</p>
        </section>

        <section className="space-y-4">
          <h2 className="font-heading text-xl">Ticker Performance Curves</h2>
          <div className="flex flex-wrap gap-2">
            {topTickers.map((ticker) => (
              <button
                key={ticker}
                className={`btn ${selectedTickers.includes(ticker) ? "btn-active" : ""}`}
                onClick={() => {
                  setSelectedTickers((prev) =>
                    prev.includes(ticker) ? prev.filter((t) => t !== ticker) : [...prev, ticker]
                  );
                }}
              >
                {ticker}
              </button>
            ))}
          </div>
          <div className="card p-4">
            <ChartPanel
              title="Ticker Curves"
              type="line"
              labels={comparisonLabels}
              values={[]}
              datasets={selectedTickers.map((t, idx) => ({
                label: t,
                data: alignSeries(buildSeriesForTicker(t)),
                color: [
                  "rgba(242, 204, 87, 0.9)",
                  "rgba(143, 202, 233, 0.9)",
                  "rgba(86, 183, 132, 0.9)",
                  "rgba(230, 230, 237, 0.8)",
                  "rgba(105, 118, 132, 0.9)"
                ][idx % 5]
              }))}
            />
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="font-heading text-xl">Holdings Table</h2>
          <HoldingsTable holdings={filteredHoldings} totalValue={totals.value} />
        </section>

        <section className="card p-4">
          <h2 className="font-heading text-xl mb-2">Benchmarks</h2>
          <p className="text-sm text-[#a9b2bf]">
            SPY, QQQ, and NASDAQ overlays are pulled via Alpha Vantage history. If you see gaps, check API limits or add a key in <code className="text-white">.env.local</code>.
          </p>
        </section>
      </div>
    </main>
  );
}
