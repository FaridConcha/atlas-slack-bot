"use client";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ArcElement
} from "chart.js";
import { Bar, Line, Pie } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ArcElement
);

type ChartPanelProps = {
  labels: string[];
  values: (number | null)[];
  title: string;
  type?: "bar" | "line" | "pie";
  datasets?: {
    label: string;
    data: (number | null)[];
    color: string;
  }[];
};

export function ChartPanel({ labels, values, title, type = "bar", datasets }: ChartPanelProps) {
  const data = {
    labels,
    datasets: []
  };
  const baseDataset = {
    label: title,
    data: values,
    backgroundColor: [
      "rgba(242, 204, 87, 0.8)",
      "rgba(143, 202, 233, 0.8)",
      "rgba(86, 183, 132, 0.8)",
      "rgba(230, 230, 237, 0.65)"
    ],
    borderRadius: 6,
    borderWidth: 0
  };

  if (type === "line" && datasets && datasets.length > 0) {
    data.datasets = datasets.map((d) => ({
      label: d.label,
      data: d.data,
      borderColor: d.color,
      backgroundColor: d.color,
      tension: 0.35,
      pointRadius: 3,
      fill: false
    }));
  } else {
    data.datasets = [baseDataset];
  }

  const options = {
    plugins: {
      legend: { display: type === "pie" },
      tooltip: { enabled: true }
    },
    scales: type === "pie" ? undefined : {
      x: { grid: { display: false }, ticks: { color: "#a9b2bf" } },
      y: { grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "#a9b2bf" } }
    }
  } as const;

  if (type === "line") {
    return <Line data={data} options={options} />;
  }

  if (type === "pie") {
    return <Pie data={data} options={options} />;
  }

  return <Bar data={data} options={options} />;
}
