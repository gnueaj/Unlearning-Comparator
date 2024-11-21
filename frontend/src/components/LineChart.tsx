import { useContext } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
  TooltipProps,
} from "recharts";

import { getCkaData } from "../utils/data/getCkaData";
import { ExperimentsContext } from "../store/experiments-context";
import { CircleIcon, TriangleIcon } from "./UI/icons";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
} from "../components/UI/chart";

const GREEN = "#567C31";
const RED = "#BE4130";
const DOT_SIZE = 10;
const ANIMATION_DURATION = 500;
const LABEL_FONT_SIZE = 8;

const chartConfig = {
  layer: {
    label: "Layer",
    color: "#000",
  },
  baselineForgetCka: {
    label: "Baseline (Forget Class)",
    color: RED,
  },
  baselineOtherCka: {
    label: "Baseline (Remain Classes)",
    color: GREEN,
  },
  comparisonForgetCka: {
    label: "Comparison (Forget Class)",
    color: RED,
  },
  comparisonOtherCka: {
    label: "Comparison (Remain Classes)",
    color: GREEN,
  },
} satisfies ChartConfig;

function CustomTooltip({ active, payload }: TooltipProps<number, string>) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-lg border border-border/50 bg-white px-2.5 py-1.5 text-sm shadow-xl">
        <p className="mb-1 font-bold">{payload[0].payload.layer}</p>
        <div className="flex items-center">
          <CircleIcon className="w-3 h-3 mr-1" style={{ color: RED }} />
          <p>
            Baseline (Forget Class): <strong>{payload[0].value}</strong>
          </p>
        </div>
        <div className="flex items-center">
          <CircleIcon className="w-3 h-3 mr-1" style={{ color: GREEN }} />
          <p>
            Baseline (Remain Classes): <strong>{payload[1].value}</strong>
          </p>
        </div>
        <div className="flex items-center">
          <TriangleIcon className="w-3 h-3 mr-1" color={RED} />
          <p>
            Comparison (Forget Class): <strong>{payload[2].value}</strong>
          </p>
        </div>
        <div className="flex items-center">
          <TriangleIcon className="w-3 h-3 mr-1" color={GREEN} />
          <p>
            Comparison (Remain Classes): <strong>{payload[3].value}</strong>
          </p>
        </div>
      </div>
    );
  }
  return null;
}

export default function MyLineChart({ dataset }: { dataset: string }) {
  const { baselineExperiment, comparisonExperiment } =
    useContext(ExperimentsContext);

  if (!baselineExperiment || !comparisonExperiment) return null;

  const ckaData = getCkaData(dataset, baselineExperiment, comparisonExperiment);

  return (
    <div className="relative bottom-0">
      <CustomLegend />
      <p className="text-[15px] text-center">
        Per-layer Similarity Before/After Unlearning
      </p>
      <ChartContainer
        className="w-[500px] h-[265px] relative -top-0.5"
        config={chartConfig}
      >
        <LineChart
          accessibilityLayer
          data={ckaData}
          margin={{
            top: 4,
            right: 14,
            bottom: 14,
            left: -30,
          }}
        >
          <CartesianGrid />
          <XAxis
            tickLine={false}
            tickMargin={-1}
            textAnchor="middle"
            tick={{ fontSize: LABEL_FONT_SIZE, fill: "#000000" }}
            ticks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
            label={{
              value: "Layers",
              position: "center",
              dx: 1,
              dy: 2,
              style: {
                fontSize: 12,
                textAnchor: "middle",
                fill: "#000000",
              },
            }}
          />
          <YAxis
            tickLine={false}
            domain={[0, 1]}
            interval={0}
            tick={{ fontSize: LABEL_FONT_SIZE, fill: "#000000" }}
            ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
            tickMargin={-2}
            label={{
              value: "CKA Similarity",
              angle: -90,
              position: "center",
              dx: 6,
              style: {
                fontSize: 12,
                textAnchor: "middle",
                fill: "#000000",
              },
            }}
          />
          <ChartTooltip cursor={false} content={<CustomTooltip />} />
          <Line
            dataKey="baselineForgetCka"
            type="linear"
            stroke={chartConfig.baselineForgetCka.color}
            strokeWidth={2}
            dot={{ fill: RED, stroke: RED }}
            animationDuration={ANIMATION_DURATION}
            activeDot={false}
          />
          <Line
            dataKey="baselineOtherCka"
            type="linear"
            stroke={chartConfig.baselineOtherCka.color}
            strokeWidth={2}
            dot={{ fill: GREEN, stroke: GREEN }}
            animationDuration={ANIMATION_DURATION}
            activeDot={false}
          />
          <Line
            dataKey="comparisonForgetCka"
            type="linear"
            stroke={chartConfig.comparisonForgetCka.color}
            strokeWidth={2}
            strokeDasharray="3 3"
            animationDuration={ANIMATION_DURATION}
            dot={({ cx, cy }) => {
              return (
                <TriangleIcon
                  x={cx - DOT_SIZE / 2}
                  y={cy - DOT_SIZE / 2}
                  width={DOT_SIZE}
                  height={DOT_SIZE}
                  color={RED}
                />
              );
            }}
            activeDot={false}
          />
          <Line
            dataKey="comparisonOtherCka"
            type="linear"
            stroke={chartConfig.comparisonOtherCka.color}
            strokeWidth={2}
            strokeDasharray="3 3"
            animationDuration={ANIMATION_DURATION}
            dot={({ cx, cy }) => {
              return (
                <TriangleIcon
                  x={cx - DOT_SIZE / 2}
                  y={cy - DOT_SIZE / 2}
                  width={DOT_SIZE}
                  height={DOT_SIZE}
                  color={GREEN}
                />
              );
            }}
            activeDot={false}
          />
        </LineChart>
      </ChartContainer>
    </div>
  );
}

function CustomLegend() {
  return (
    <div className="absolute top-[76px] left-10 text-[10px] leading-3">
      <div className="flex items-center">
        <div className="relative">
          <CircleIcon
            className={`mr-2 w-[${DOT_SIZE}px] h-[${DOT_SIZE}px]`}
            style={{ color: GREEN }}
          />
          <div
            className="absolute top-1/2 w-[18px] h-[1px]"
            style={{
              backgroundColor: GREEN,
              transform: "translate(-4px, -50%)",
            }}
          />
        </div>
        <span>Baseline (Remain Classes)</span>
      </div>
      <div className="flex items-center">
        <div className="relative">
          <TriangleIcon width={10} height={10} color={GREEN} className="mr-2" />
          <div
            className="absolute top-1/2 w-[18px]"
            style={{
              borderTop: `1px dashed ${GREEN}`,
              transform: "translate(-4px, -50%)",
            }}
          />
        </div>
        <span>Comparison (Remain Classes)</span>
      </div>
      <div className="flex items-center">
        <div className="relative">
          <CircleIcon
            className={`mr-2 w-[${DOT_SIZE}px] h-[${DOT_SIZE}px]`}
            style={{ color: RED }}
          />
          <div
            className="absolute top-1/2 w-[18px] h-[1px]"
            style={{
              backgroundColor: RED,
              transform: "translate(-4px, -50%)",
            }}
          />
        </div>
        <span>Baseline (Forget Class)</span>
      </div>
      <div className="mb-1 flex items-center">
        <div className="relative">
          <TriangleIcon width={10} height={10} color={RED} className="mr-2" />
          <div
            className="absolute top-1/2 w-[18px]"
            style={{
              borderTop: `1px dashed ${RED}`,
              transform: "translate(-4px, -50%)",
            }}
          />
        </div>
        <span>Comparison (Forget Class)</span>
      </div>
    </div>
  );
}
