import { useState, useContext } from "react";
import * as d3 from "d3";

import DatasetModeSelector from "../components/DatasetModeSelector";
import BubbleChart from "../components/BubbleChart";
import Indicator from "../components/Indicator";
import { BaselineComparisonContext } from "../store/baseline-comparison-context";
import { ForgetClassContext } from "../store/forget-class-context";
import { Target02Icon, ShortArrow, LongArrow } from "../components/UI/icons";

export const TRAIN = "train";
export const TEST = "test";
export const BUBBLE = "bubble";
export const LABEL_HEATMAP = "label-heatmap";
export const CONFIDENCE_HEATMAP = "confidence-heatmap";

export type ChartModeType = "bubble" | "label-heatmap" | "confidence-heatmap";

export default function Predictions({
  width,
  height,
}: {
  width: number;
  height: number;
}) {
  const { baseline, comparison } = useContext(BaselineComparisonContext);
  const { forgetClass } = useContext(ForgetClassContext);

  const [datasetMode, setDatasetMode] = useState(TRAIN);
  const [hoveredY, setHoveredY] = useState<number | null>(null);

  const forgetClassExist = forgetClass !== undefined;
  const allSelected = baseline !== "" && comparison !== "";

  return (
    <section
      style={{ width: `${width}vw`, height }}
      className="p-1 flex flex-col border transition-all z-10 relative"
    >
      <div className="flex justify-between">
        <div className="flex items-center mr-2">
          <Target02Icon />
          <h5 className="font-semibold ml-1 text-lg">Predictions</h5>
        </div>
        {forgetClassExist && allSelected && (
          <DatasetModeSelector onValueChange={setDatasetMode} />
        )}
      </div>
      {forgetClassExist ? (
        !allSelected ? (
          <Indicator about="BaselineComparison" />
        ) : (
          <div
            className={`flex items-center relative top-5 ${
              window.innerWidth < 1800 ? "" : "ml-1.5"
            }`}
          >
            <BubbleChart
              mode="Baseline"
              datasetMode={datasetMode}
              hoveredY={hoveredY}
              onHover={(y) => setHoveredY(y)}
              onHoverEnd={() => setHoveredY(null)}
              width={width * 9}
            />
            <BubbleChart
              mode="Comparison"
              datasetMode={datasetMode}
              showYAxis={false}
              hoveredY={hoveredY}
              onHover={(y) => setHoveredY(y)}
              onHoverEnd={() => setHoveredY(null)}
              width={width * 9}
            />
          </div>
        )
      ) : (
        <Indicator about="ForgetClass" />
      )}
      {allSelected && forgetClassExist && <BubbleChartLegend />}
    </section>
  );
}

function BubbleChartLegend() {
  return (
    <div className="flex items-center absolute top-1.5 left-1/2 -translate-x-[50%] gap-11 text-[#666666]">
      <div
        className="grid grid-cols-3 gap-x-2 place-items-center relative left-2.5 text-[10px]"
        style={{ gridTemplateRows: "18px 14px" }}
      >
        <div className="w-1.5 h-1.5 rounded-full bg-[#666666]" />
        <div className="w-3 h-3 rounded-full bg-[#666666]" />
        <div className="w-[18px] h-[18px] rounded-full bg-[#666666]" />
        <span>Less</span>
        <ShortArrow />
        <span>More</span>
        <span className="absolute top-[27px] text-[13px]">Frequent</span>
      </div>
      <div className="flex flex-col items-center gap-1 relative top-0.5">
        <ColorBar />
        <div className="text-nowrap flex items-center gap-2 text-[10px]">
          <span>Less</span>
          <LongArrow />
          <span>More</span>
        </div>
        <span className="absolute top-[25px] text-[13px]">Confident</span>
      </div>
    </div>
  );
}

const ColorBar = () => {
  const steps = 10;
  const colors = Array.from({ length: steps }, (_, i) => {
    const percent = (i / (steps - 1)) * 100;
    const color = d3.interpolateWarm(i / (steps - 1));
    return `${color} ${percent}%`;
  });

  const gradient = `linear-gradient(to right, ${colors.join(", ")})`;

  return <div className="w-[110px] h-2.5" style={{ background: gradient }} />;
};
