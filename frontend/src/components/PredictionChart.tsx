import Heatmap from "../components/Heatmap";
import { BUBBLE } from "../views/Predictions";
import { NeuralNetworkIcon, GitCompareIcon } from "./ui/icons";

type HeatmapData = { x: string; y: string; value: number }[];

interface Props {
  mode: "Baseline" | "Comparison";
  id: string;
  data: HeatmapData;
  chartMode: string;
  isExpanded: boolean;
}

export default function PredictionChart({
  mode,
  id,
  data,
  chartMode,
  isExpanded,
}: Props) {
  return (
    <div className="flex flex-col items-center -mt-1.5">
      <div className="flex items-center ml-4">
        {mode === "Baseline" ? (
          <NeuralNetworkIcon className="mr-[3px]" />
        ) : (
          <GitCompareIcon className="mr-[3px]" />
        )}
        <span className="text-[17px]">
          {mode} Model ({id})
        </span>
      </div>
      {chartMode === BUBBLE ? (
        <div className="flex flex-col items-center">
          <img
            src="/bubble.png"
            alt="bubble chart img"
            style={{
              height: isExpanded ? "420px" : "195px",
              marginRight: isExpanded ? "10px" : "0",
            }}
          />
          <span
            style={{ fontSize: isExpanded ? "16px" : "13px" }}
            className="text-[11px] font-extralight -mt-[5px]"
          >
            Prediction
          </span>
        </div>
      ) : (
        <div className="flex flex-col items-center">
          <Heatmap length={220} data={data} />
          <span
            style={{ fontSize: isExpanded ? "16px" : "13px" }}
            className="absolute text-[11px] font-extralight bottom-0.5 ml-14"
          >
            Prediction
          </span>
        </div>
      )}
    </div>
  );
}