import React, { useState, useContext } from "react";

import Indicator from "../components/Indicator";
import Embeddings from "./Embeddings";
import PrivacyAttack from "./PrivacyAttack";
import { ForgetClassContext } from "../store/forget-class-context";
import { Separator } from "../components/UI/separator";
import { forgetClassNames } from "../constants/forgetClassNames";
import { TABLEAU10 } from "../constants/tableau10";
import {
  ChartScatterIcon,
  CircleIcon,
  MultiplicationSignIcon,
  DarkShieldIcon,
} from "../components/UI/icons";

const EMBEDDINGS = "embeddings";
const ATTACK = "attack";
const HEIGHT = 635;

export default function Core({
  width,
  height,
}: {
  width: number;
  height: number;
}) {
  const { forgetClass } = useContext(ForgetClassContext);

  const [displayMode, setDisplayMode] = useState(EMBEDDINGS);

  const forgetClassExist = forgetClass !== undefined;

  const handleDisplayModeChange = (e: React.MouseEvent<HTMLDivElement>) => {
    const id = e.currentTarget.id;
    if (id === EMBEDDINGS) setDisplayMode(EMBEDDINGS);
    else setDisplayMode(ATTACK);
  };

  const isEmbeddingMode = displayMode === EMBEDDINGS;

  return (
    <section
      style={{ width: `${width}vw`, height }}
      className="w-full p-1 border border-l-0"
    >
      <div className="flex justify-between items-center">
        <div className="flex items-center mb-0.5">
          <div
            id={EMBEDDINGS}
            onClick={handleDisplayModeChange}
            className={`relative z-10 flex items-center mr-3 cursor-pointer pb-0.5 px-1 ${
              !isEmbeddingMode && "text-gray-400 border-none"
            }`}
          >
            <ChartScatterIcon
              className={!isEmbeddingMode ? "opacity-40" : ""}
            />
            <button className="font-semibold ml-[3px] text-lg">
              Embeddings
            </button>
            {isEmbeddingMode && <UnderLine />}
          </div>
          <div
            onClick={handleDisplayModeChange}
            className={`relative z-10 flex items-center cursor-pointer pb-0.5 px-1 ${
              isEmbeddingMode && "text-gray-400 border-none"
            }`}
          >
            <DarkShieldIcon
              className={`w-4 h-4 ${isEmbeddingMode && "opacity-40"}`}
            />
            <button id={ATTACK} className="font-semibold ml-[3px] text-lg">
              Privacy Attack
            </button>
            {!isEmbeddingMode && <UnderLine />}
          </div>
        </div>
        {forgetClassExist && isEmbeddingMode && <EmbeddingLegend />}
      </div>
      {forgetClassExist ? (
        isEmbeddingMode ? (
          <Embeddings height={HEIGHT} />
        ) : (
          <PrivacyAttack height={HEIGHT} />
        )
      ) : (
        <Indicator about="ForgetClass" />
      )}
    </section>
  );
}

function UnderLine() {
  return (
    <div className="absolute w-full h-0.5 bg-black right-0 bottom-[3px]" />
  );
}

function EmbeddingLegend() {
  const fontSize = window.innerWidth < 1800 ? "text-xs" : "text-sm";

  return (
    <div
      className={
        "flex items-center border border-b-white rounded-t-[6px] px-2 py-1 relative top-[2px] z-10 " +
        fontSize
      }
    >
      <div className="flex items-center mr-5">
        <span className="font-medium mr-2.5">Data Type</span>
        <ul className="flex items-center gap-2.5">
          <li className="flex items-center">
            <CircleIcon className="w-2 h-2 mr-1 text-[#4f5562]" />
            <span>Remaining Data</span>
          </li>
          <li className="flex items-center">
            <MultiplicationSignIcon className="text-[#4f5562] mr-1" />
            <span>Forgetting Target</span>
          </li>
        </ul>
      </div>
      <div className="flex items-center">
        <span className="font-medium mr-2.5">Prediction</span>
        <ul className="flex items-center gap-2.5">
          {forgetClassNames.map((name, idx) => (
            <li key={idx} className="flex items-center">
              <div
                style={{ backgroundColor: TABLEAU10[idx] }}
                className="w-3.5 h-3.5 mr-1"
              />
              <span>{name}</span>
            </li>
          ))}
        </ul>
      </div>
      <Separator
        orientation="horizontal"
        className="absolute bottom-[1px] h-[1px] w-[calc(100%-16px)]"
      />
    </div>
  );
}
