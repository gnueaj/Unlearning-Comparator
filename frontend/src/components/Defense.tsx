import React, { useState } from "react";

import { Slider } from "./UI/slider";
import { Label } from "./UI/label";
import { DEFENSE_METHODS } from "../constants/defense";
import { DefenseConfigurationData } from "../types/settings";
import {
  HyperparametersIcon,
  StartPointIcon,
  BrightShieldIcon,
} from "./UI/icons";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./UI/select";

export default function Defense() {
  const [unlearnedModels, setUnlearnedModels] = useState<string[]>([]);
  const [epochs, setEpochs] = useState([30]);
  const [learningRateLog, setLearningRateLog] = useState([-2]);
  const [batchSizeLog, setBatchSizeLog] = useState([5]);

  const handleRunBtnClick = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const fd = new FormData(e.currentTarget);
    const configState = Object.fromEntries(
      fd.entries()
    ) as unknown as DefenseConfigurationData;
  };

  return (
    <form
      className="w-full h-full flex flex-col items-start justify-between"
      onSubmit={handleRunBtnClick}
    >
      <div className="w-full grid grid-cols-2 gap-y-2">
        {/* Initial Checkpoint */}
        <div className="flex items-center">
          <StartPointIcon className="w-4 h-4 mr-1" />
          <Label
            className="inline text-base text-nowrap"
            htmlFor="initial-checkpoint"
          >
            Initial Checkpoint
          </Label>
        </div>
        <Select name="initial-checkpoint">
          <SelectTrigger
            className="h-[25px] text-base overflow-ellipsis whitespace-nowrap"
            id="initial-checkpoint"
          >
            <SelectValue
              placeholder={
                unlearnedModels.length > 0
                  ? unlearnedModels[0].length > 15
                    ? unlearnedModels[0].slice(0, 15) + "..."
                    : unlearnedModels[0]
                  : ""
              }
            />
          </SelectTrigger>
          <SelectContent defaultValue={unlearnedModels[0]}>
            {unlearnedModels.map((item, idx) => (
              <SelectItem key={idx} value={item}>
                {item}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {/* Method */}
        <div className="flex items-center mb-1">
          <BrightShieldIcon className="w-4 h-4 mr-1" />
          <Label className="text-base text-nowrap" htmlFor="method">
            Method
          </Label>
        </div>
        <Select defaultValue={DEFENSE_METHODS[0]}>
          <SelectTrigger className="h-[25px] text-base" id="method">
            <SelectValue placeholder={DEFENSE_METHODS[0]} />
          </SelectTrigger>
          <SelectContent>
            {DEFENSE_METHODS.map((method, idx) => (
              <SelectItem key={idx} value={method}>
                {method}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {/* Hyperparameters */}
      <div className="mt-1">
        <div className="flex items-center mb-2">
          <HyperparametersIcon className="w-3.5 ml-[1px] mr-[5px]" />
          <p>Hyperparameters</p>
        </div>
        <div className="ml-10 grid grid-cols-[auto,1fr] grid-rows-3 gap-y-2">
          <span className="text-sm">Epochs</span>
          <div className="flex items-center">
            <Slider
              onValueChange={(value: number[]) => setEpochs(value)}
              value={epochs}
              defaultValue={[5]}
              className="w-[194px] mx-2 cursor-pointer"
              min={1}
              max={50}
              step={1}
            />
            <span className="text-sm">{epochs}</span>
          </div>
          <span className="text-sm">Learning Rate</span>
          <div className="flex items-center">
            <Slider
              onValueChange={setLearningRateLog}
              value={learningRateLog}
              defaultValue={learningRateLog}
              className="w-[194px] mx-2 cursor-pointer"
              min={-4}
              max={-1}
              step={1}
            />
            <span className="text-sm">1e{learningRateLog[0]}</span>
          </div>
          <span className="text-sm">Batch Size</span>
          <div className="flex items-center">
            <Slider
              onValueChange={setBatchSizeLog}
              value={batchSizeLog}
              defaultValue={batchSizeLog}
              className="w-[194px] mx-2 cursor-pointer"
              min={0}
              max={10}
              step={1}
            />
            <span className="text-sm">{Math.pow(2, batchSizeLog[0])}</span>
          </div>
        </div>
      </div>
    </form>
  );
}
