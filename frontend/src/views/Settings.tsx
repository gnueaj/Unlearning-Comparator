import { useState, useEffect } from "react";
import styles from "./Settings.module.css";

import Title from "../components/Title";
import ContentBox from "../components/ContentBox";
import ConfigurationModeSelector from "../components/UI/ConfigurationModeSelector";
import TrainingConfiguration from "../components/UI/TrainingConfiguration";
import UnlearningConfiguration from "../components/UI/UnlearningConfiguration";
import DefenseConfiguration from "../components/UI/DefenseConfiguration";

const API_URL = "http://localhost:8000";

export default function Settings() {
  const [mode, setMode] = useState(0); // 0: Training, 1: Unlearning, 2:Defense
  const [operationStatus, setOperationStatus] = useState(0); // 0: Idle, 1: Predefined, 2: Custom
  const [trainedModels, setTrainedModels] = useState<string[]>([]);
  const [unlearnedModels, setUnlearnedModels] = useState<string[]>([]);

  useEffect(() => {
    const func = async () => {
      try {
        const res = await fetch(`${API_URL}/trained_models`);
        if (!res.ok) {
          alert("Error occurred while fetching trained models.");
          return;
        }
        const json = await res.json();
        setTrainedModels(json);
      } catch (err) {
        console.log(err);
      }
    };
    func();
  }, []);

  const handleConfigModeChange = (e: React.MouseEvent<HTMLButtonElement>) => {
    setMode(+e.currentTarget.id);
  };

  return (
    <section className={styles.settings}>
      <Title title="Settings" />
      <ContentBox height={325}>
        <ConfigurationModeSelector
          mode={mode}
          status={operationStatus}
          onClick={handleConfigModeChange}
        />
        {mode === 0 ? (
          <TrainingConfiguration
            operationStatus={operationStatus}
            setOperationStatus={setOperationStatus}
            setTrainedModels={setTrainedModels}
          />
        ) : mode === 1 ? (
          <UnlearningConfiguration
            operationStatus={operationStatus}
            setOperationStatus={setOperationStatus}
            trainedModels={trainedModels}
            setUnlearnedModels={setUnlearnedModels}
          />
        ) : (
          <DefenseConfiguration />
        )}
      </ContentBox>
    </section>
  );
}
