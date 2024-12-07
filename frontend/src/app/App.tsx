import { useState, useEffect, useContext } from "react";

import Header from "../components/Header";
import Experiments from "../views/Experiments";
import Progress from "../views/Progress";
import Accuracies from "../views/Accuracies";
import Core from "../views/Core";
import Predictions from "../views/Predictions";
import Correlations from "../views/Correlations";
import { ExperimentsContext } from "../store/experiments-context";

const CORE_WIDTH = 1312;
const EXPERIMENTS_WIDTH = 1032;
const ANALYSIS_VIEW_WIDTH = 493;
const PROGRESS_WIDTH = CORE_WIDTH - EXPERIMENTS_WIDTH;

const CORE_HEIGHT = 677;
const EXPERIMENTS_PROGRESS_HEIGHT = 234;
const ACCURACIES_HEIGHT = 293;
const PREDICTIONS_HEIGHT = 316;
const CORRELATIONS_HEIGHT =
  EXPERIMENTS_PROGRESS_HEIGHT +
  CORE_HEIGHT -
  ACCURACIES_HEIGHT -
  PREDICTIONS_HEIGHT;

export default function App() {
  const { isExperimentLoading } = useContext(ExperimentsContext);

  const [isPageLoading, setIsPageLoading] = useState(true);

  useEffect(() => {
    setIsPageLoading(false);
  }, []);

  if (isPageLoading) return <div></div>;

  return (
    <section className="w-[1805px] relative">
      <Header />
      {!isExperimentLoading && (
        <div className="flex items-center">
          <div>
            <div className="flex items-center">
              <Experiments
                width={EXPERIMENTS_WIDTH}
                height={EXPERIMENTS_PROGRESS_HEIGHT}
              />
              <Progress
                width={PROGRESS_WIDTH}
                height={EXPERIMENTS_PROGRESS_HEIGHT}
              />
            </div>
            <Core width={CORE_WIDTH} height={CORE_HEIGHT} />
          </div>
          <div>
            <Accuracies
              width={ANALYSIS_VIEW_WIDTH}
              height={ACCURACIES_HEIGHT}
            />
            <Predictions
              width={ANALYSIS_VIEW_WIDTH}
              height={PREDICTIONS_HEIGHT}
            />
            <Correlations
              width={ANALYSIS_VIEW_WIDTH}
              height={CORRELATIONS_HEIGHT}
            />
          </div>
        </div>
      )}
    </section>
  );
}
