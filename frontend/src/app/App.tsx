import { useState, useEffect, useContext } from "react";

import Header from "../components/Header";
import Experiments from "../views/Experiments";
import Progress from "../views/Progress";
import Accuracies from "../views/Accuracies";
import Core from "../views/Core";
import Predictions from "../views/Predictions";
import Correlations from "../views/Correlations";
import { ExperimentsContext } from "../store/experiments-context";

const HUNDRED = 100;
const TOTAL_WIDTH = 1805;
const CORE_WIDTH = (1312 / TOTAL_WIDTH) * HUNDRED;
const EXPERIMENTS_WIDTH = (1032 / TOTAL_WIDTH) * HUNDRED;
const ANALYSIS_VIEW_WIDTH = (493 / TOTAL_WIDTH) * HUNDRED;
const PROGRESS_WIDTH = (280 / TOTAL_WIDTH) * HUNDRED;

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
    <section className="w-screen relative">
      <Header distance={CORE_WIDTH} />
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
