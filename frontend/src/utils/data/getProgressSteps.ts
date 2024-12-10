import { UnlearningStatus } from "../../types/settings";

const TO_FIXED_LENGTH = 3;

export const getProgressSteps = (
  status: UnlearningStatus,
  completedSteps: number[],
  activeStep: number,
  umapProgress: number,
  ckaProgress: number
) => {
  const method = status && status.method ? status.method : "";

  return [
    {
      step: 1,
      title: "Unlearn",
      description: `Method: **${
        method ? method.replace("-", "") : "-"
      }** | Epochs: **${
        !completedSteps.includes(1)
          ? "-"
          : status.current_epoch + "/" + status.total_epochs
      }**\nUnlearning Accuracy: **${
        completedSteps.includes(1) &&
        (status.current_epoch > 1 ||
          (status.total_epochs === 1 && completedSteps.includes(2)))
          ? status.current_unlearn_accuracy === 0
            ? 0
            : status.current_unlearn_accuracy.toFixed(TO_FIXED_LENGTH)
          : "-"
      }**`,
    },
    {
      step: 2,
      title: "Evaluate",
      description: `Training Accuracy: **${
        completedSteps.includes(3) ||
        (completedSteps.includes(2) && status.progress.includes("Test"))
          ? status.p_training_accuracy === 0
            ? 0
            : status.p_training_accuracy.toFixed(TO_FIXED_LENGTH)
          : "-"
      }**\nTest Accuracy: **${
        completedSteps.includes(3)
          ? status.p_test_accuracy === 0
            ? 0
            : status.p_test_accuracy.toFixed(TO_FIXED_LENGTH)
          : "-"
      }**`,
    },
    {
      step: 3,
      title: "Analyze",
      description: `${
        (activeStep === 3 &&
          (status.progress.includes("UMAP") ||
            status.progress.includes("CKA"))) ||
        (completedSteps.includes(3) &&
          (status.progress === "Idle" || status.progress === "Unlearning"))
          ? `Computing UMAP Embedding... **${
              !status.progress.includes("UMAP") ? "100" : umapProgress
            }%**`
          : "Computing UMAP Embedding"
      }\n${
        (activeStep === 3 && status.progress.includes("CKA")) ||
        (completedSteps.includes(3) &&
          (status.progress === "Idle" || status.progress === "Unlearning"))
          ? `Calculating CKA Similarity... **${
              status.progress === "Idle" || status.progress === "Unlearning"
                ? "100"
                : ckaProgress
            }%**`
          : "Calculating CKA Similarity"
      }\n${
        completedSteps.includes(3) &&
        (status.progress === "Idle" || status.progress === "Unlearning")
          ? `Done! Experiment ID: **${status.recent_id}**`
          : ""
      }`,
    },
  ];
};
