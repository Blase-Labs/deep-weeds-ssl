from weed_tuner import WeedTuner
from semi_supervised_simulator import SemiSupervisedSimulator

def main():
    """
    Runs the full supervised and semi-supervised simulation pipeline.

    This function performs two key operations:
    
    1. **Supervised fine-tuning**:
        - Initializes a `WeedTuner` instance on labeled data from `device_id=2`.
        - Trains a MobileNetV2-based classifier using a balanced subset of labeled species data.
        - Produces a tuned model and label map for downstream use.

    2. **Semi-supervised simulation**:
        - Initializes a `SemiSupervisedSimulator` using the fine-tuned model.
        - Simulates domain shift by evaluating on data from `device_id=1`.
        - Iteratively fine-tunes the model using pseudo-labeled data over 20 cycles.
        - Compares accuracy of the static vs. updated model over time.

    Returns:
        None
    """
    labels_path = "/workspace/labels.csv"

    tuner = WeedTuner(
        labels_path=labels_path,
        device_id=2
    )
    tuner.samples_per_class = 100
    tuner.epochs = 20
    tuner.tune()

    simulator = SemiSupervisedSimulator(
        labels_path=labels_path,
        weed_model=tuner.model,
        label_map=tuner.label_map,
        og_X = tuner.X_train,
        og_y = tuner.y_train,
        device_id=1,
        n_cycles=30
    )
    simulator.simulator()

if __name__ == "__main__":
    main()

