from weed_tuner import WeedTuner
from semi_supervised_simulator import SemiSupervisedSimulator

def main():
    labels_path = "/workspace/labels.csv"

    # Fine tune pretrained model
    tuner = WeedTuner(
        labels_path=labels_path,
        device_id=2
    )
    tuner.samples_per_class = 100
    tuner.epochs = 7
    tuner.tune()

    # Run semi-supervised fine tuning comp with static model
    # on out of training distribution data (i.e. new device)
    simulator = SemiSupervisedSimulator(
        labels_path=labels_path,
        weed_model=tuner.model,
        label_map=tuner.label_map,
        og_X = tuner.X_train,
        og_y = tuner.y_train,
        device_id=1
    )
    simulator.simulator()

if __name__ == "__main__":
    main()