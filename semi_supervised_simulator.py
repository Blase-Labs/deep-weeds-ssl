import random
from collections import Counter

import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import clone_model
from tensorflow.keras import regularizers
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
import cv2

class SemiSupervisedSimulator:
    """
    Simulates semi-supervised fine-tuning of a pretrained image classification model 
    using unlabeled data collected from a new device (i.e., out-of-distribution domain).

    This class benchmarks the effectiveness of iterative on-device learning by comparing:
    - A static model trained on one device (Device A)
    - A fine-tuned model updated continuously with pseudo-labeled samples from a new device (Device B)

    Key Features:
    - Loads and filters image data by device ID.
    - Simulates hardware-induced domain shift (e.g., compression, blur, contrast).
    - Applies weak/strong augmentations and confidence-based pseudo-labeling (FixMatch-style).
    - Reuses a small replay buffer (prior samples) to stabilize fine-tuning.
    - Adjusts confidence thresholds dynamically based on class distribution.
    - Compares static vs. fine-tuned model accuracy across multiple simulation cycles.
    - Produces a per-cycle accuracy plot for visualizing model improvement.

    Parameters:
        weed_model (tf.keras.Model): The pretrained image classification model (from Device A).
        labels_path (str): Path to CSV with filenames and ground-truth species labels.
        label_map (dict): Mapping of class names to integer IDs.
        og_X (list of np.ndarray): Initial training images from the source device.
        og_y (list of int): Corresponding labels for `og_X`.
        total_samples (int): Total number of new-device samples to simulate across cycles.
        device_id (int): ID for filtering new-device images from the dataset.
        base_path (str): Root directory where images are stored.
        img_shape (tuple): Target size for all input images.
        trainable_layers (int): Number of layers to keep trainable during fine-tuning.
        n_cycles (int): Number of semi-supervised training cycles to run.
        inference_pred_split (float): Fraction of each cycle's data to use for testing vs. training.

    Key Methods:
        - simulator(): Runs the full fine-tuning simulation and benchmarking loop.
        - model_setup(): Clones and compiles a regularized version of the base model for adaptation.
        - inference(): Performs evaluation and stores per-image prediction results.
        - fine_tune_data_handler(): Applies filtering, augmentation, and pseudo-labeling.
        - fine_tuner(): Trains the fine-tuned model on the augmented and pseudo-labeled dataset.
        - model_comp(): Plots and compares performance over time between static and fine-tuned models.

    Example Usage:
        simulator = SemiSupervisedSimulator(
            labels_path="/workspace/labels.csv",
            weed_model=pretrained_model,
            label_map=label_map,
            og_X=X_train,
            og_y=y_train,
            device_id=1
        )
        simulator.simulator()  # Runs multi-cycle SSL + benchmarking
    """
    def __init__(
            self,
            weed_model,
            labels_path,
            label_map,
            og_X,
            og_y,
            total_samples=1000,
            device_id=1,
            base_path="/workspace",
            img_shape=(256, 256),
            trainable_layers=1,
            n_cycles=10,
            inference_pred_split=0.2
    ):
        self.n_prev_samples = 10
        self.device_id = device_id
        self.labels_path = labels_path
        self.label_map = label_map
        og_data = list(zip(og_X, og_y))
        random.shuffle(og_data)
        sampled_data = og_data[:self.n_prev_samples]
        self.replay_X, self.replay_y = zip(*sampled_data)
        self.total_samples = total_samples
        self.weed_model = weed_model
        self.base_path = base_path
        self.img_shape = img_shape
        self.trainable_layers = trainable_layers
        self.n_cycles = n_cycles
        self.inference_pred_split = inference_pred_split

        self.new_data_df = None
        self.X_paths = []
        self.y = []

        self.per_class_thresholds = {
            0: 0.90,
            1: 0.90,
            2: 0.90,
            3: 0.90,
            4: 0.90,
            5: 0.90,
            6: 0.90,
            7: 0.90,
            8: 0.90
        }

        self.fine_tune_model = None

        self.beg_split = 0
        self.chunk_size = self.total_samples // self.n_cycles
        self.end_split = int(self.chunk_size)
        self.split_point = None

        self.train_df = None
        self.fine_tune_X = list(self.replay_X)
        self.fine_tune_y = list(self.replay_y)
        self.fine_tune_tracker = {
            "X": [],
            "y": []
        }
        self.pseudo_labeled_y = []

        self.confidence_threshold = 0.95

        self.weed_model_test_df = pd.DataFrame()
        self.fine_tune_test_df = pd.DataFrame()

    # ----- Data -----
    def _get_instrument_id(self, filename):
        """
        Extracts the numeric instrument/device ID from a filename.

        Assumes filenames follow the format: <prefix>-<ID>.jpg, where <ID> 
        identifies the camera or acquisition device.

        Args:
            filename (str): Name of the image file.

        Returns:
            int: Parsed device ID, or -1 if parsing fails.
        """
        try:
            return int(filename.split("-")[-1].split(".")[0])
        except:
            return -1
        
    def collect_filenames(self):
        """
        Loads and filters the label CSV for images captured by the target device.

        - Extracts device IDs from filenames.
        - Filters rows where the InstrumentID matches `self.device_id`.
        - Randomly shuffles the filtered data.
        - Selects the first `self.total_samples` rows.

        Returns:
            pd.DataFrame: Filtered and shuffled DataFrame of filenames and labels.
        """
        df = pd.read_csv(self.labels_path)
        df = df[df["Filename"].str.endswith(".jpg", na=False)]
        df["InstrumentID"] = df["Filename"].apply(self._get_instrument_id)
        df = df[df["InstrumentID"] == self.device_id]

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.head(self.total_samples)

        return df
    
    def image_to_tensor(self, img_file):
        """
        Loads and preprocesses a single image into a normalized float32 tensor.

        - Loads image from disk and converts to RGB.
        - Resizes to `self.img_shape`.
        - Normalizes pixel values to the [0, 1] range.

        Args:
            img_file (str): Relative path to the image file.

        Returns:
            np.ndarray: Normalized image tensor of shape (H, W, 3).
        """
        img_path = self.base_path + "/images/" + str(img_file)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_shape)
        arr = np.array(img) / 255.0
        return arr.astype(np.float32)

    def load_prediction_data(self):
        """
        Loads and prepares new-device data for inference and fine-tuning.

        - Calls `collect_filenames()` to get filtered samples.
        - Maps species names to integer labels using `self.label_map`.
        - Stores image file paths and associated labels for downstream use.

        Returns:
            None
        """
        self.new_data_df = self.collect_filenames()

        self.new_data_df.loc[:, "Label"] = self.new_data_df["Species"].map(self.label_map)

        self.X_paths = self.new_data_df["Filename"].tolist()
        self.y = self.new_data_df["Label"].tolist()

    # ----- Model -----
    def model_setup(self):
        """
        Initializes the fine-tuning model by cloning the base model and configuring it for domain adaptation.

        - Clones the pretrained base model (`self.weed_model`) to avoid mutating the original.
        - Unfreezes the last `self.trainable_layers` of the base model (MobileNetV2) for fine-tuning.
        - Applies L2 regularization to all trainable layers to reduce overfitting on small pseudo-labeled data.
        - Rebuilds the model with updated layer configs to ensure regularization takes effect.
        - Compiles the model using Adam optimizer with a low learning rate for stable adaptation.

        This model is stored as `self.fine_tune_model` and is ready for use in the SSL simulation cycles.

        Returns:
            None
        """
        self.fine_tune_model = clone_model(self.weed_model)
        self.fine_tune_model.set_weights(self.weed_model.get_weights())

        base_model = self.fine_tune_model.get_layer("mobilenetv2_1.00_224")
        base_model.trainable = True
        for layer in base_model.layers[:-self.trainable_layers]:
            layer.trainable = False

        l2_weight = 1e-2
        for layer in base_model.layers[-self.trainable_layers:]:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = regularizers.l2(l2_weight)
        if hasattr(self.fine_tune_model.layers[-1], "kernel_regularizer"):
            self.fine_tune_model.layers[-1].kernel_regularizer = regularizers.l2(l2_weight)

        config = self.fine_tune_model.get_config()
        weights = self.fine_tune_model.get_weights()
        self.fine_tune_model = tf.keras.models.Model.from_config(config)
        self.fine_tune_model.set_weights(weights)

        self.fine_tune_model.compile(
            optimizer=tf.keras.optimizers.Adam(0.00001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.fine_tune_model.summary()

    # ----- Inference -----
    def return_tensors(self):
        """
        Splits the current data chunk into training and testing subsets and converts images and labels to tensors.

        - Uses `self.inference_pred_split` to divide the current cycle's data into train/test sets.
        - Loads and preprocesses images into normalized float32 tensors.
        - Converts labels into int32 tensors.
        - Updates `self.split_point` to track the boundary between training and test data.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
                - X_train: Training images (batch, H, W, C)
                - y_train: Corresponding labels
                - X_test: Test images
                - y_test: Corresponding labels
        """
        self.split_point = self.beg_split + int((self.end_split - self.beg_split) * (1 - self.inference_pred_split))

        X_train_paths = self.X_paths[self.beg_split : self.split_point]
        y_train = self.y[self.beg_split : self.split_point]

        X_test_paths = self.X_paths[self.split_point : self.end_split]
        y_test = self.y[self.split_point : self.end_split]

        X_train = [self.image_to_tensor(path) for path in X_train_paths]
        X_test = [self.image_to_tensor(path) for path in X_test_paths]

        X_train = tf.convert_to_tensor(np.stack(X_train), dtype=tf.float32)
        X_test = tf.convert_to_tensor(np.stack(X_test), dtype=tf.float32)
        y_train = tf.convert_to_tensor(np.array(y_train), dtype=tf.int32)
        y_test = tf.convert_to_tensor(np.array(y_test), dtype=tf.int32)

        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        return X_train, y_train, X_test, y_test
    
    def inference(self, X, y, model, mode):
        """
        Runs model inference on a batch of data and logs predictions with confidence scores.

        - Evaluates the model on the provided data.
        - Converts prediction probabilities to class labels and confidence scores.
        - Decodes numeric labels into original species names using `self.label_map`.
        - Stores predictions in `self.train_df`.
        - Appends test results to either `self.weed_model_test_df` or `self.fine_tune_test_df` 
        depending on the model and mode.

        Args:
            X (tf.Tensor): Input image batch.
            y (tf.Tensor or np.ndarray): Ground truth labels.
            model (tf.keras.Model): Model to evaluate.
            mode (str): Indicates the data split being evaluated ("train" or "test").

        Returns:
            None
        """
        model.evaluate(X, y, batch_size = X.shape[0])
        y_probs = model.predict(X, batch_size = X.shape[0])
        y_pred_class = np.argmax(y_probs, axis=1)
        y_pred_conf = np.max(y_probs, axis=1)

        if tf.is_tensor(y):
            y = y.numpy()

        if tf.is_tensor(y_pred_class):
            y_pred_class = y_pred_class.numpy()

        label_decoder = {v: k for k, v in self.label_map.items()}
        decoded_preds = [label_decoder[i] for i in y_pred_class]
        decoded_truth = [label_decoder[i] for i in y]

        if mode == "train":
            filenames = self.new_data_df.iloc[self.beg_split : self.split_point]["Filename"].values
        elif mode == "test":
            filenames = self.new_data_df.iloc[self.split_point : self.end_split]["Filename"].values

        df = pd.DataFrame({
            "Filename": filenames,
            "Predicted_Class": decoded_preds,
            "Confidence": y_pred_conf,
            "True_Class": decoded_truth
        })

        if mode == "test":
            if model == self.weed_model:
                self.weed_model_test_df = pd.concat([self.weed_model_test_df, df], ignore_index=True)
            elif model == self.fine_tune_model:
                self.fine_tune_test_df = pd.concat([self.fine_tune_test_df, df], ignore_index=True)
            return  # no return for test
        elif mode == "train":
            return df
        
    # ----- Fine tuner -----
    def weak_augment_image(self, img_arr):
        """
        Applies weak data augmentation to an input image.

        Performs simple geometric transformations:
        - Random horizontal and vertical flips.

        Args:
            img_arr (np.ndarray): Normalized input image array (H, W, 3).

        Returns:
            tf.Tensor: Augmented image tensor with pixel values clipped to [0, 1].
        """
        img_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
        img_tensor = tf.image.random_flip_left_right(img_tensor)
        img_tensor = tf.image.random_flip_up_down(img_tensor)
        img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)
        return img_tensor

    def augment_image(self, img_arr):
        """
        Applies strong augmentation to simulate more realistic variability in unlabeled data.

        Includes geometric and color-based distortions:
        - Random flips
        - Random brightness, contrast, saturation, and hue shifts

        Args:
            img_arr (np.ndarray): Normalized input image array (H, W, 3).

        Returns:
            tf.Tensor: Strongly augmented image tensor with values clipped to [0, 1].
        """
        img_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
        img_tensor = tf.image.random_flip_left_right(img_tensor)
        img_tensor = tf.image.random_flip_up_down(img_tensor)
        img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)
        img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.2)
        img_tensor = tf.image.random_saturation(img_tensor, lower=0.8, upper=1.2)
        img_tensor = tf.image.random_hue(img_tensor, max_delta=0.1)
        img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)
        return img_tensor

    def simulate_device_B_effects(self, img_np):
        """
        Applies synthetic perturbations to simulate image degradation from a different device (e.g., compression, blur).

        Simulates:
        - JPEG compression artifacts
        - Gaussian blur
        - Brightness and contrast shift

        Args:
            img_np (np.ndarray): Normalized image array (H, W, 3) in float32 format.

        Returns:
            np.ndarray: Transformed image array normalized to [0, 1].
        """
        img_uint8 = (img_np * 255).astype(np.uint8)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        _, encimg = cv2.imencode('.jpg', img_uint8, encode_param)
        img_jpeg = cv2.imdecode(encimg, 1)

        img_blur = cv2.GaussianBlur(img_jpeg, (3, 3), sigmaX=1)
        img_final = cv2.convertScaleAbs(img_blur, alpha=0.9, beta=10)

        return img_final.astype(np.float32) / 255.0
    
    def compute_class_thresholds(self, max_per_class, min_threshold=0.6, max_threshold=0.95):
        """
        Dynamically computes per-class confidence thresholds for pseudo-label acceptance.

        - Classes with fewer samples receive lower thresholds to encourage exploration.
        - More frequent classes require higher confidence for acceptance.

        Args:
            max_per_class (int): Maximum allowed samples per class.
            min_threshold (float): Minimum possible confidence threshold.
            max_threshold (float): Maximum possible confidence threshold.

        Returns:
            dict: Mapping from class index to confidence threshold.
        """
        class_freqs = Counter(self.pseudo_labeled_y)

        thresholds = {}
        for label in range(len(self.label_map)):
            count = class_freqs.get(label, 0)
            freq_ratio = min(count / max_per_class, 1.0)
            # Invert it: rare = lower threshold
            thresholds[label] = max_threshold - (max_threshold - min_threshold) * (1.0 - freq_ratio)

        return thresholds

    def fine_tune_data_handler(self, df):
        """
        Filters and pseudo-labels confident samples for fine-tuning using a FixMatch-style strategy.

        - Applies weak and strong augmentations.
        - Performs dual prediction consistency check (weak vs. strong view).
        - Only retains samples where model agrees and meets per-class confidence thresholds.
        - Simulates domain shift via `simulate_device_B_effects`.
        - Updates `fine_tune_X`, `fine_tune_y`, and pseudo-label trackers.

        Returns:
            None
        """
        max_per_class = 50
        self.per_class_thresholds = self.compute_class_thresholds(max_per_class=max_per_class)
        df["Label"] = df["Predicted_Class"].map(self.label_map)

        # Track accepted samples per class
        class_counts = Counter([int(label) for label in self.fine_tune_y])

        accepted = 0
        for row in df.itertuples(index=False):
            img_path = row.Filename
            img_arr = self.image_to_tensor(img_path)

            simulated_img = self.simulate_device_B_effects(img_arr)

            x_w = self.weak_augment_image(img_arr).numpy()
            x_s = self.augment_image(simulated_img).numpy()

            weak_probs = self.fine_tune_model.predict(x_w[np.newaxis, ...])[0]
            weak_label = np.argmax(weak_probs)

            sorted_probs = np.sort(weak_probs)
            margin = sorted_probs[-1] - sorted_probs[-2]
            if margin < 0.01: continue

            if row.Confidence >= self.per_class_thresholds.get(weak_label, self.confidence_threshold):
                strong_probs = self.fine_tune_model.predict(x_s[np.newaxis, ...])[0]
                strong_label = np.argmax(strong_probs)

                if strong_label == weak_label == row.Label:
                    if class_counts[strong_label] >= max_per_class:
                        continue
                    
                    self.pseudo_labeled_y.append(int(row.Label))

                    self.fine_tune_X.append(img_arr)
                    self.fine_tune_y.append(row.Label)

                    self.fine_tune_tracker["X"].append(row.Predicted_Class)
                    self.fine_tune_tracker["y"].append(row.True_Class)

                    class_counts[strong_label] += 1
                    accepted += 1

        print(f"[INFO] Accepted {accepted}/{len(df)} pseudo-labeled samples (FixMatch style).")
        print(f"[UPDATE] Fine tune set size: {len(self.fine_tune_X)}")
        print("[DEBUG] Per-class thresholds:", self.per_class_thresholds)

        self.train_df = None

    def fine_tuner(self, curr_cycle):
        """
        Trains the fine-tune model on pseudo-labeled data accumulated so far.

        - Adjusts learning rate linearly with cycle number.
        - Increases training epochs gradually over time to stabilize adaptation.

        Args:
            curr_cycle (int): Current simulation cycle index.

        Returns:
            None
        """
        X = np.array(self.fine_tune_X)
        y = np.array(self.fine_tune_y)

        base_lr = 1e-5
        max_lr = 1e-4
        scaled_lr = min(max_lr, base_lr + curr_cycle * 1e-6)
        tf.keras.backend.set_value(self.fine_tune_model.optimizer.learning_rate, scaled_lr)

        epochs = min(1, 4 + curr_cycle // 5)

        self.fine_tune_model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=16
        )

    # ----- Simulator -----
    def model_comp(self):
        """
        Compares the accuracy of the static and fine-tuned models across completed cycles.

        - Computes per-sample accuracy for both models.
        - Aggregates accuracy per cycle based on `self.chunk_size`.
        - Dynamically infers the number of valid cycles based on available data.
        - Plots cycle-level accuracy trends and saves to disk.

        Returns:
            None
        """
        self.fine_tune_test_df["Accuracy"] = (
            self.fine_tune_test_df["Predicted_Class"] == self.fine_tune_test_df["True_Class"]
        ).astype(int)

        self.weed_model_test_df["Accuracy"] = (
            self.weed_model_test_df["Predicted_Class"] == self.weed_model_test_df["True_Class"]
        ).astype(int)

        samples_per_cycle = int(self.chunk_size * self.inference_pred_split)
        num_samples = len(self.fine_tune_test_df)
        actual_n_cycles = num_samples // samples_per_cycle

        fine_tune_avg = []
        static_avg = []

        for cycle in range(actual_n_cycles):
            start = cycle * samples_per_cycle
            end = start + samples_per_cycle

            fine_cycle_acc = self.fine_tune_test_df.iloc[start:end]["Accuracy"].mean()
            static_cycle_acc = self.weed_model_test_df.iloc[start:end]["Accuracy"].mean()

            fine_tune_avg.append(fine_cycle_acc)
            static_avg.append(static_cycle_acc)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, actual_n_cycles + 1), fine_tune_avg, label="Fine-Tuned Model", marker='o')
        plt.plot(range(1, actual_n_cycles + 1), static_avg, label="Static Model", marker='x')
        plt.xlabel("Cycle")
        plt.ylabel("Average Accuracy")
        plt.title("Accuracy per Cycle")
        plt.legend()
        plt.grid(True)

        plot_path = "/workspace/accuracy_per_cycle.png"
        plt.savefig(plot_path)
        plt.close()

        print(f"[PLOT SAVED] {plot_path}")
        print("Fine tune overall accuracy:", self.fine_tune_test_df["Accuracy"].mean())
        print("Static model overall accuracy:", self.weed_model_test_df["Accuracy"].mean())

    def fine_tune_data_quality(self):
        """
        Measures the accuracy of pseudo-labels in the fine-tuning dataset.

        - Compares pseudo-labels (`fine_tune_tracker["X"]`) with ground truth (`fine_tune_tracker["y"]`).
        - Prints the match rate as a proxy for data quality.
        - Displays per-class distribution of accepted pseudo-labels.

        Returns:
            None
        """
        n = len(self.fine_tune_tracker["X"])

        acc_count = 0
        for i in range(n):
            if self.fine_tune_tracker["X"][i] == self.fine_tune_tracker["y"][i]:
                acc_count += 1
        
        if acc_count > 0:
            acc = acc_count / n
            print("Quality of fine tune data:", acc)
            print("Fine tune data value count:", pd.Series(self.fine_tune_tracker["X"]).value_counts())

    def simulator(self):
        """
        Runs the full semi-supervised learning simulation and evaluation pipeline.

        For each simulation cycle:
        - Loads the next data chunk and splits into train/test.
        - Performs inference and pseudo-labeling on weakly augmented images.
        - Applies a FixMatch-style consistency check with strong augmentations.
        - Updates the fine-tune set with confident, validated pseudo-labels.
        - Fine-tunes the model incrementally using accumulated pseudo-labels.
        - Evaluates both static and updated models on the test split.

        After all cycles:
        - Plots and logs comparative accuracy trends using `model_comp()`.

        Returns:
            None
        """
        self.load_prediction_data()
        self.model_setup()

        curr_cycle = 1
        while self.end_split < self.total_samples + 1:
            print(f"Beginning cycle: {curr_cycle}/{self.n_cycles}")
            X_train, y_train, X_test, y_test = self.return_tensors()

            train_df = self.inference(X_train, y_train, self.fine_tune_model, mode="train")

            confidence_threshold = max(0.6, 0.9 - 0.005 * (curr_cycle - 1))
            self.confidence_threshold = confidence_threshold
            print(f"[INFO] Using confidence threshold: {confidence_threshold:.2f}")
            self.fine_tune_data_handler(train_df)
            self.fine_tune_data_quality()

            if self.fine_tune_X:
                self.fine_tuner(curr_cycle)

            print("Evaluating both models on test sets...")
            self.inference(X_test, y_test, self.weed_model, mode="test")
            self.inference(X_test, y_test, self.fine_tune_model, mode="test")

            self.beg_split += self.chunk_size
            self.end_split += self.chunk_size

            curr_cycle += 1

        self.model_comp()
