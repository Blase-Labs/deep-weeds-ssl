import random
import math
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
    def __init__(
            self,
            weed_model,
            labels_path,
            label_map,
            og_X,
            og_y,
            total_samples=6000,
            device_id=1,
            base_path="/workspace",
            img_shape=(256, 256),
            trainable_layers=1,
            n_cycles=30,
            inference_pred_split=0.2
    ):
        self.n_prev_samples = 100
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
        try:
            return int(filename.split("-")[-1].split(".")[0])
        except:
            return -1  # Invalid/malformed filename
        
    def collect_filenames(self):
        # Filter file
        df = pd.read_csv(self.labels_path)
        df = df[df["Filename"].str.endswith(".jpg", na=False)]
        df["InstrumentID"] = df["Filename"].apply(self._get_instrument_id)
        df = df[df["InstrumentID"] == self.device_id]

        # Shuffle and sample the first N
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
        df = df.head(self.total_samples)  # grab first N samples

        return df
    
    def image_to_tensor(self, img_file):
        img_path = self.base_path + "/images/" + str(img_file)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_shape)
        arr = np.array(img) / 255.0
        return arr.astype(np.float32)

    def load_prediction_data(self):
        self.new_data_df = self.collect_filenames()

        self.new_data_df.loc[:, "Label"] = self.new_data_df["Species"].map(self.label_map)

        self.X_paths = self.new_data_df["Filename"].tolist()
        self.y = self.new_data_df["Label"].tolist()

    # ----- Model -----
    def model_setup(self):
        self.fine_tune_model = clone_model(self.weed_model)
        self.fine_tune_model.set_weights(self.weed_model.get_weights())

        # Unfreeze just the last N layers of the base model
        base_model = self.fine_tune_model.get_layer("mobilenetv2_1.00_224")
        base_model.trainable = True
        for layer in base_model.layers[:-self.trainable_layers]:
            layer.trainable = False

        # Add L2 regularization to trainable layers
        l2_weight = 1e-2  # You can tune this (start small)
        for layer in base_model.layers[-self.trainable_layers:]:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = regularizers.l2(l2_weight)
        if hasattr(self.fine_tune_model.layers[-1], "kernel_regularizer"):
            self.fine_tune_model.layers[-1].kernel_regularizer = regularizers.l2(l2_weight)

        # IMPORTANT: Rebuild model with updated regularization config
        # This step re-applies the layer configs so regularization takes effect
        config = self.fine_tune_model.get_config()
        weights = self.fine_tune_model.get_weights()
        self.fine_tune_model = tf.keras.models.Model.from_config(config)
        self.fine_tune_model.set_weights(weights)

        # Compile the regularized model
        self.fine_tune_model.compile(
            optimizer=tf.keras.optimizers.Adam(0.00001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.fine_tune_model.summary()

    # ----- Inference -----
    def return_tensors(self):
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

        self.train_df = pd.DataFrame({
            "Filename": filenames,
            "Predicted_Class": decoded_preds,
            "Confidence": y_pred_conf,
            "True_Class": decoded_truth
        })

        if mode == "test":
            if model == self.weed_model:
                self.weed_model_test_df = pd.concat([self.weed_model_test_df, self.train_df], ignore_index=True)
            elif model == self.fine_tune_model:
                self.fine_tune_test_df = pd.concat([self.fine_tune_test_df, self.train_df], ignore_index=True) 
        
    # ----- Fine tuner -----
    def fine_tune_data_initializer(self):
        pass

    def weak_augment_image(self, img_arr):
        # Just basic flip + resize — keep it simple
        img_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
        img_tensor = tf.image.random_flip_left_right(img_tensor)
        img_tensor = tf.image.random_flip_up_down(img_tensor)
        img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)
        return img_tensor

    def augment_image(self, img_arr):
        # img_tensor: tf.Tensor with shape (H, W, C), dtype float32, values in [0, 1]
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
        img_uint8 = (img_np * 255).astype(np.uint8)

        # JPEG compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        _, encimg = cv2.imencode('.jpg', img_uint8, encode_param)
        img_jpeg = cv2.imdecode(encimg, 1)

        # Slight Gaussian blur
        img_blur = cv2.GaussianBlur(img_jpeg, (3, 3), sigmaX=1)

        # Slight brightness + contrast shift
        img_final = cv2.convertScaleAbs(img_blur, alpha=0.9, beta=10)

        return img_final.astype(np.float32) / 255.0
    
    def compute_class_thresholds(self, max_per_class, min_threshold=0.2, max_threshold=0.95):
        class_freqs = Counter(self.pseudo_labeled_y)

        thresholds = {}
        for label in range(len(self.label_map)):
            count = class_freqs.get(label, 0)
            freq_ratio = min(count / max_per_class, 1.0)
            # Invert it: rare = lower threshold
            thresholds[label] = max_threshold - (max_threshold - min_threshold) * (1.0 - freq_ratio)

        return thresholds

    def fine_tune_data_handler(self):
        max_per_class = 100
        self.per_class_thresholds = self.compute_class_thresholds(max_per_class=max_per_class)
        df = self.train_df.copy()
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
            weak_confidence = np.max(weak_probs)

            sorted_probs = np.sort(weak_probs)
            margin = sorted_probs[-1] - sorted_probs[-2]
            if margin < 0.01: continue

            if row.Confidence >= self.per_class_thresholds.get(weak_label, self.confidence_threshold):
                strong_probs = self.fine_tune_model.predict(x_s[np.newaxis, ...])[0]
                strong_label = np.argmax(strong_probs)

                if strong_label == weak_label == row.Label:
                    if class_counts[strong_label] >= max_per_class:
                        continue  # Enforce total cap
                    # self.fine_tune_X.append(x_s)
                    # self.fine_tune_y.append(int(strong_label))
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
        self.fine_tune_test_df["Accuracy"] = (
            self.fine_tune_test_df["Predicted_Class"] == self.fine_tune_test_df["True_Class"]
        ).astype(int)
        
        self.weed_model_test_df["Accuracy"] = (
            self.weed_model_test_df["Predicted_Class"] == self.weed_model_test_df["True_Class"]
        ).astype(int)

        # Calculate how many samples per cycle
        samples_per_cycle = self.chunk_size

        fine_tune_avg = []
        static_avg = []

        for cycle in range(self.n_cycles):
            start = cycle * samples_per_cycle
            end = start + samples_per_cycle

            fine_cycle_acc = self.fine_tune_test_df.iloc[start:end]["Accuracy"].mean()
            static_cycle_acc = self.weed_model_test_df.iloc[start:end]["Accuracy"].mean()

            fine_tune_avg.append(fine_cycle_acc)
            static_avg.append(static_cycle_acc)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.n_cycles + 1), fine_tune_avg, label="Fine-Tuned Model", marker='o')
        plt.plot(range(1, self.n_cycles + 1), static_avg, label="Static Model", marker='x')
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
        self.load_prediction_data()
        self.model_setup()

        curr_cycle = 1
        while self.end_split < self.total_samples + 1:
            print(f"Beginning cycle: {curr_cycle}/{self.n_cycles}")
            # Load tensors
            X_train, y_train, X_test, y_test = self.return_tensors()

            # Run and return predictions
            self.inference(X_train, y_train, self.fine_tune_model, mode="train")

            # Update fine tune set
            confidence_threshold = max(0.6, 0.9 - 0.005 * (curr_cycle - 1))
            self.confidence_threshold = confidence_threshold
            print(f"[INFO] Using confidence threshold: {confidence_threshold:.2f}")
            self.fine_tune_data_handler()
            self.fine_tune_data_quality()

            # Fine tune model
            if self.fine_tune_X:
                self.fine_tuner(curr_cycle)

            # Run predictions on test split
            print("Evaluating both models on test sets...")
            self.inference(X_test, y_test, self.weed_model, mode="test")
            self.inference(X_test, y_test, self.fine_tune_model, mode="test")

            # Update beg and end split idx
            self.beg_split += self.chunk_size
            self.end_split += self.chunk_size

            curr_cycle += 1

        self.model_comp()
