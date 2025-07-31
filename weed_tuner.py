import random

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import resample
from PIL import Image

class WeedTuner:
    """
    Fine-tunes a lightweight image classification model to distinguish plant species 
    from labeled images collected on a specific device.

    This class supports:
    - Loading and balancing a labeled dataset (`labels.csv`) by species.
    - Preprocessing images into tensors with consistent shape and normalization.
    - Fine-tuning a MobileNetV2 model with a customizable number of trainable layers.
    - Evaluating performance on a held-out validation set.
    - Returning the tuned model and label map for downstream use.

    Designed for use in domain adaptation or field testing pipelines, 
    e.g., training on one device (e.g., phone or Pi camera) and comparing
    performance on different devices via semi-supervised learning.

    Attributes:
        labels_path (str): Path to a CSV containing image filenames and labels.
        img_shape (tuple): Target shape to resize images to.
        device_id (int): Filters the dataset to only include images from a specific device.
        base_path (str): Root directory where images are stored.
        samples_per_class (int): Number of images sampled per class during balancing.
        trainable_layers (int): Number of layers to keep trainable during fine-tuning.
        learning_rate (float): Learning rate for the Adam optimizer.
        epochs (int): Number of training epochs.

    Key Methods:
        tune(): Orchestrates the full pipeline — data loading, model setup, training, and evaluation.
        collect_fileneames(): Reads and filters the dataset for the current device and balances it by class.
        to_tensor(): Converts a DataFrame of filenames and labels to normalized image and label tensors.
        model_setup(): Constructs and compiles a fine-tuning-ready MobileNetV2 model.
        print_predictions(): Outputs a DataFrame comparing predicted and true labels with confidence scores.

    Example:
        tuner = WeedTuner(labels_path="/workspace/labels.csv", device_id=2)
        tuner.samples_per_class = 100
        tuner.epochs = 5
        tuner.tune()
        model = tuner.model  # fine-tuned Keras model
    """
    def __init__(
            self,
            labels_path,
            img_shape=(256, 256),
            device_id=0,
            base_path="/workspace",
            trainable_layers=1,
    ):
        self.labels_path = labels_path
        self.device_id = device_id
        self.samples_per_class = 100
        self.base_path = base_path
        self.trainable_layers = trainable_layers

        random.seed(42)

        self.df_val = None

        self.img_shape = img_shape
        self.label_map = None
        self.out_dim = None
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_dataset = None
        self.validation_data = None
        self.epochs = 3

        self.model = None

    # ----- DATA HANDLING -----
    def _get_instrument_id(self, filename):
        """
        Extracts the numeric instrument/device ID from a filename.

        Assumes filenames follow the format: <prefix>-<ID>.jpg
        where <ID> is an integer indicating the camera or device used.

        Args:
            filename (str): Name of the image file.

        Returns:
            int: Parsed device ID, or -1 if the filename format is invalid.
        """
        try:
            return int(filename.split("-")[-1].split(".")[0])
        except:
            return -1

    def collect_fileneames(self):
        """
        Loads and filters the label CSV for a specific device, and balances 
        the dataset across all species by resampling.

        Returns:
            pd.DataFrame: A balanced DataFrame of image filenames and labels
            corresponding to the specified device ID.
        """
        # Filter file
        df = pd.read_csv(self.labels_path)
        df = df[df["Filename"].str.endswith(".jpg", na=False)]
        df["InstrumentID"] = df["Filename"].apply(self._get_instrument_id)
        df = df[df["InstrumentID"] == self.device_id]

        # Balance the dataset across all species
        df_balanced = []
        for species, group in df.groupby("Species"):
            sampled = resample(group, n_samples=self.samples_per_class, replace=True, random_state=42)
            df_balanced.append(sampled)

        df_final = pd.concat(df_balanced).reset_index(drop=True)

        return df_final

    def image_to_tensor(self, img_file):
        """
        Loads an image file and converts it into a normalized tensor.

        Opens the image, resizes it to the configured shape, and scales pixel values to [0, 1].

        Args:
            img_file (str): Relative filename of the image.

        Returns:
            np.ndarray: Normalized image array of shape (H, W, 3) as float32.
        """
        img_path = self.base_path + "/images/" + str(img_file)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_shape)
        arr = np.array(img) / 255.0
        return arr.astype(np.float32)
    
    def label_mapper(self, df):
        """
        Creates and stores a label-to-index mapping based on the 'Species' column.

        This mapping is stored in `self.label_map` and used for converting 
        categorical labels to integer class IDs.

        Args:
            df (pd.DataFrame): DataFrame containing a 'Species' column.
        """
        label_map = {label: idx for idx, label in enumerate(sorted(set(df["Species"])))}
        self.label_map = label_map
    
    def to_tensor(self, df_train, df_val):
        """
        Converts training and validation DataFrames into image and label tensors.

        - Maps species names to integer labels using `label_mapper`.
        - Loads and normalizes images.
        - Stores resulting tensors in instance attributes: 
        `X_train`, `y_train`, `X_val`, and `y_val`.

        Args:
            df_train (pd.DataFrame): Training data.
            df_val (pd.DataFrame): Validation data.
        """
        print("Mapping labels...")
        self.label_mapper(df_train)

        df_train = df_train.copy()
        df_val = df_val.copy()

        df_train.loc[:, "Label"] = df_train["Species"].map(self.label_map)
        df_val.loc[:, "Label"] = df_val["Species"].map(self.label_map)

        df_train_x = []
        df_train_y = []

        df_val_x = []
        df_val_y = []
 
        print("Storing training X and y's...")
        for row in df_train.itertuples(index=False):
            img_path = row.Filename
            img_arr = self.image_to_tensor(img_path)
            df_train_x.append(img_arr)
            df_train_y.append(row.Label)

        print("Storing validation X and y's...")
        for row in df_val.itertuples(index=False):
            img_path = row.Filename
            img_arr = self.image_to_tensor(img_path)
            df_val_x.append(img_arr)
            df_val_y.append(row.Label)

        print("Converting to tensors...")
        X_train = tf.convert_to_tensor(np.stack(df_train_x), dtype=tf.float32)
        y_train = tf.convert_to_tensor(np.stack(df_train_y), dtype=tf.int32)

        X_val = tf.convert_to_tensor(np.stack(df_val_x), dtype=tf.float32)
        y_val = tf.convert_to_tensor(np.stack(df_val_y), dtype=tf.int32)

        print("Train shape:", X_train.shape, y_train.shape)
        print("Val shape:", X_val.shape, y_val.shape)
        print("Unique train labels:", np.unique(y_train))
        print("Unique val labels:", np.unique(y_val))

        self.X_train = X_train 
        self.y_train = y_train 
        self.X_val = X_val
        self.y_val = y_val

    def loader(self):
        """
        Loads, filters, balances, and splits the dataset.

        Performs the following steps:
        - Filters dataset by device ID and balances classes.
        - Shuffles the dataset and splits into training/validation sets.
        - Converts images and labels into tensors via `to_tensor`.
        
        Returns:
            None
        """
        label_df = self.collect_fileneames()
        df_shuffled = label_df.sample(frac=1, random_state=42).reset_index(drop=True)

        split_ratio = 0.8
        split_index = int(len(df_shuffled) * split_ratio)

        df_train = df_shuffled[:split_index]
        self.df_val = df_shuffled[split_index:]

        self.to_tensor(df_train, self.df_val)

    # ----- TRAINING -----
    def model_setup(self):
        """
        Constructs and compiles a MobileNetV2-based classification model for fine-tuning.

        - Loads MobileNetV2 with ImageNet weights and excludes the top layer.
        - Freezes all but the last `trainable_layers` layers.
        - Adds a global average pooling layer, dropout, and a softmax classification head.
        - Compiles the model with Adam optimizer and sparse categorical crossentropy loss.

        Returns:
            tf.keras.Model: The compiled fine-tuning model.
        """
        x_dim = self.img_shape[0]
        y_dim = self.img_shape[1]

        # Load the base model with imagenet weights without top layer
        model = tf.keras.applications.MobileNetV2(
            include_top=False, weights='imagenet', input_shape=(x_dim, y_dim, 3))
        model.trainable = True
        for layer in model.layers[:-self.trainable_layers]:
            layer.trainable = False
        
        # Fine tuning architecture
        input = tf.keras.Input(shape=(x_dim, y_dim, 3), name="input")
        x = model(input, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        self.out_dim = len(self.label_map)
        predictions = tf.keras.layers.Dense(self.out_dim, activation='softmax', name="output")(x)
        model = tf.keras.models.Model(inputs=input, outputs=predictions, name="ft_net")
        
        # Compile model
        model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def print_predictions(self, model):
        """
        Runs inference on the validation set and prints predicted vs. true labels.

        - Computes predicted class and confidence score for each validation image.
        - Decodes label indices back to original class names.
        - Outputs a DataFrame showing filenames, predicted classes, confidence scores, and ground truth.

        Args:
            model (tf.keras.Model): The trained model to evaluate.
        """
        y_probs = model.predict(self.X_val, batch_size=32)
        y_pred_class = np.argmax(y_probs, axis=1)
        y_pred_conf = np.max(y_probs, axis=1)

        label_decoder = {v: k for k, v in self.label_map.items()}

        decoded_preds = [label_decoder[i] for i in y_pred_class]
        y_val_np = self.y_val.numpy() if isinstance(self.y_val, tf.Tensor) else self.y_val
        decoded_truth = [label_decoder[i] for i in y_val_np]

        results_df = pd.DataFrame({
            "Filename": self.df_val["Filename"].values,
            "Predicted_Class": decoded_preds,
            "Confidence": y_pred_conf,
            "True_Class": decoded_truth
        })

        print(results_df)

    def tune(self):
        """
        Runs the complete fine-tuning pipeline:
        - Loads and balances the dataset.
        - Prepares and compiles the model architecture.
        - Trains the model on the training set and evaluates on the validation set.
        - Stores the trained model in `self.model`.

        This is the main entry point for supervised fine-tuning on a specific device.

        Returns:
            None
        """
        print("Preparing data...")
        self.loader()

        print("Preparing model...")
        model = self.model_setup()
        model.summary()

        print("Tuning...")
        history = model.fit(
            self.X_train, 
            self.y_train,
            self.train_dataset,
            epochs = self.epochs,
            validation_data=(self.X_val, self.y_val)
        )

        print("Running model on val set...")
        self.print_predictions(model)

        self.model = model
