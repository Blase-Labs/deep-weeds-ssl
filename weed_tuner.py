import os
import random

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import resample
from PIL import Image

class WeedTuner:
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
        try:
            return int(filename.split("-")[-1].split(".")[0])
        except:
            return -1  # Invalid/malformed filename

    def collect_fileneames(self):
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
        img_path = self.base_path + "/images/" + str(img_file)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_shape)
        arr = np.array(img) / 255.0
        return arr.astype(np.float32)
    
    def label_mapper(self, df):
        label_map = {label: idx for idx, label in enumerate(sorted(set(df["Species"])))}
        self.label_map = label_map
    
    def to_tensor(self, df_train, df_val):
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
        label_df = self.collect_fileneames()
        df_shuffled = label_df.sample(frac=1, random_state=42).reset_index(drop=True)

        split_ratio = 0.8
        split_index = int(len(df_shuffled) * split_ratio)

        df_train = df_shuffled[:split_index]
        self.df_val = df_shuffled[split_index:]

        self.to_tensor(df_train, self.df_val)

    # ----- TRAINING -----
    def model_setup(self):
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