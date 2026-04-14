"""
Model 1: Custom Emotion CNN
Train a CNN from scratch on FER-2013 dataset to classify 7 emotions.

Usage:
    python training/train_emotion_cnn.py

Output:
    trained_models/emotion_cnn.keras   (full model)
    trained_models/emotion_cnn.h5      (weights-only backup)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data", "fer2013")
TRAIN_DIR    = os.path.join(DATA_DIR, "train")
VAL_DIR      = os.path.join(DATA_DIR, "test")          # FER-2013 uses "test" as validation
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "trained_models")
MODEL_PATH   = os.path.join(MODEL_DIR, "emotion_cnn.keras")

IMG_SIZE     = 48
BATCH_SIZE   = 64
EPOCHS       = 50
NUM_CLASSES  = 7
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data Loading ─────────────────────────────────────────────────────────────
def make_dataset(directory, augment=False):
    ds = keras.utils.image_dataset_from_directory(
        directory,
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=augment
    )
    normalization = layers.Rescaling(1./255)
    ds = ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        augmenter = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


# ── Model Architecture (Upgraded to Custom ResNet) ─────────────────────────
def residual_block(x, filters, kernel_size=3):
    """A basic residual block (skip-connection) to combat vanishing gradients."""
    shortcut = x
    
    # Path 1
    x = layers.Conv2D(filters, kernel_size, padding="same", 
                      kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Path 2
    x = layers.Conv2D(filters, kernel_size, padding="same",
                      kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    
    # Match dimensions for shortcut jump if filters increased
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([shortcut, x])
    x = layers.Activation("relu")(x)
    return x


def build_emotion_cnn():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Initial Convolution
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Residual Block 1
    x = residual_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Residual Block 2
    x = residual_block(x, 128)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Residual Block 3
    x = residual_block(x, 256)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="Advanced_Emotion_ResNet")
    return model


# ── Training ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: Training data not found at {TRAIN_DIR}")
        print("Please run:  python training/download_datasets.py")
        exit(1)

    print("Loading datasets...")
    train_ds = make_dataset(TRAIN_DIR, augment=True)
    val_ds   = make_dataset(VAL_DIR,   augment=False)

    # Compute class weights to handle imbalance in FER-2013
    import collections
    label_counts = collections.Counter()
    for _, labels in train_ds.unbatch():
        label_counts[int(tf.argmax(labels).numpy())] += 1
    total = sum(label_counts.values())
    class_weights = {k: total / (NUM_CLASSES * v) for k, v in label_counts.items()}

    model = build_emotion_cnn()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir="training/logs/emotion_cnn")
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\nFinal Evaluation:")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")

    # Classification report
    all_preds, all_labels = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        all_preds.extend(tf.argmax(preds, axis=1).numpy())
        all_labels.extend(tf.argmax(labels, axis=1).numpy())
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS))

    # Plot Training History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history["accuracy"], label="Train"); plt.plot(history.history["val_accuracy"], label="Val"); plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history.history["loss"], label="Train"); plt.plot(history.history["val_loss"], label="Val"); plt.title("Loss"); plt.legend()
    plt.savefig("training/emotion_cnn_history.png"); plt.close()

    print(f"\n✓ Model saved to {MODEL_PATH}")
