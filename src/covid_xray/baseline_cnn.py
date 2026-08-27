"""
Baseline CNN trained from scratch — COVID-19 chest X-ray classification.

Reference point for the transfer-learning models (MobileNetV2 / EfficientNetB0).
Deliberately small: ~1.3M params, no pretrained weights, 1-channel input.

Expects preprocessed arrays from the existing cv2 pipeline:
    X_train, X_val, X_test : float32, shape (N, 224, 224, 1), scaled to [0, 1]
    y_train, y_val, y_test : int labels, shape (N,)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

SEED = 42
IMG_SIZE = 224
N_CLASSES = 4
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]

keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------- floor checks

def trivial_baselines(X_train, y_train, X_test, y_test, side=32):
    """Majority-class and logistic-regression-on-pixels floors.

    If the CNN doesn't clearly beat these, the problem is upstream
    (labels, splits, normalization) — not the architecture.
    """
    def flatten(X):
        small = tf.image.resize(X, (side, side)).numpy()
        return small.reshape(len(small), -1)

    Xtr, Xte = flatten(X_train), flatten(X_test)

    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, y_train)
    print("--- majority class ---")
    print(classification_report(y_test, dummy.predict(Xte),
                                target_names=CLASS_NAMES, zero_division=0))

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced",
                                n_jobs=-1, random_state=SEED).fit(Xtr, y_train)
    print(f"--- logistic regression on {side}x{side} pixels ---")
    print(classification_report(y_test, logreg.predict(Xte),
                                target_names=CLASS_NAMES, zero_division=0))


# ---------------------------------------------------------------- architecture

def build_baseline_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=N_CLASSES):
    """4 conv blocks, BN + ReLU, GAP head. ~1.3M params."""
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax", dtype="float32")(x)

    return keras.Model(inputs, outputs, name="baseline_cnn")


# ---------------------------------------------------------------- training

def train(X_train, y_train, X_val, y_val, epochs=40, batch_size=32):
    weights = compute_class_weight("balanced",
                                   classes=np.arange(N_CLASSES), y=y_train)
    class_weight = dict(enumerate(weights))
    print("class weights:", {CLASS_NAMES[i]: round(w, 3)
                             for i, w in class_weight.items()})

    model = build_baseline_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=4, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint("models/baseline_cnn.keras",
                                        monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    return model, history


# ---------------------------------------------------------------- evaluation

def evaluate(model, X_test, y_test):
    """Macro-F1 and per-class recall — accuracy is misleading here."""
    probs = model.predict(X_test, batch_size=32, verbose=0)
    preds = probs.argmax(axis=1)

    print(classification_report(y_test, preds, target_names=CLASS_NAMES,
                                digits=3, zero_division=0))
    print("confusion matrix (rows = true):")
    print(confusion_matrix(y_test, preds))

    auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    print(f"macro AUC (OvR): {auc:.4f}")

    covid = CLASS_NAMES.index("COVID")
    cm = confusion_matrix(y_test, preds)
    print(f"COVID recall: {cm[covid, covid] / cm[covid].sum():.4f}")

    return probs, preds


if __name__ == "__main__":
    # X_train, y_train, X_val, y_val, X_test, y_test = load_processed(...)
    # trivial_baselines(X_train, y_train, X_test, y_test)
    # model, history = train(X_train, y_train, X_val, y_val)
    # evaluate(model, X_test, y_test)
    pass
