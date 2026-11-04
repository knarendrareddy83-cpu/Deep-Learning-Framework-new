
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    auc
)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import joblib
import os

# -----------------------------
# 1. Synthetic Imbalanced & Noisy Data Generation
# -----------------------------
def generate_imbalanced_noisy_data(n_samples=5000):
    np.random.seed(42)

    X = np.random.randn(n_samples, 10)

    logits = (
        2 * X[:, 0] +
        1.5 * X[:, 1] -
        1.2 * X[:, 2] +
        np.random.normal(0, 2, n_samples)
    )

    y = (logits > np.percentile(logits, 85)).astype(int)

    noise_idx = np.random.choice(len(y), size=int(0.1 * len(y)), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(10)])
    df["target"] = y

    return df


# -----------------------------
# 2. Data Loading & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_imbalanced_noisy_data()
    df.to_csv("imbalanced_noisy_data.csv", index=False)
    print("Imbalanced and noisy dataset created and saved.")


# -----------------------------
# 3. Data Loading & Inspection
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("imbalanced_noisy_data.csv")
    print("\nDataset Head:")
    print(df.head())
    print("\nClass Distribution:")
    print(df["target"].value_counts())
    return df


# -----------------------------
# 4. Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")

    return X_scaled, y


# -----------------------------
# 5. Deep Learning Model
# -----------------------------
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# 6. Training with Class Weights
# -----------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = {
        0: 1.0,
        1: (len(y_train) / sum(y_train)) * 0.5
    }

    model = build_model(X.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )

    model.save("imbalanced_dl_model.h5")
    print("Model saved as imbalanced_dl_model.h5")

    return model, X_test, y_test


# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_probs = model.predict(X_test).flatten()
    y_pred = (y_probs > 0.5).astype(int)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")
    plt.show()


# -----------------------------
# 8. Inference Function
# -----------------------------
def intelligent_predict(input_features):
    scaler = joblib.load("scaler.pkl")
    model = load_model("imbalanced_dl_model.h5")

    input_features = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_features)

    prediction = model.predict(input_scaled)
    return int(prediction[0][0] > 0.5)


# -----------------------------
# 9. Main Execution
# -----------------------------
def main():
    if not os.path.exists("imbalanced_noisy_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X_scaled, y = preprocess_data(df)

    model, X_test, y_test = train_model(X_scaled, y)
    evaluate_model(model, X_test, y_test)

    sample_input = np.random.randn(10)
    prediction = intelligent_predict(sample_input)
    print("\nSample Prediction:", prediction)


if __name__ == "__main__":
    main()
