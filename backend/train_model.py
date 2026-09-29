import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================
# 1. LOAD DATASET
# =========================

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print(data.head())
print("\nDataset shape:", data.shape)


# =========================
# 2. INPUT AND OUTPUT
# =========================

X = data.drop("Disease", axis=1)

y = data["Disease"]


# =========================
# 3. COLUMNS
# =========================

categorical_columns = [
    "Fever",
    "Cough",
    "Fatigue",
    "Difficulty Breathing",
    "Gender"
]

numeric_columns = [
    "Age"
]


# =========================
# 4. PREPROCESSING
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numeric",
            "passthrough",
            numeric_columns
        )
    ]
)


# =========================
# 5. AI MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# =========================
# 6. COMPLETE PIPELINE
# =========================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================
# 7. SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 8. TRAIN
# =========================

print("\nTraining AI model...")

pipeline.fit(X_train, y_train)


# =========================
# 9. TEST
# =========================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAI Model Accuracy:", accuracy)


# =========================
# 10. SAVE MODEL
# =========================

joblib.dump(
    pipeline,
    "model.pkl"
)

print("\nmodel.pkl created successfully!")
print("Training completed!")