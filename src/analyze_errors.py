
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Reproduce the existing train/test split
messages = pd.read_csv(
    "data/sms+spam+collection/SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

X_train, X_test, y_train, y_test = train_test_split(
    messages["message"],
    messages["label"],
    test_size=0.2,
    random_state=42,
    stratify=messages["label"]
)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Oversample training data only
oversampler = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = oversampler.fit_resample(
    X_train_tfidf, y_train
)

models = {
    "Naive Bayes": lambda: MultinomialNB(alpha=0.1),
    "Logistic Regression": lambda: LogisticRegression(
        C=10.0, max_iter=1000
    ),
    "SVM": lambda: LinearSVC(C=1.0)
}

training_conditions = {
    "Original": (X_train_tfidf, y_train),
    "Balanced": (X_train_balanced, y_train_balanced)
}

error_records = []
summary_records = []

for model_name, make_model in models.items():
    for condition, (X_fit, y_fit) in training_conditions.items():
        model = make_model()
        model.fit(X_fit, y_fit)
        predictions = model.predict(X_test_tfidf)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=["ham", "spam"]
        ).ravel()

        summary_records.append({
            "model": model_name,
            "training_condition": condition,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        })

        # Preserve original test-row identifiers
        for index, actual, predicted, message in zip(
            X_test.index,
            y_test,
            predictions,
            X_test
        ):
            if actual != predicted:
                error_records.append({
                    "model": model_name,
                    "training_condition": condition,
                    "message_id": int(index),
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "error_type": (
                        "False Negative"
                        if actual == "spam"
                        else "False Positive"
                    ),
                    "message": message
                })

summary_df = pd.DataFrame(summary_records)
errors_df = pd.DataFrame(error_records)

summary_df.to_csv(
    RESULTS_DIR / "error_analysis_summary.csv",
    index=False
)

errors_df.to_csv(
    RESULTS_DIR / "misclassified_messages.csv",
    index=False
)

print("\nError analysis summary:")
print(summary_df.to_string(index=False))

# Plot 1: False positives versus false negatives
plot_data = summary_df.copy()
plot_data["label"] = (
    plot_data["model"]
    + "\n"
    + plot_data["training_condition"]
)

ax = plot_data.set_index("label")[
    ["false_positives", "false_negatives"]
].plot(kind="bar", figsize=(11, 6))

ax.set_title("Spam Classification Errors: Original vs Balanced")
ax.set_xlabel("Model and Training Condition")
ax.set_ylabel("Number of Misclassified Messages")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "classification_error_comparison.png",
    dpi=300
)
plt.close()

# Plot 2: Changes in error counts after balancing
original = summary_df[
    summary_df["training_condition"] == "Original"
].set_index("model")

balanced = summary_df[
    summary_df["training_condition"] == "Balanced"
].set_index("model")

changes = balanced[
    ["false_positives", "false_negatives"]
] - original[
    ["false_positives", "false_negatives"]
]

ax = changes.plot(kind="bar", figsize=(9, 6))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Change in Classification Errors After Oversampling")
ax.set_xlabel("Classifier")
ax.set_ylabel("Balanced Minus Original Error Count")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "oversampling_error_tradeoff.png",
    dpi=300
)
plt.close()

print("\nSaved error-analysis CSV files and two graphs.")
print("Total misclassification records:", len(errors_df))
