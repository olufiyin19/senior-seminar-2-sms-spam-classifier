
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = "data/sms+spam+collection/SMSSpamCollection"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEEDS = [42, 43, 44, 45, 46]

# Ratio = number of spam training examples / ham examples.
# None means the original, unbalanced training data.
BALANCING_RATIOS = {
    "Original": None,
    "25%": 0.25,
    "40%": 0.40,
    "50%": 0.50,
    "75%": 0.75,
    "100%": 1.00,
}

MODEL_FACTORIES = {
    "Naive Bayes": lambda: MultinomialNB(alpha=0.1),
    "Logistic Regression": lambda: LogisticRegression(
        C=10.0,
        max_iter=1000,
        random_state=42,
    ),
    "SVM": lambda: LinearSVC(
        C=1.0,
        random_state=42,
    ),
}


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

messages = pd.read_csv(
    DATA_PATH,
    sep="\t",
    names=["label", "message"],
)

print("Dataset shape:", messages.shape)
print("\nOriginal class distribution:")
print(messages["label"].value_counts())


# --------------------------------------------------
# 3. Evaluation function
# --------------------------------------------------

def evaluate_predictions(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=["ham", "spam"],
    ).ravel()

    return {
        "precision": precision_score(
            y_true, predictions, pos_label="spam"
        ),
        "recall": recall_score(
            y_true, predictions, pos_label="spam"
        ),
        "f1_score": f1_score(
            y_true, predictions, pos_label="spam"
        ),
        "accuracy": accuracy_score(y_true, predictions),
        "false_positive_rate": fp / (fp + tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


# --------------------------------------------------
# 4. Run repeated experiments
# --------------------------------------------------

experiment_results = []

for seed in RANDOM_SEEDS:
    print(f"\nRunning experiments for seed {seed}...", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        messages["message"],
        messages["label"],
        test_size=0.2,
        random_state=seed,
        stratify=messages["label"],
    )

    # Fit TF-IDF on training messages only.
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    original_counts = y_train.value_counts()
    ham_count = int(original_counts["ham"])
    spam_count = int(original_counts["spam"])

    for condition, ratio in BALANCING_RATIOS.items():

        if ratio is None:
            X_fit = X_train_tfidf
            y_fit = y_train
        else:
            target_spam_count = int(np.ceil(ham_count * ratio))

            # Oversampling cannot reduce the minority class.
            if target_spam_count <= spam_count:
                raise ValueError(
                    f"Target ratio {ratio} does not exceed "
                    f"the original spam-to-ham ratio."
                )

            oversampler = RandomOverSampler(
                sampling_strategy={"spam": target_spam_count},
                random_state=seed,
            )

            X_fit, y_fit = oversampler.fit_resample(
                X_train_tfidf,
                y_train,
            )

        for model_name, make_model in MODEL_FACTORIES.items():
            model = make_model()
            model.fit(X_fit, y_fit)

            predictions = model.predict(X_test_tfidf)

            metrics = evaluate_predictions(
                y_test,
                predictions,
            )

            experiment_results.append({
                "seed": seed,
                "model": model_name,
                "training_condition": condition,
                "spam_to_ham_ratio": (
                    spam_count / ham_count
                    if ratio is None
                    else ratio
                ),
                "training_samples": len(y_fit),
                "test_samples": len(y_test),
                **metrics,
            })

    print(f"Completed seed {seed}.")


# --------------------------------------------------
# 5. Save detailed results
# --------------------------------------------------

results_df = pd.DataFrame(experiment_results)

results_path = RESULTS_DIR / "balancing_experiment_results.csv"
results_df.to_csv(results_path, index=False)

print("\nTotal evaluations:", len(results_df))
print("Saved:", results_path)


# --------------------------------------------------
# 6. Summarize results across seeds
# --------------------------------------------------

metric_columns = [
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "false_positives",
    "false_negatives",
]

summary = (
    results_df.groupby(
        ["model", "training_condition"],
        as_index=False,
    )[metric_columns]
    .agg(["mean", "std"])
)

summary.columns = [
    "_".join(str(part) for part in column if part)
    if isinstance(column, tuple)
    else column
    for column in summary.columns
]

summary_path = RESULTS_DIR / "balancing_experiment_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nMean performance across five seeds:")
print(
    results_df.groupby(
        ["model", "training_condition"]
    )[["precision", "recall", "f1_score", "false_positive_rate"]]
    .mean()
    .round(4)
    .to_string()
)

print("\nSaved:", summary_path)


# --------------------------------------------------
# 7. Plot F1-score by balancing ratio
# --------------------------------------------------

condition_order = list(BALANCING_RATIOS.keys())
x_positions = np.arange(len(condition_order))

plt.figure(figsize=(10, 6))

for model_name in MODEL_FACTORIES:
    subset = results_df[
        results_df["model"] == model_name
    ]

    means = (
        subset.groupby("training_condition")["f1_score"]
        .mean()
        .reindex(condition_order)
    )

    stds = (
        subset.groupby("training_condition")["f1_score"]
        .std()
        .reindex(condition_order)
    )

    plt.errorbar(
        x_positions,
        means,
        yerr=stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label=model_name,
    )

plt.xticks(x_positions, condition_order)
plt.xlabel("Training Class-Balancing Condition")
plt.ylabel("Mean Spam F1-Score")
plt.title("Spam F1-Score Across Balancing Ratios")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "balancing_ratio_f1_comparison.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 8. Plot recall and false-positive trade-off
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for model_name in MODEL_FACTORIES:
    subset = results_df[
        results_df["model"] == model_name
    ]

    grouped = subset.groupby("training_condition")

    recall_means = (
        grouped["recall"]
        .mean()
        .reindex(condition_order)
    )

    fpr_means = (
        grouped["false_positive_rate"]
        .mean()
        .reindex(condition_order)
    )

    axes[0].plot(
        x_positions,
        recall_means,
        marker="o",
        label=model_name,
    )

    axes[1].plot(
        x_positions,
        fpr_means,
        marker="o",
        label=model_name,
    )

axes[0].set_title("Spam Recall")
axes[0].set_ylabel("Mean Recall")

axes[1].set_title("False-Positive Rate")
axes[1].set_ylabel("Mean False-Positive Rate")

for ax in axes:
    ax.set_xticks(x_positions, condition_order)
    ax.set_xlabel("Balancing Condition")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "balancing_recall_fpr_tradeoff.png",
    dpi=300,
)
plt.close()

print("\nSaved both comparison graphs.")
