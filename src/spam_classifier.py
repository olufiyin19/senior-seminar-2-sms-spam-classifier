import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    make_scorer
)
from imblearn.over_sampling import RandomOverSampler

# Evaluation function
def evaluate_model(y_true, predictions):
    precision = precision_score(
        y_true, predictions, pos_label="spam"
    )
    recall = recall_score(
        y_true, predictions, pos_label="spam"
    )
    f1 = f1_score(
        y_true, predictions, pos_label="spam"
    )
    accuracy = accuracy_score(
        y_true, predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=["ham", "spam"]
    ).ravel()

    false_positive_rate = fp / (fp + tn)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy
    }

# 5-fold stratified cross-validation for training-only model tuning
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Load sms dataset in the classifier file
messages = pd.read_csv(
    "data/sms+spam+collection/SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

# X: Messages, y: Labels
X = messages["message"]
y = messages["label"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Tune Naive Bayes using training data only
nb_tuning_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

nb_param_grid = {
    "classifier__alpha": [0.1, 0.5, 1.0]
}

spam_f1_scorer = make_scorer(
    f1_score,
    pos_label="spam"
)

nb_grid_search = GridSearchCV(
    nb_tuning_pipeline,
    nb_param_grid,
    cv=cv,
    scoring=spam_f1_scorer
)

nb_grid_search.fit(
    X_train,
    y_train
)

print("\nNaive Bayes tuning results:")
print(
    "Best alpha:",
    nb_grid_search.best_params_["classifier__alpha"]
)
print(
    "Best cross-validation F1:",
    nb_grid_search.best_score_
)

# Tune Logistic Regression using training data only
logistic_tuning_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_param_grid = {
    "classifier__C": [0.1, 1.0, 10.0]
}

logistic_grid_search = GridSearchCV(
    logistic_tuning_pipeline,
    logistic_param_grid,
    cv=cv,
    scoring=spam_f1_scorer
)

logistic_grid_search.fit(
    X_train,
    y_train
)

print("\nLogistic Regression tuning results:")
print(
    "Best C:",
    logistic_grid_search.best_params_["classifier__C"]
)
print(
    "Best cross-validation F1:",
    logistic_grid_search.best_score_
)

# Tune SVM using training data only
svm_tuning_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LinearSVC())
])

svm_param_grid = {
    "classifier__C": [0.1, 1.0, 10.0]
}

svm_grid_search = GridSearchCV(
    svm_tuning_pipeline,
    svm_param_grid,
    cv=cv,
    scoring=spam_f1_scorer
)

svm_grid_search.fit(
    X_train,
    y_train
)

print("\nSVM tuning results:")
print(
    "Best C:",
    svm_grid_search.best_params_["classifier__C"]
)
print(
    "Best cross-validation F1:",
    svm_grid_search.best_score_
)

# Balance only the training data using random oversampling
oversampler = RandomOverSampler(random_state=42)

X_train_balanced, y_train_balanced = oversampler.fit_resample(
    X_train_tfidf,
    y_train
)

print("\nOriginal training class distribution:")
print(y_train.value_counts())

print("\nBalanced training class distribution:")
print(y_train_balanced.value_counts())

print("\nTest class distribution (unchanged):")
print(y_test.value_counts())

model = MultinomialNB(alpha=0.1)

model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_test_tfidf)
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(y_test, predictions)

plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.savefig("results/naive_bayes_confusion_matrix.png")
plt.close()

# Logistic Regression baseline
logistic_model = LogisticRegression(
    C=10.0,
    max_iter=1000
)

logistic_model.fit(X_train_tfidf, y_train)
logistic_predictions = logistic_model.predict(X_test_tfidf)

print("\nLogistic Regression Results:")
print(classification_report(y_test, logistic_predictions))

ConfusionMatrixDisplay.from_predictions(y_test, logistic_predictions)

plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("results/logistic_regression_confusion_matrix.png")
plt.close()

# SVM baseline
svm_model = LinearSVC(C=1.0)

svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)

print("\nSVM Results:")
print(classification_report(y_test, svm_predictions))

ConfusionMatrixDisplay.from_predictions(y_test, svm_predictions)

plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("results/svm_confusion_matrix.png")
plt.close()

# Naive Bayes with balanced training data
balanced_nb_model = MultinomialNB(alpha=0.1)

balanced_nb_model.fit(X_train_balanced, y_train_balanced)
balanced_nb_predictions = balanced_nb_model.predict(X_test_tfidf)

print("\nNaive Bayes Results - Balanced Training:")
print(classification_report(y_test, balanced_nb_predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    balanced_nb_predictions
)

plt.title("Naive Bayes - Balanced Training Confusion Matrix")
plt.tight_layout()
plt.savefig("results/naive_bayes_balanced_confusion_matrix.png")
plt.close()

# Logistic Regression with balanced training data
balanced_logistic_model = LogisticRegression(
    C=10.0,
    max_iter=1000
)

balanced_logistic_model.fit(X_train_balanced, y_train_balanced)
balanced_logistic_predictions = balanced_logistic_model.predict(X_test_tfidf)

print("\nLogistic Regression Results - Balanced Training:")
print(classification_report(y_test, balanced_logistic_predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    balanced_logistic_predictions
)

plt.title("Logistic Regression - Balanced Training Confusion Matrix")
plt.tight_layout()
plt.savefig("results/logistic_regression_balanced_confusion_matrix.png")
plt.close()

# SVM with balanced training data
balanced_svm_model = LinearSVC(C=1.0)

balanced_svm_model.fit(X_train_balanced, y_train_balanced)
balanced_svm_predictions = balanced_svm_model.predict(X_test_tfidf)

print("\nSVM Results - Balanced Training:")
print(classification_report(y_test, balanced_svm_predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    balanced_svm_predictions
)

plt.title("SVM - Balanced Training Confusion Matrix")
plt.tight_layout()
plt.savefig("results/svm_balanced_confusion_matrix.png")
plt.close()

# Compare all model results
results = []

model_predictions = [
    ("Naive Bayes", "Original", predictions),
    ("Naive Bayes", "Balanced", balanced_nb_predictions),
    ("Logistic Regression", "Original", logistic_predictions),
    ("Logistic Regression", "Balanced", balanced_logistic_predictions),
    ("SVM", "Original", svm_predictions),
    ("SVM", "Balanced", balanced_svm_predictions)
]

for model_name, condition, model_preds in model_predictions:
    metrics = evaluate_model(y_test, model_preds)

    results.append({
        "model": model_name,
        "training_condition": condition,
        **metrics
    })

results_df = pd.DataFrame(results)

print("\nFinal Model Comparison:")
print(results_df.to_string(index=False))
results_df.to_csv(
    "results/model_comparison_results.csv",
    index=False
)

print("\nResults saved to results/model_comparison_results.csv")

# Create comparison chart for spam-class metrics
plot_data = results_df[
    ["model", "training_condition", "precision", "recall", "f1_score"]
].copy()

plot_data["label"] = (
    plot_data["model"] + "\n" + plot_data["training_condition"]
)

ax = plot_data.set_index("label")[
    ["precision", "recall", "f1_score"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Spam Detection Performance: Original vs Balanced Training")
plt.xlabel("Model and Training Condition")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.close()

print("Comparison figure saved to results/model_comparison.png")