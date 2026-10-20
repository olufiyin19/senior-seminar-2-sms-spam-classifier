from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

DATA_PATH = Path('data/sms+spam+collection/SMSSpamCollection')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
METHODS = ['Original', 'Oversampling', 'Undersampling', 'Class Weighting']
MODELS = ['Naive Bayes', 'Logistic Regression', 'SVM']


def make_model(name, weighted=False):
    if name == 'Naive Bayes':
        if weighted:
            raise ValueError('MultinomialNB does not support class_weight directly')
        return MultinomialNB(alpha=0.1)
    if name == 'Logistic Regression':
        return LogisticRegression(C=10.0, max_iter=1000, random_state=42,
                                  class_weight='balanced' if weighted else None)
    if name == 'SVM':
        return LinearSVC(C=1.0, random_state=42,
                         class_weight='balanced' if weighted else None)
    raise ValueError(name)


def evaluate(y_true, predicted):
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=['ham', 'spam']).ravel()
    return {
        'precision': precision_score(y_true, predicted, pos_label='spam', zero_division=0),
        'recall': recall_score(y_true, predicted, pos_label='spam', zero_division=0),
        'f1_score': f1_score(y_true, predicted, pos_label='spam', zero_division=0),
        'accuracy': accuracy_score(y_true, predicted),
        'false_positive_rate': fp / (fp + tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'true_negatives': int(tn),
    }


def main():
    messages = pd.read_csv(DATA_PATH, sep='\t', names=['label', 'message'])
    assert set(messages['label'].unique()) == {'ham', 'spam'}
    previous_path = RESULTS_DIR / 'balancing_experiment_results.csv'
    if not previous_path.exists():
        raise FileNotFoundError(f'Baseline results missing: {previous_path}')
    previous = pd.read_csv(previous_path)
    records = []

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            messages['message'], messages['label'], test_size=0.2,
            random_state=seed, stratify=messages['label']
        )
        tfidf = TfidfVectorizer()
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)
        training_options = {
            'Original': (X_train_vec, y_train),
            'Oversampling': RandomOverSampler(random_state=seed).fit_resample(X_train_vec, y_train),
            'Undersampling': RandomUnderSampler(random_state=seed).fit_resample(X_train_vec, y_train),
            'Class Weighting': (X_train_vec, y_train),
        }
        for method, (X_fit, y_fit) in training_options.items():
            for model_name in MODELS:
                if method == 'Class Weighting' and model_name == 'Naive Bayes':
                    continue
                model = make_model(model_name, weighted=(method == 'Class Weighting'))
                model.fit(X_fit, y_fit)
                metrics = evaluate(y_test, model.predict(X_test_vec))
                records.append({
                    'seed': seed, 'model': model_name, 'method': method,
                    'train_samples': len(y_fit), 'train_ham': int((y_fit == 'ham').sum()),
                    'train_spam': int((y_fit == 'spam').sum()),
                    'test_samples': len(y_test), **metrics,
                })
        print(f'Finished seed {seed}', flush=True)

    results = pd.DataFrame(records)
    assert len(results) == 55, f'Expected 55 evaluations; got {len(results)}'

    # Check every comparable baseline against the saved midterm experiment.
    comparisons = 0
    metrics_to_check = ['precision', 'recall', 'f1_score', 'accuracy',
                        'false_positive_rate', 'false_positives', 'false_negatives',
                        'true_positives', 'true_negatives']
    for seed in SEEDS:
        for model_name in MODELS:
            for method, old_condition in [('Original', 'Original'), ('Oversampling', '100%')]:
                new_row = results.loc[(results.seed == seed) & (results.model == model_name) &
                                      (results.method == method)]
                old_row = previous.loc[(previous.seed == seed) & (previous.model == model_name) &
                                       (previous.training_condition == old_condition)]
                if len(new_row) != 1 or len(old_row) != 1:
                    raise AssertionError(f'Missing baseline for {seed}, {model_name}, {method}')
                for metric in metrics_to_check:
                    if not np.isclose(float(new_row.iloc[0][metric]), float(old_row.iloc[0][metric]),
                                      rtol=1e-7, atol=1e-8):
                        raise AssertionError(f'Baseline mismatch: {seed}, {model_name}, {method}, {metric}')
                comparisons += 1
    print(f'PASS: {comparisons} baseline configurations reproduced all {len(metrics_to_check)} metrics.')

    detail_path = RESULTS_DIR / 'balancing_method_results.csv'
    results.to_csv(detail_path, index=False)
    summary = (results.groupby(['model', 'method'], as_index=False)
               .agg(f1_mean=('f1_score', 'mean'), f1_std=('f1_score', 'std'),
                    precision_mean=('precision', 'mean'), recall_mean=('recall', 'mean'),
                    fpr_mean=('false_positive_rate', 'mean'),
                    fp_mean=('false_positives', 'mean'), fn_mean=('false_negatives', 'mean')))
    summary_path = RESULTS_DIR / 'balancing_method_summary.csv'
    summary.to_csv(summary_path, index=False)
    print('\nMean metrics across five seeds:')
    print(summary.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(MODELS))
    offsets = {'Original': -0.27, 'Oversampling': -0.09,
               'Undersampling': 0.09, 'Class Weighting': 0.27}
    for method in METHODS:
        values = []
        errors = []
        for model_name in MODELS:
            row = summary[(summary.model == model_name) & (summary.method == method)]
            values.append(float(row.iloc[0].f1_mean) if len(row) else np.nan)
            errors.append(float(row.iloc[0].f1_std) if len(row) else 0)
        ax.bar(x + offsets[method], values, width=0.18, yerr=errors, capsize=3, label=method)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel('Mean spam F1-score (five splits)')
    ax.set_title('Effect of Imbalance-Handling Method on Spam F1')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'balancing_method_f1_comparison.png', dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model_name in MODELS:
        subset = summary[summary.model == model_name].set_index('method').reindex(METHODS)
        axes[0].plot(METHODS, subset.recall_mean, marker='o', label=model_name)
        axes[1].plot(METHODS, subset.fpr_mean, marker='o', label=model_name)
    axes[0].set_title('Spam Recall')
    axes[1].set_title('False-Positive Rate')
    for ax in axes:
        ax.tick_params(axis='x', rotation=20)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'balancing_method_recall_fpr.png', dpi=250)
    plt.close(fig)
    print('\nSaved:', detail_path, summary_path)
    print('Saved: results/balancing_method_f1_comparison.png')
    print('Saved: results/balancing_method_recall_fpr.png')


if __name__ == '__main__':
    main()
