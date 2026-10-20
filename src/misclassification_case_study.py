"""Analyze prediction changes caused by full random oversampling (seed 42).

Run from the senior-seminar-2 repository root:
    python src/misclassification_case_study.py

Only aggregate data and figures are written to results/. Message text is written
locally to ~/sms_spam_case_study_private.csv (outside the Git repository).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler

DATA = Path('data/sms+spam+collection/SMSSpamCollection')
RESULTS = Path('results')
PRIVATE = Path.home() / 'sms_spam_case_study_private.csv'
SEED = 42


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=['ham', 'spam']).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                precision=precision_score(y, pred, pos_label='spam'),
                recall=recall_score(y, pred, pos_label='spam'),
                f1_score=f1_score(y, pred, pos_label='spam'),
                false_positive_rate=fp/(fp+tn), accuracy=accuracy_score(y, pred))


def build_model(name):
    if name == 'Naive Bayes':
        return MultinomialNB(alpha=0.1)
    if name == 'Logistic Regression':
        return LogisticRegression(C=10.0, max_iter=1000)
    return LinearSVC(C=1.0)


def category(actual, before, after):
    if before == after:
        return 'Unchanged correct' if before == actual else 'Unchanged error'
    return 'Corrected by oversampling' if after == actual else 'Introduced by oversampling'


def main():
    if not DATA.exists():
        raise FileNotFoundError(f'Missing dataset: {DATA}. Run from the repository root.')
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, sep='\t', names=['label', 'message'])
    x_train, x_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=SEED,
        stratify=df['label'])
    tfidf = TfidfVectorizer()
    train_matrix = tfidf.fit_transform(x_train)
    test_matrix = tfidf.transform(x_test)
    over_matrix, over_labels = RandomOverSampler(random_state=SEED).fit_resample(
        train_matrix, y_train)
    summary, changes, private_rows = [], [], []
    for name in ['Naive Bayes', 'Logistic Regression', 'SVM']:
        original = build_model(name).fit(train_matrix, y_train).predict(test_matrix)
        balanced = build_model(name).fit(over_matrix, over_labels).predict(test_matrix)
        for condition, preds in [('Original', original), ('Balanced', balanced)]:
            summary.append({'model': name, 'training_condition': condition,
                            **metrics(y_test, preds)})
        rows = []
        for index, truth, message, p0, p1 in zip(
                x_test.index, y_test, x_test, original, balanced):
            status = category(truth, p0, p1)
            rows.append({'model': name, 'true_label': truth, 'change_category': status})
            if status != 'Unchanged correct':
                private_rows.append({'model': name, 'dataset_index': int(index),
                                     'true_label': truth, 'original_prediction': p0,
                                     'balanced_prediction': p1,
                                     'change_category': status, 'message': message})
        tally = pd.DataFrame(rows).groupby(
            ['true_label', 'change_category']).size().to_dict()
        for actual in ['ham', 'spam']:
            for status in ['Unchanged correct', 'Unchanged error',
                           'Corrected by oversampling', 'Introduced by oversampling']:
                changes.append({'model': name, 'true_label': actual,
                                'change_category': status,
                                'count': int(tally.get((actual, status), 0))})
        orig = summary[-2]
        bal = summary[-1]
        print(f'\n{name}: original FP={orig["fp"]}, FN={orig["fn"]}; '
              f'balanced FP={bal["fp"]}, FN={bal["fn"]}; '
              f'F1 {orig["f1_score"]:.6f} -> {bal["f1_score"]:.6f}')
    summary_df = pd.DataFrame(summary)
    changes_df = pd.DataFrame(changes)
    # Verify reproduction against the existing six-condition baseline, if present.
    baseline_path = RESULTS / 'model_comparison_results.csv'
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        checks = ['precision', 'recall', 'f1_score', 'false_positive_rate', 'accuracy']
        for row in summary:
            match = baseline[(baseline['model'] == row['model']) &
                             (baseline['training_condition'] == row['training_condition'])]
            if len(match) != 1:
                raise AssertionError(f'Expected one baseline for {row["model"]} {row["training_condition"]}')
            for metric in checks:
                if not np.isclose(float(match.iloc[0][metric]), row[metric], atol=1e-8, rtol=0):
                    raise AssertionError(f'Baseline mismatch: {row["model"]} {row["training_condition"]} {metric}: '
                                         f'{float(match.iloc[0][metric])} vs {row[metric]}')
        print('\nPASS: All six model conditions reproduced five baseline metrics.')
    else:
        print('\nNOTE: results/model_comparison_results.csv not found; baseline check skipped.')
    summary_df.to_csv(RESULTS / 'misclassification_case_summary.csv', index=False)
    changes_df.to_csv(RESULTS / 'misclassification_case_transitions.csv', index=False)
    pd.DataFrame(private_rows).to_csv(PRIVATE, index=False)
    plot = summary_df.pivot(index='model', columns='training_condition', values=['fp', 'fn'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(axes, ['fp', 'fn'], ['False positives (ham flagged as spam)',
                                                     'False negatives (spam missed)']):
        subset = plot[metric].reindex(['Naive Bayes', 'Logistic Regression', 'SVM'])
        subset[['Original', 'Balanced']].plot.bar(ax=ax, rot=0)
        ax.set_title(label)
        ax.set_xlabel('')
        ax.set_ylabel('Messages')
        ax.legend(title='Training')
    fig.suptitle('Seed 42: errors before and after full oversampling')
    fig.tight_layout()
    fig.savefig(RESULTS / 'misclassification_case_comparison.png', dpi=200)
    plt.close(fig)
    print('\nPrediction transition counts (by true class):')
    print(changes_df[changes_df['count'] > 0].to_string(index=False))
    print('\nSaved public aggregate outputs:')
    print('  results/misclassification_case_summary.csv')
    print('  results/misclassification_case_transitions.csv')
    print('  results/misclassification_case_comparison.png')
    print(f'Private message-level examples (DO NOT COMMIT): {PRIVATE}')
    print('Inspect that local CSV for specific examples; avoid sharing raw SMS text publicly.')


if __name__ == '__main__':
    main()
