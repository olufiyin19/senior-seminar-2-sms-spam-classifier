"""Paired, descriptive stability analysis of Seminar II spam-balancing experiments.

Run from the repository root:
    python src/balancing_stability_analysis.py

Requires only existing experiment CSVs; no models are retrained.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path('results')
NEW_PATH = RESULTS / 'balancing_method_results.csv'
OLD_PATH = RESULTS / 'balancing_experiment_results.csv'
MODELS = ['Naive Bayes', 'Logistic Regression', 'SVM']
METHODS = ['Oversampling', 'Undersampling', 'Class Weighting']
SEEDS = [42, 43, 44, 45, 46]


def validate(df, name, condition_column, expected_conditions):
    needed = {'seed', 'model', condition_column, 'f1_score', 'precision', 'recall',
              'false_positive_rate', 'false_positives', 'false_negatives'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'{name} missing columns: {sorted(missing)}')
    if df.duplicated(['seed', 'model', condition_column]).any():
        raise ValueError(f'{name} has duplicate seed/model/condition rows')
    for model in MODELS:
        for seed in SEEDS:
            for condition in expected_conditions(model):
                rows = df[(df.seed == seed) & (df.model == model) &
                          (df[condition_column].astype(str) == condition)]
                if len(rows) != 1:
                    raise ValueError(f'{name}: expected one row for {model}, {seed}, {condition}; got {len(rows)}')


def main():
    for path in [NEW_PATH, OLD_PATH]:
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}. Run the prior experiments first.')
    new = pd.read_csv(NEW_PATH)
    old = pd.read_csv(OLD_PATH)
    validate(new, 'Method experiment', 'method',
             lambda m: ['Original', 'Oversampling', 'Undersampling'] +
             ([] if m == 'Naive Bayes' else ['Class Weighting']))
    ratios = ['Original', '25%', '40%', '50%', '75%', '100%']
    validate(old, 'Midterm ratio experiment', 'training_condition', lambda m: ratios)

    # Paired comparisons: within each model and split, compare each method to
    # the *same* original baseline. This controls for differences among splits.
    baseline = new[new.method == 'Original'].set_index(['model', 'seed'])
    rows = []
    for _, r in new.iterrows():
        if r['method'] == 'Original':
            continue
        b = baseline.loc[(r['model'], r['seed'])]
        rows.append({
            'model': r['model'], 'seed': int(r['seed']), 'method': r['method'],
            'delta_f1': float(r['f1_score'] - b['f1_score']),
            'delta_precision': float(r['precision'] - b['precision']),
            'delta_recall': float(r['recall'] - b['recall']),
            'delta_fpr': float(r['false_positive_rate'] - b['false_positive_rate']),
            'extra_false_positives': int(r['false_positives'] - b['false_positives']),
            'fewer_missed_spam': int(b['false_negatives'] - r['false_negatives']),
        })
    paired = pd.DataFrame(rows)
    if len(paired) != 40:
        raise AssertionError(f'Expected 40 method-vs-original paired comparisons; got {len(paired)}')
    summary = (paired.groupby(['model', 'method'], as_index=False)
               .agg(mean_delta_f1=('delta_f1', 'mean'),
                    std_delta_f1=('delta_f1', 'std'),
                    min_delta_f1=('delta_f1', 'min'),
                    max_delta_f1=('delta_f1', 'max'),
                    seeds_f1_improved=('delta_f1', lambda s: int((s > 0).sum())),
                    mean_delta_recall=('delta_recall', 'mean'),
                    mean_delta_fpr=('delta_fpr', 'mean'),
                    mean_extra_fp=('extra_false_positives', 'mean'),
                    mean_fewer_fn=('fewer_missed_spam', 'mean')))

    # Extend stability analysis to all six previously tested oversampling ratios.
    ratio_baseline = old[old.training_condition.astype(str) == 'Original'].set_index(['model', 'seed'])
    ratio_rows = []
    for _, r in old.iterrows():
        if str(r['training_condition']) == 'Original':
            continue
        b = ratio_baseline.loc[(r['model'], r['seed'])]
        ratio_rows.append({
            'model': r['model'], 'seed': int(r['seed']), 'ratio': str(r['training_condition']),
            'delta_f1': float(r['f1_score'] - b['f1_score']),
            'extra_false_positives': int(r['false_positives'] - b['false_positives']),
            'fewer_missed_spam': int(b['false_negatives'] - r['false_negatives']),
        })
    ratio_paired = pd.DataFrame(ratio_rows)
    if len(ratio_paired) != 75:
        raise AssertionError(f'Expected 75 ratio-vs-original comparisons; got {len(ratio_paired)}')
    ratio_summary = (ratio_paired.groupby(['model', 'ratio'], as_index=False)
                     .agg(mean_delta_f1=('delta_f1', 'mean'),
                          std_delta_f1=('delta_f1', 'std'),
                          seeds_f1_improved=('delta_f1', lambda s: int((s > 0).sum())),
                          mean_extra_fp=('extra_false_positives', 'mean'),
                          mean_fewer_fn=('fewer_missed_spam', 'mean')))

    # Baseline consistency: these two CSVs should reproduce the same 30
    # original and 100%-oversampling rows, as established in Experiment A.
    checks = 0
    for model in MODELS:
        for seed in SEEDS:
            for method, ratio in [('Original', 'Original'), ('Oversampling', '100%')]:
                a = new[(new.model == model) & (new.seed == seed) & (new.method == method)].iloc[0]
                b = old[(old.model == model) & (old.seed == seed) &
                        (old.training_condition.astype(str) == ratio)].iloc[0]
                for metric in ['f1_score', 'precision', 'recall', 'false_positive_rate',
                               'false_positives', 'false_negatives']:
                    if not np.isclose(float(a[metric]), float(b[metric]), rtol=1e-7, atol=1e-8):
                        raise AssertionError(f'Baseline mismatch: {model}, seed {seed}, {method}, {metric}')
                checks += 1
    print(f'PASS: {checks} matching baseline configurations across both CSV files.')

    paths = {
        'paired': RESULTS / 'balancing_stability_paired.csv',
        'summary': RESULTS / 'balancing_stability_summary.csv',
        'ratio_paired': RESULTS / 'balancing_ratio_stability_paired.csv',
        'ratio_summary': RESULTS / 'balancing_ratio_stability_summary.csv',
    }
    for key, df in [('paired', paired), ('summary', summary),
                    ('ratio_paired', ratio_paired), ('ratio_summary', ratio_summary)]:
        df.to_csv(paths[key], index=False)

    print('\nMethod stability versus the same-split original baseline:')
    print(summary.round(4).to_string(index=False))
    print('\nOversampling ratio stability versus same-split original baseline:')
    print(ratio_summary.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        subset = paired[paired.model == model]
        methods = [m for m in METHODS if m in subset.method.unique()]
        for x, method in enumerate(methods):
            vals = subset[subset.method == method].sort_values('seed').delta_f1.to_numpy()
            ax.scatter(np.full(len(vals), x), vals, s=42, alpha=0.7)
            ax.plot([x - .22, x + .22], [vals.mean()] * 2, linewidth=3)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(range(len(methods)), methods, rotation=25, ha='right')
        ax.set_title(model)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Paired spam F1 difference vs original')
    fig.suptitle('Stability of Imbalance-Handling Methods Across Five Matched Splits')
    fig.tight_layout()
    fig.savefig(RESULTS / 'balancing_stability_paired_f1.png', dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, model in zip(axes, MODELS):
        sub = paired[paired.model == model]
        for method in METHODS:
            part = sub[sub.method == method]
            if part.empty:
                continue
            ax.scatter(part.extra_false_positives, part.fewer_missed_spam,
                       label=method, s=48, alpha=.75)
        ax.axvline(0, color='gray', linewidth=1)
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_title(model)
        ax.set_xlabel('Additional legitimate messages flagged')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Additional spam messages detected')
    fig.suptitle('Per-Split Trade-off Relative to Original Training')
    fig.tight_layout()
    fig.savefig(RESULTS / 'balancing_stability_error_tradeoff.png', dpi=250)
    plt.close(fig)

    print('\nSaved four CSVs and two PNGs in results/:')
    for path in paths.values():
        print(' ', path)
    print(' ', RESULTS / 'balancing_stability_paired_f1.png')
    print(' ', RESULTS / 'balancing_stability_error_tradeoff.png')
    print('\nInterpretation: descriptive results from five splits, not a statistical-significance test.')


if __name__ == '__main__':
    main()
