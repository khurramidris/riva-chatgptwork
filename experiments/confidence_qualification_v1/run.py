"""Retrospective error-model qualification; see PROTOCOL.md for claim boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rival.research.calibration import VectorizedPersonaCalibrator
from rival.research.firewall import opinionqa_family_split

PARENT_REVISION = '902a3eb975b9e765f5a0c41e66afb6abe2ead5c6'
ROLES = ('population_fit', 'error_fit', 'bound_calibration', 'final_test')
NUMERIC_FEATURES = ('entropy', 'max_probability', 'top_two_margin',
                    'concentration', 'log_word_count', 'similarity_to_population_fit',
                    'raw_to_calibrated_tvd')


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def tvd(prediction, truth):
    return 0.5 * np.abs(np.asarray(prediction) - np.asarray(truth)).sum(axis=1)


def purge_earlier_families(roles, families):
    roles = np.asarray(roles, dtype=int).copy()
    families = np.asarray(families)
    for family in np.unique(families):
        members = families == family
        roles[members & (roles < roles[members].max())] = -1
    return roles


def make_splits(ids, texts):
    split = opinionqa_family_split(ids, texts, folds=20,
                                    seed='rival-confidence-20260905-v1')
    families = np.asarray(split.family_ids)
    family_roles = np.select([split.folds < 10, split.folds < 14, split.folds < 17],
                             [0, 1, 2], default=3)
    waves = np.array([int(re.search(r'_W(\d+)', item)[1]) for item in ids])
    if set(waves) != {26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92}:
        raise ValueError('unexpected wave set; revise protocol before execution')
    wave_roles = np.select([waves <= 41, waves <= 49, waves <= 54], [0, 1, 2], default=3)
    wave_roles = purge_earlier_families(wave_roles, families)
    schemes = {'family': family_roles, 'wave': wave_roles}
    for scheme, roles in schemes.items():
        for family in np.unique(families):
            used = set(roles[(families == family) & (roles >= 0)])
            if len(used) > 1:
                raise AssertionError(f'{scheme}: family crosses roles')
        if any(np.sum(roles == role) < 10 for role in range(4)):
            raise ValueError(f'{scheme}: insufficient rows in a prespecified role')
    return families, waves, schemes


def upper_residual_quantile(errors, predicted, families, alpha=0.10):
    """One score per complete family; +1 rank gives finite-sample correction."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    errors, predicted, families = map(np.asarray, (errors, predicted, families))
    if len(errors) != len(predicted) or len(errors) != len(families) or not len(errors):
        raise ValueError('nonempty aligned calibration arrays required')
    if not (np.isfinite(errors).all() and np.isfinite(predicted).all()):
        raise ValueError('finite calibration arrays required')
    scores = sorted(float(max(0.0, (errors - predicted)[families == f].max()))
                    for f in np.unique(families))
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return (1.0 if rank > len(scores) else scores[rank - 1]), len(scores)


def features(probabilities, raw_probabilities, texts, fit_mask):
    """No human outcomes or human sample counts are accepted by this function."""
    p = np.asarray(probabilities)
    sorted_p = np.sort(p, axis=1)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                 stop_words='english', max_features=5000)
    fit_text = vectorizer.fit_transform(texts[fit_mask])
    similarity = cosine_similarity(vectorizer.transform(texts), fit_text).max(axis=1)
    return np.column_stack([
        -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1) / np.log(p.shape[1]),
        p.max(axis=1), sorted_p[:, -1] - sorted_p[:, -2], (p * p).sum(axis=1),
        np.log1p([len(text.split()) for text in texts]), similarity,
        tvd(raw_probabilities, p),
    ])


def error_predictions(numeric, texts, errors, fit_mask):
    # The caller supplies B errors only, so D errors cannot be consumed accidentally.
    if len(errors) != int(fit_mask.sum()):
        raise ValueError('error labels must contain only error_fit rows')
    scaler = StandardScaler().fit(numeric[fit_mask])
    x_num = scaler.transform(numeric)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                 stop_words='english', max_features=5000)
    vectorizer.fit(texts[fit_mask])
    x_text = vectorizer.transform(texts)
    matrices = {'numeric_ridge': sparse.csr_matrix(x_num),
                'text_ridge': x_text,
                'combined_ridge': sparse.hstack([sparse.csr_matrix(x_num), x_text]).tocsr()}
    predictions = {'mean_error': np.full(len(texts), float(np.mean(errors)))}
    for name, matrix in matrices.items():
        model = Ridge(alpha=10.0, solver='lsqr').fit(matrix[fit_mask], errors)
        predictions[name] = np.clip(model.predict(matrix), 0, 1)
    return predictions


def summarize_error_model(errors, predictions, upper, families, ids):
    good = errors < 0.16
    unique_families = np.unique(families)
    order = np.lexsort((ids, predictions))
    selective = []
    for fraction in (0.25, 0.50, 0.75, 1.0):
        selected = order[:math.ceil(len(order) * fraction)]
        selective.append({'retention': fraction, 'n': len(selected),
                          'mean_tvd': float(errors[selected].mean())})
    accepted = upper <= 0.16
    return {
        'rmse': float(np.sqrt(np.mean((errors - predictions) ** 2))),
        'family_macro_rmse': float(np.sqrt(np.mean([
            np.mean((errors[families == f] - predictions[families == f]) ** 2)
            for f in unique_families]))),
        'auroc_tvd_below_0_16': float(roc_auc_score(good, -predictions)) if len(set(good)) == 2 else None,
        'observed_good_rate': float(good.mean()),
        'mean_expected_tvd': float(predictions.mean()),
        'mean_upper_bound': float(upper.mean()),
        'mean_bound_increment': float(np.mean(upper - predictions)),
        'question_coverage': float(np.mean(errors <= upper)),
        'complete_family_coverage': float(np.mean([
            np.all(errors[families == f] <= upper[families == f]) for f in unique_families])),
        'final_families': len(unique_families),
        'accepted_questions': int(accepted.sum()),
        'accepted_mean_tvd': float(errors[accepted].mean()) if accepted.any() else None,
        'accepted_good_fraction': float(good[accepted].mean()) if accepted.any() else None,
        'risk_by_retention_descriptive': selective,
    }


def evaluate_scheme(name, ids, texts, human, answers, families, roles, out):
    a, b, c, d = (roles == role for role in range(4))
    calibrator = VectorizedPersonaCalibrator(max_iter=150, learning_rate=1.0)
    calibrator.fit(human[a], answers[a])
    raw = calibrator.raw_predict(answers, human.shape[1])
    calibrated = calibrator.predict(answers)
    historical = np.broadcast_to(human[a].mean(axis=0), human.shape)
    uniform = np.full_like(human, 1.0 / human.shape[1])
    numeric = features(calibrated, raw, texts, a)
    prediction_by_model = error_predictions(numeric, texts, tvd(calibrated[b], human[b]), b)
    calibration_errors = tvd(calibrated[c], human[c])
    bounds = {}
    prediction_rows = []
    for model, predicted in prediction_by_model.items():
        q, n_families = upper_residual_quantile(calibration_errors, predicted[c], families[c])
        bounds[model] = np.clip(predicted + q, 0, 1)
        for index in np.flatnonzero(d):
            prediction_rows.append({'question_id': ids[index], 'family_id': families[index],
                                    'model': model, 'predicted_tvd': float(predicted[index]),
                                    'upper_tvd': float(bounds[model][index]),
                                    'calibration_families': n_families,
                                    'residual_quantile': q,
                                    'distribution': calibrated[index].tolist()})
    # Save all predictions and upper bounds BEFORE scoring final human outcomes.
    predictions_path = out / f'{name}_predictions.json'
    write_json(predictions_path, prediction_rows)
    final_errors = tvd(calibrated[d], human[d])
    error_results = {model: summarize_error_model(final_errors, predicted[d], bounds[model][d],
                                                  families[d], ids[d])
                     for model, predicted in prediction_by_model.items()}
    distributions = {}
    for model, prediction in [('raw_personas', raw), ('calibrated_personas', calibrated),
                              ('historical_positions', historical), ('uniform', uniform)]:
        err = tvd(prediction[d], human[d])
        distributions[model] = {'mean_tvd': float(err.mean()), 'median_tvd': float(np.median(err)),
                                'p90_tvd': float(np.quantile(err, 0.9))}
    baseline = error_results['mean_error']
    challenger = error_results['combined_ridge']
    gate = {'rmse_better_than_mean': challenger['rmse'] < baseline['rmse'],
            'complete_family_coverage_at_least_90pct': challenger['complete_family_coverage'] >= 0.90,
            'at_least_ten_accepted_questions': challenger['accepted_questions'] >= 10}
    return {'role_counts': {role: int(np.sum(roles == i)) for i, role in enumerate(ROLES)},
            'purged_questions': int(np.sum(roles < 0)),
            'distribution_metrics': distributions, 'error_models': error_results,
            'prespecified_gate': gate, 'status': 'PASS' if all(gate.values()) else 'FAIL',
            'predictions_sha256': digest_bytes(predictions_path.read_bytes()),
            'interpretation': 'Exploratory aggregate prediction test; no prospective or commercial qualification.'}


def run(output_dir):
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError('Refusing to overwrite prior results; preserve the evaluation ledger.')
    source = ROOT / 'rival/datasets/opinionqa'
    qs_bytes = (source / 'questions.json').read_bytes()
    csv_bytes = (source / 'persona_answers.csv').read_bytes()
    qs = json.loads(qs_bytes)
    ids = np.array(sorted(qs))
    texts = np.array([qs[item]['question'] for item in ids])
    families, waves, schemes = make_splits(ids, texts)
    out.mkdir(parents=True)
    source_paths = [source / 'questions.json', source / 'persona_answers.csv',
                    ROOT / 'rival/research/calibration.py', ROOT / 'rival/research/firewall.py',
                    Path(__file__), Path(__file__).with_name('PROTOCOL.md')]
    manifest = {'schema': 'rival.confidence_qualification.v1', 'parent_revision': PARENT_REVISION,
                'sources': {str(p.relative_to(ROOT)): digest_bytes(p.read_bytes()) for p in source_paths},
                'features': NUMERIC_FEATURES,
                'rows': [{'question_id': item, 'family_id': family, 'wave': int(wave),
                          **{name: ROLES[int(roles[i])] if roles[i] >= 0 else 'purged'
                             for name, roles in schemes.items()}}
                         for i, (item, family, wave) in enumerate(zip(ids, families, waves))]}
    write_json(out / 'manifest.json', manifest)
    # Distribution labels are parsed only after the manifest has been recorded.
    human = np.array([[qs[item]['choice_counts'][str(k)] for k in range(1, 6)] for item in ids], float)
    human /= human.sum(axis=1, keepdims=True)
    frame = pd.read_csv(source / 'persona_answers.csv', index_col=0)
    if set(frame.index) != set(ids) or frame.shape[1] != 2058:
        raise ValueError('unexpected prediction matrix: abort rather than silently subset')
    answers = frame.loc[ids].to_numpy(dtype=int) - 1
    results = {name: evaluate_scheme(name, ids, texts, human, answers, families, roles, out)
               for name, roles in schemes.items()}
    report = {'schema': 'rival.confidence_qualification.v1', 'status':
              'PASS' if all(v['status'] == 'PASS' for v in results.values()) else 'FAIL',
              'questions': len(ids), 'synthetic_personas': answers.shape[1],
              'provider_calls': 0, 'provider_cost_usd': 0,
              'manifest_sha256': digest_bytes((out / 'manifest.json').read_bytes()),
              'splits': results,
              'limits': ['Released predictions, not a newly trained behavior model.',
                         'Historical public survey counts, not prospective outcomes.',
                         'Error bounds require family exchangeability; wave drift can break coverage.',
                         'No claim of individual prediction, causal accuracy, or Simile parity.',
                         'No tuning on final results is allowed by this protocol.']}
    write_json(out / 'report.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args().output_dir)
