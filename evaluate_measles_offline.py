#!/usr/bin/env python3
"""Offline, label-blind fusion of saved GPT/Gemini binary and top-3 predictions.

No API calls. Supports the shared measles_eval_common.py CSV schema and older
measles_chatgpt_*.csv files. Select four CSVs explicitly or discover them in a
results directory. Does not optimize rules against the test labels.

Example commands:
python evaluate_measles_offline.py \
    --results-dir ./measles_results \
    --weight-binary 2 \
    --weight-top3 1 \
    --threshold 0.5
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

CHANNELS = ('gpt_binary', 'gpt_top3', 'gemini_binary', 'gemini_top3')
VALID = {'measles', 'non_measles'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', type=Path, default=Path('measles_results'))
    for name in CHANNELS:
        p.add_argument('--' + name.replace('_', '-'), type=Path,
                       help='Explicit CSV for this channel (recommended if multiple runs exist)')
    p.add_argument('--output-dir', type=Path, default=Path('measles_fusion_results'))
    p.add_argument('--allow-mismatch', action='store_true',
                   help='Permit mismatched prompt/image settings; results marked exploratory')
    p.add_argument('--allow-partial', action='store_true',
                   help='Evaluate only images common to all four channels')
    p.add_argument('--weight-binary', type=float, default=1.0)
    p.add_argument('--weight-top3', type=float, default=1.0)
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Fraction of total weighted votes needed (inclusive); predefined, not fitted')
    return p.parse_args()


def provider_from_row(row, path):
    provider = (row.get('provider') or row.get('model') or '').lower()
    if provider in ('openai', 'chatgpt', 'gpt'):
        return 'gpt'
    if provider == 'gemini':
        return 'gemini'
    name = path.name.lower()
    if 'gemini' in name:
        return 'gemini'
    if any(t in name for t in ('openai', 'chatgpt', 'gpt')):
        return 'gpt'
    raise ValueError(f'Cannot identify provider in {path}; use provider column')


def mode_from_row(row, path):
    mode = (row.get('evaluation_mode') or '').lower()
    if mode in ('binary', 'top3'):
        return mode
    if 'top3' in path.name.lower():
        return 'top3'
    if 'binary' in path.name.lower():
        return 'binary'
    raise ValueError(f'Cannot identify mode for {path}')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    return rows


def discover(args):
    chosen = {}
    for channel in CHANNELS:
        explicit = getattr(args, channel)
        if explicit:
            chosen[channel] = explicit.resolve()
            continue
        matches = []
        for path in sorted(args.results_dir.glob('*.csv')):
            try:
                rows = read_csv(path)
                ids = {(provider_from_row(r, path), mode_from_row(r, path)) for r in rows}
                if ids == {tuple(channel.split('_', 1))}:
                    matches.append(path.resolve())
            except (ValueError, OSError, KeyError):
                continue
        if len(matches) != 1:
            raise ValueError(f'{channel}: found {len(matches)} matching files in {args.results_dir}. '
                             f'Provide --{channel.replace("_", "-")} PATH. Candidates: {matches}')
        chosen[channel] = matches[0]
    if len(set(chosen.values())) != 4:
        raise ValueError('Each channel must have a distinct CSV')
    return chosen


def normalize_path(value):
    value = str(value or '').strip().replace('\\', '/')
    if not value:
        raise ValueError('Missing image path')
    return re.sub(r'^\./', '', value)


def parse_guesses(value):
    if isinstance(value, list):
        return value
    if not value or not value.strip():
        return []
    try:
        obj = json.loads(value)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    raise ValueError(f'Invalid top3_guesses JSON: {value[:100]}')


def is_measles_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Empty or non-string diagnosis')
    s = name.casefold().strip()
    if re.search(r'\b(?:rubella|german measles)\b', s):
        if re.search(r'\b(?:rubeola|measles)\b', s) and s not in ('rubella', 'german measles'):
            raise ValueError(f'Ambiguous measles/rubella entry: {name}')
        return False
    if re.search(r'\b(?:rule out|r/o|versus|vs\.?|not measles|non.measles)\b', s):
        raise ValueError(f'Ambiguous diagnosis: {name}')
    return bool(re.search(r'\b(?:measles|rubeola)\b', s))


def load_channel(channel, path):
    data = {}
    rows = read_csv(path)
    for row in rows:
        actual = f'{provider_from_row(row, path)}_{mode_from_row(row, path)}'
        if actual != channel:
            raise ValueError(f'{path}: mixed/wrong channel {actual} (expected {channel})')
        key = normalize_path(row.get('path'))
        if key in data:
            raise ValueError(f'Duplicate image {key} in {path}; do not silently select a run')
        truth = row.get('truth', '').strip()
        if truth not in VALID:
            raise ValueError(f'{path}: invalid truth for {key}: {truth}')
        label = (row.get('predicted') or '').strip()
        if label not in VALID or (row.get('error') or '').strip():
            raise ValueError(f'{path}: invalid/failed prediction for {key}; fix input first')
        guesses = []
        rank = 0
        if channel.endswith('top3'):
            guesses = parse_guesses(row.get('top3_guesses', ''))
            if len(guesses) != 3 or len(set(str(x).casefold() for x in guesses)) != 3:
                raise ValueError(f'{path}: expected three distinct diagnoses for {key}')
            rank = next((i for i, x in enumerate(guesses, 1) if is_measles_name(x)), 0)
            if label != ('measles' if rank else 'non_measles'):
                raise ValueError(f'{path}: stored label conflicts with top3 for {key}')
        data[key] = {'truth': truth, 'label': label, 'rank': rank,
                     'guesses': guesses, 'sha': (row.get('image_sha256') or '').strip(),
                     'model': row.get('model_id', ''), 'run_id': row.get('run_id', '')}
    return data


def snapshot(path):
    name = path.name
    if not name.endswith('.csv'):
        return None
    candidate = path.with_name(name[:-4] + '.config.json')
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding='utf-8'))


def check_compatibility(inputs, datasets, allow_mismatch):
    issues = []
    keys = [set(d) for d in datasets.values()]
    if any(s != keys[0] for s in keys[1:]):
        issues.append('Image sets differ across channels')
    common = set.intersection(*keys)
    for key in common:
        truths = {d[key]['truth'] for d in datasets.values()}
        if len(truths) != 1:
            raise ValueError(f'Ground-truth conflict for {key}')
        shas = {d[key]['sha'] for d in datasets.values() if d[key]['sha']}
        if len(shas) > 1:
            issues.append('Image hashes differ across channels (could be different preprocessing)')
            break
    snapshots = {ch: snapshot(path) for ch, path in inputs.items()}
    if any(s is None for s in snapshots.values()):
        issues.append('Missing one or more configuration snapshots; prompt/preprocessing compatibility unverified')
    else:
        for field in ('dataset_dir', 'max_edge', 'jpeg_quality', 'sampling', 'seed', 'limit_per_class'):
            vals = {json.dumps(s.get(field), sort_keys=True) for s in snapshots.values()}
            if len(vals) > 1:
                issues.append(f'Configuration differs: {field}')
        for mode in ('binary', 'top3'):
            prompts = {json.dumps(s.get('prompts', {}).get(mode), sort_keys=True)
                       for s in snapshots.values() if s.get('mode') == mode}
            if len(prompts) > 1:
                issues.append(f'{mode} prompts differ across providers')
    issues = list(dict.fromkeys(issues))
    critical = [x for x in issues if x != 'Image sets differ across channels']
    if critical and not allow_mismatch:
        raise ValueError('Compatibility checks failed:\n - ' + '\n - '.join(critical) +
                         '\nUse --allow-mismatch ONLY for exploratory analysis.')
    return issues, common


def stats(rows, col):
    tp = sum(r['truth'] == 'measles' and r[col] == 1 for r in rows)
    fn = sum(r['truth'] == 'measles' and r[col] == 0 for r in rows)
    fp = sum(r['truth'] == 'non_measles' and r[col] == 1 for r in rows)
    tn = sum(r['truth'] == 'non_measles' and r[col] == 0 for r in rows)
    ratio = lambda x, y: x / y if y else None
    se, sp = ratio(tp, tp+fn), ratio(tn, tn+fp)
    return {'strategy': col, 'n': len(rows), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sensitivity': se, 'specificity': sp, 'accuracy': ratio(tp+tn, len(rows)),
            'balanced_accuracy': (se+sp)/2 if se is not None and sp is not None else None,
            'precision': ratio(tp, tp+fp) if tp+fp else 0.0,
            'f1': ratio(2*tp, 2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    if args.weight_binary < 0 or args.weight_top3 < 0 or args.weight_binary + args.weight_top3 <= 0:
        raise ValueError('Weights must be nonnegative and not both zero')
    if not 0 < args.threshold <= 1:
        raise ValueError('--threshold must be in (0, 1]')
    inputs = discover(args)
    data = {ch: load_channel(ch, path) for ch, path in inputs.items()}
    issues, common = check_compatibility(inputs, data, args.allow_mismatch)
    if not common:
        raise ValueError('No common images')
    if any(set(d) != common for d in data.values()) and not args.allow_partial:
        raise ValueError('Input image sets differ; use --allow-partial for intersection-only evaluation')
    if issues:
        print('COMPATIBILITY WARNINGS:', *issues, sep='\n - ')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(common):
        item = {'path': path, 'truth': data['gpt_binary'][path]['truth']}
        for ch in CHANNELS:
            d = data[ch][path]
            item[ch] = int(d['label'] == 'measles')
            if ch.endswith('top3'):
                item[ch + '_rank'] = d['rank']
                item[ch + '_guesses'] = json.dumps(d['guesses'], ensure_ascii=False)
        vals = [item[ch] for ch in CHANNELS]
        item['binary_or'] = int(item['gpt_binary'] or item['gemini_binary'])
        item['binary_and'] = int(item['gpt_binary'] and item['gemini_binary'])
        item['top3_or'] = int(item['gpt_top3'] or item['gemini_top3'])
        item['top3_and'] = int(item['gpt_top3'] and item['gemini_top3'])
        item['hybrid_or'] = int(any(vals))
        item['hybrid_and'] = int(all(vals))
        item['vote_at_least_2'] = int(sum(vals) >= 2)
        item['vote_at_least_3'] = int(sum(vals) >= 3)
        item['vote_all_4'] = int(sum(vals) == 4)
        item['gpt_within_model_or'] = int(item['gpt_binary'] or item['gpt_top3'])
        item['gemini_within_model_or'] = int(item['gemini_binary'] or item['gemini_top3'])
        weighted = args.weight_binary*(item['gpt_binary']+item['gemini_binary']) + args.weight_top3*(item['gpt_top3']+item['gemini_top3'])
        total = 2*(args.weight_binary+args.weight_top3)
        item['weighted_score'] = round(weighted/total, 8)
        item['weighted_vote'] = int(weighted/total >= args.threshold - 1e-12)
        rows.append(item)
    strategies = list(CHANNELS) + ['binary_or', 'binary_and', 'top3_or', 'top3_and',
                  'hybrid_or', 'hybrid_and', 'vote_at_least_2', 'vote_at_least_3', 'vote_all_4',
                  'gpt_within_model_or', 'gemini_within_model_or', 'weighted_vote']
    scores = [stats(rows, col) for col in strategies]
    # Descriptive agreement and complementarity, not model selection.
    comparisons = []
    for a, b in combinations(CHANNELS, 2):
        agree = sum(r[a] == r[b] for r in rows)
        both_tp = sum(r['truth'] == 'measles' and r[a] and r[b] for r in rows)
        either_tp = sum(r['truth'] == 'measles' and (r[a] or r[b]) for r in rows)
        unique_a = sum(r['truth'] == 'measles' and r[a] and not r[b] for r in rows)
        unique_b = sum(r['truth'] == 'measles' and r[b] and not r[a] for r in rows)
        pa = sum(r[a] for r in rows)/len(rows)
        pb = sum(r[b] for r in rows)/len(rows)
        pe = pa*pb + (1-pa)*(1-pb)
        comparisons.append({'channel_a': a, 'channel_b': b, 'agreement': agree/len(rows),
                            'cohen_kappa': (agree/len(rows)-pe)/(1-pe) if pe < 1 else None,
                            'both_true_positives': both_tp, 'either_true_positives': either_tp,
                            'unique_true_positives_a': unique_a, 'unique_true_positives_b': unique_b})
    write_csv(out/'aligned_predictions.csv', rows)
    write_csv(out/'fusion_metrics.csv', scores)
    write_csv(out/'pairwise_complementarity.csv', comparisons)
    manifest = {'inputs': {ch: str(path) for ch, path in inputs.items()},
                'models': {ch: sorted({d['model'] for d in data[ch].values()}) for ch in CHANNELS},
                'n_common': len(rows), 'n_by_channel': {ch: len(d) for ch, d in data.items()},
                'compatibility_warnings': issues, 'exploratory': bool(issues),
                'weights': {'binary': args.weight_binary, 'top3': args.weight_top3},
                'threshold': args.threshold,
                'note': 'Fusion rules are predefined and label-blind. Comparing many rules on the same test set is exploratory; do not claim the highest score is unbiased generalization. No ROC AUC from subjective confidence.'}
    (out/'fusion_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nAligned {len(rows)} images; outputs: {out}')
    print(f'{"Strategy":27} {"Sens":>7} {"Spec":>7} {"Acc":>7} {"F1":>7}  TP FP TN FN')
    for s in scores:
        fmt = lambda x: f'{x:.3f}' if x is not None else 'NA'
        print(f'{s["strategy"]:27} {fmt(s["sensitivity"]):>7} {fmt(s["specificity"]):>7} '
              f'{fmt(s["accuracy"]):>7} {fmt(s["f1"]):>7}  '
              f'{s["tp"]:2} {s["fp"]:2} {s["tn"]:2} {s["fn"]:2}')
    print('\nNo model inference performed. For generalization, choose a rule on validation cases and test it on held-out cases.')


if __name__ == '__main__':
    main()
