#!/usr/bin/env python3
"""Shared dataset, prompt, parsing, checkpointing and scoring for rash-image benchmarks."""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import time
from pathlib import Path
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
FIELDS = ['path', 'truth', 'provider', 'model_id', 'evaluation_mode', 'run_id', 'image_sha256',
          'predicted', 'top3_guesses', 'confidence_subjective', 'seconds', 'error', 'raw_response']


def arguments(provider):
    ap = argparse.ArgumentParser(description=f'Assess measles rash photographs using {provider}.')
    ap.add_argument('--config', type=Path, default=Path(__file__).with_name('measles_config.json'))
    ap.add_argument('--data', type=Path, help='Override dataset_dir in JSON')
    ap.add_argument('--model', help=f'Override models.{provider} in JSON')
    ap.add_argument('--mode', choices=['binary', 'top3'], help='Override mode in JSON')
    ap.add_argument('--limit-per-class', type=int, help='Override limit_per_class; 0 = all')
    ap.add_argument('--overwrite', action='store_true', help='Recompute every selected image in this run')
    return ap.parse_args()


def config_load(args, provider):
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    for key in ('dataset_dir', 'output_dir', 'models', 'prompts'):
        if key not in cfg:
            raise ValueError(f'Missing configuration key: {key}')
    cfg['dataset_dir'] = str(args.data.resolve() if args.data else (cfg_path.parent / cfg['dataset_dir']).resolve())
    cfg['output_dir'] = str((cfg_path.parent / cfg['output_dir']).resolve())
    cfg['mode'] = args.mode or cfg.get('mode', 'binary')
    cfg['limit_per_class'] = args.limit_per_class if args.limit_per_class is not None else cfg.get('limit_per_class', 0)
    cfg['model_id'] = args.model or cfg['models'][provider]
    if cfg['mode'] not in ('binary', 'top3') or cfg['limit_per_class'] < 0:
        raise ValueError('Invalid mode or limit_per_class')
    if cfg.get('sampling', 'first') not in ('first', 'random'):
        raise ValueError('sampling must be first or random')
    return cfg


def images_for_run(cfg):
    root = Path(cfg['dataset_dir'])
    result = []
    rng = random.Random(int(cfg.get('seed', 42)))
    for label in ('measles', 'non_measles'):
        folder = root / label
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing class folder: {folder}')
        paths = sorted(p for p in folder.rglob('*') if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in EXTS)
        if cfg.get('sampling', 'first') == 'random':
            rng.shuffle(paths)
        if cfg['limit_per_class']:
            paths = paths[:cfg['limit_per_class']]
        result.extend((p, p.relative_to(root).as_posix(), label) for p in paths)
        print(f'{label}: {len(paths)} selected images')
    if not result:
        raise ValueError('No images found')
    return result


def prepared_image(path, cfg):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            im = im.convert('RGBA')
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert('RGB')
        edge = int(cfg.get('max_edge', 1024))
        if edge > 0:
            im.thumbnail((edge, edge))
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=int(cfg.get('jpeg_quality', 90)))
    return buf.getvalue()


def parse_response(raw, mode):
    # Robust JSON extraction: handle markdown fences (```json ... ```) and leading/trailing text.
    raw_str = (raw or '').strip()
    if raw_str.startswith('```'):
        raw_str = re.sub(r'^```(?:json)?\s*', '', raw_str, flags=re.IGNORECASE)
        raw_str = re.sub(r'\s*```$', '', raw_str)
        raw_str = raw_str.strip()
    try:
        obj = json.loads(raw_str)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw_str, re.DOTALL)
        if not m:
            raise ValueError(f'Response is not valid JSON: {raw_str[:200]!r}')
        obj = json.loads(m.group(0))

    if not isinstance(obj, dict):
        raise ValueError('Response must be a JSON object')

    if mode == 'binary':
        label = str(obj.get('label', '')).strip().lower().replace('-', '_')
        if label not in ('measles', 'non_measles'):
            raise ValueError(f'Invalid binary label: {obj.get("label")!r}')
        conf = obj.get('confidence')
        if conf is not None:
            if isinstance(conf, bool):
                raise ValueError(f'Invalid confidence (boolean): {conf!r}')
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                raise ValueError(f'Invalid confidence (non-numeric): {conf!r}')
            if not 0.0 <= conf <= 1.0:
                raise ValueError(f'Invalid confidence (out of [0, 1] range): {conf!r}')
        return label, [], conf

    guesses = obj.get('top3')
    if not isinstance(guesses, list) or len(guesses) != 3:
        raise ValueError('top3 must contain exactly three diagnoses')
    clean = []
    for guess in guesses:
        if not isinstance(guess, str) or not guess.strip():
            raise ValueError('Each diagnosis must be a nonempty string')
        g = guess.strip()
        # Reject ambiguous differential phrasing (e.g. "A versus B", "rule out C", "X / Y")
        if re.search(r'\b(?:versus|vs\.?|rule out|r/o|possible|not measles|non[ -_]measles)\b', g, re.I):
            raise ValueError(f'Ambiguous diagnosis: {g!r}')
        # Reject slash alternatives (e.g. "measles / scarlet fever"), but allow standard clinical acronyms like SJS/TEN
        if '/' in g and not re.search(r'\bSJS/TEN\b', g, re.I):
            raise ValueError(f'Ambiguous diagnosis (slash alternative): {g!r}')
        clean.append(g)

    if len({g.casefold() for g in clean}) != 3:
        raise ValueError('Top-3 diagnoses must be distinct')

    def is_measles_name(g):
        s = g.casefold()
        # Remove rubella / german measles references to avoid 'measles' substring inside 'german measles'
        s_without_rubella = re.sub(r'\b(?:german\s+measles|rubella)\b', '', s)
        has_measles = bool(re.search(r'\b(?:measles|rubeola)\b', s_without_rubella))
        has_rubella = bool(re.search(r'\b(?:german\s+measles|rubella)\b', s))
        if has_measles and has_rubella:
            raise ValueError(f'Ambiguous measles/rubella diagnosis: {g!r}')
        return has_measles

    label = 'measles' if any(is_measles_name(g) for g in clean) else 'non_measles'
    return label, clean, None


def config_fingerprint(cfg, provider):
    # Include all settings that affect the request or image selection, not the output location.
    relevant = {'provider': provider, 'model': cfg['model_id'], 'mode': cfg['mode'],
                'prompt': cfg['prompts'][cfg['mode']], 'max_edge': cfg.get('max_edge', 1024),
                'jpeg_quality': cfg.get('jpeg_quality', 90), 'provider_options': cfg.get(provider, {}),
                'sampling': cfg.get('sampling', 'first'), 'seed': cfg.get('seed', 42),
                'limit_per_class': cfg['limit_per_class']}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def read_rows(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_rows(csv_path, rows):
    temp = csv_path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    temp.replace(csv_path)


def metrics(rows, selected, cfg, provider, run_id):
    valid = [r for r in rows if r['predicted'] in ('measles', 'non_measles')]
    tp = sum(r['truth'] == 'measles' and r['predicted'] == 'measles' for r in valid)
    tn = sum(r['truth'] == 'non_measles' and r['predicted'] == 'non_measles' for r in valid)
    fp = sum(r['truth'] == 'non_measles' and r['predicted'] == 'measles' for r in valid)
    fn = sum(r['truth'] == 'measles' and r['predicted'] == 'non_measles' for r in valid)
    ratio = lambda a, b: a / b if b else None
    sens, spec = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {'provider': provider, 'model': cfg['model_id'], 'evaluation_mode': cfg['mode'],
            'run_id': run_id, 'n_selected': len(selected), 'n': len(valid),
            'n_positive': tp + fn, 'n_negative': tn + fp,
            'accuracy': ratio(tp + tn, len(valid)),
            'balanced_accuracy': (sens + spec) / 2 if sens is not None and spec is not None else None,
            'sensitivity': sens, 'specificity': spec,
            'precision': ratio(tp, tp + fp) if tp + fp else 0.0,
            'f1': ratio(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'errors': sum(r['predicted'] == 'ERROR' for r in rows),
            'complete': len(valid) == len(selected),
            'note': 'Image-level estimates; subjective confidence is not a calibrated measles score. No ROC AUC.'}


def run(provider, api_call):
    args = arguments(provider)
    cfg = config_load(args, provider)
    selected = images_for_run(cfg)
    run_id = config_fingerprint(cfg, provider)
    folder = Path(cfg['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r'[^A-Za-z0-9._-]+', '_', cfg['model_id'])
    stem = f'measles_{provider}_{safe_model}_{cfg["mode"]}_{run_id}'
    csv_path = folder / (stem + '.csv')
    json_path = folder / (stem + '.metrics.json')
    # Snapshot full effective config, including prompt, for reproducibility.
    (folder / (stem + '.config.json')).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    rows = [] if args.overwrite else read_rows(csv_path)
    # One run per output file; no cross-run contamination.
    rows = [r for r in rows if r.get('run_id') == run_id and r.get('provider') == provider]
    done = {(r['path'], r['image_sha256']) for r in rows if r['predicted'] in ('measles', 'non_measles')}
    for i, (path, rel, truth) in enumerate(selected, 1):
        image = prepared_image(path, cfg)
        digest = hashlib.sha256(image).hexdigest()
        if (rel, digest) in done:
            print(f'[{i}/{len(selected)}] skip {rel}')
            continue
        print(f'[{i}/{len(selected)}] {provider} {cfg["model_id"]} {cfg["mode"]}: {rel}', flush=True)
        predicted, guesses, confidence, raw, error, elapsed = 'ERROR', [], None, '', '', 0.0
        for attempt in range(int(cfg.get('retries', 2)) + 1):
            t0 = time.monotonic()
            try:
                raw = api_call(cfg, image)
                predicted, guesses, confidence = parse_response(raw, cfg['mode'])
                error = ''
                elapsed = time.monotonic() - t0
                break
            except Exception as exc:
                elapsed = time.monotonic() - t0
                error = f'{type(exc).__name__}: {exc}'
                if attempt < int(cfg.get('retries', 2)):
                    time.sleep(min(3 * 2 ** attempt, 30))
        rows = [r for r in rows if r.get('path') != rel]
        rows.append({'path': rel, 'truth': truth, 'provider': provider, 'model_id': cfg['model_id'],
                     'evaluation_mode': cfg['mode'], 'run_id': run_id, 'image_sha256': digest,
                     'predicted': predicted, 'top3_guesses': json.dumps(guesses, ensure_ascii=False),
                     'confidence_subjective': confidence, 'seconds': round(elapsed, 2),
                     'error': error, 'raw_response': raw})
        write_rows(csv_path, rows)
        print('  ', error[:200] if error else (guesses if cfg['mode'] == 'top3' else predicted))
        if cfg.get('sleep_seconds', 0):
            time.sleep(float(cfg['sleep_seconds']))
    selected_paths = {rel for _, rel, _ in selected}
    # Exclude any stale rows from a prior selection, even if present in the same file.
    current = [r for r in rows if r['path'] in selected_paths]
    result = metrics(current, selected, cfg, provider, run_id)
    json_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Predictions:', csv_path)
    print('Metrics:', json_path)
    print(json.dumps(result, indent=2))
    if cfg['mode'] == 'top3':
        print('Top-3 positive only if measles/rubeola appears among three valid diagnoses.')
    else:
        print('Binary positive only if model labels image measles.')
    if not result['complete']:
        print('WARNING: Incomplete run. Check ERROR rows; metrics exclude failed requests.')
