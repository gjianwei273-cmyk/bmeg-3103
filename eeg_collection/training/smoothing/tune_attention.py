"""Causal attention calibration with whole-session held-out evaluation (stdlib only)."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ...data_collecting import DATA_DIR


@dataclass(frozen=True)
class Config:
    method: str
    parameter: float
    threshold: float

    def __post_init__(self):
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 100:
            raise ValueError('threshold must be in (0, 100]')
        if self.method == 'raw' and self.parameter == 0:
            return
        if self.method == 'ma' and math.isfinite(self.parameter) and self.parameter >= 2 and int(self.parameter) == self.parameter:
            return
        if self.method == 'ema' and math.isfinite(self.parameter) and 0 < self.parameter <= 1:
            return
        raise ValueError('invalid smoothing method/parameter')


@dataclass(frozen=True)
class Protocol:
    delay_weight: float = .005
    flip_weight: float = .002
    horizon_s: float = 15
    confirm_samples: int = 3

    def __post_init__(self):
        if (not all(math.isfinite(x) and x >= 0 for x in (self.delay_weight, self.flip_weight))
                or not math.isfinite(self.horizon_s) or self.horizon_s <= 0
                or self.confirm_samples < 1 or int(self.confirm_samples) != self.confirm_samples):
            raise ValueError('invalid scoring weights, horizon or confirmation count')


def smooth(values, config):
    """Start once per session; no phase/label is available to this filter."""
    output, window = [], deque()
    total = 0.0
    for value in values:
        if config.method == 'raw':
            filtered = value
        elif config.method == 'ma':
            window.append(value)
            total += value
            if len(window) > config.parameter:
                total -= window.popleft()
            filtered = total / len(window)
        else:
            filtered = value if not output else config.parameter * value + (1 - config.parameter) * output[-1]
        output.append(filtered)
    return output


def load_sessions(directory):
    sessions, excluded, seen = [], [], set()
    for path in sorted(Path(directory).glob('*.csv')):
        try:
            meta_path = path.with_suffix('.json')
            meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
            if meta['source'] != 'serial' or meta['status'] != 'completed':
                raise ValueError('requires completed serial session')
            session_id = str(meta['session_id'])
            if not session_id or session_id in seen:
                raise ValueError('empty or duplicate session_id')
            rows, skipped = [], 0
            previous_time, previous_phase_time, previous_label = -1, -1, None
            phase_origin = None
            with path.open(encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    if row['session_id'] != session_id or row['subject_id'] != meta['subject_id'] or row['source'] != 'serial':
                        raise ValueError('row provenance differs from metadata')
                    label = int(row['label'])
                    if row['phase'] not in ('relax', 'focus') or label != int(row['phase'] == 'focus'):
                        raise ValueError('phase/label mismatch')
                    time_value, phase_time = float(row['elapsed_s']), float(row['phase_elapsed_s'])
                    if (not all(math.isfinite(x) and x >= 0 for x in (time_value, phase_time))
                            or time_value < previous_time or phase_time > time_value
                            or (label == previous_label and phase_time < previous_phase_time)):
                        raise ValueError('nonfinite or nonmonotonic receive times')
                    origin = time_value - phase_time
                    if label != previous_label:
                        phase_origin = origin
                    elif abs(origin - phase_origin) > .002:
                        raise ValueError('elapsed_s and phase_elapsed_s have inconsistent phase origins')
                    previous_time, previous_phase_time, previous_label = time_value, phase_time, label
                    if row['valid'] != '1' or row['training_eligible'] != '1':
                        skipped += 1
                        continue
                    attention = float(row['attention'])
                    if not math.isfinite(attention) or not 0 < attention <= 100:
                        raise ValueError('eligible attention must be finite and in (0,100]')
                    rows.append(dict(time=time_value, phase_time=phase_time,
                                     label=label, attention=attention))
            if {r['label'] for r in rows} != {0, 1}:
                raise ValueError('both eligible classes are required')
            # Each file from this collector must be one relax block then one focus block.
            if sum(a['label'] != b['label'] for a, b in zip(rows, rows[1:])) != 1 or rows[0]['label'] != 0:
                raise ValueError('expected one relax block followed by one focus block')
            digest = hashlib.sha256(path.read_bytes() + meta_path.read_bytes()).hexdigest()
            sessions.append(dict(id=session_id, subject=meta['subject_id'], rows=rows,
                                 skipped_rows=skipped, csv_file=path.name, sha256=digest))
            seen.add(session_id)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            excluded.append(dict(file=path.name, reason=str(exc)))
    return sessions, excluded


def evaluate(session, config, protocol, filtered=None):
    rows = session['rows']
    if filtered is None:
        filtered = smooth([r['attention'] for r in rows], config)
    predictions = [int(value >= config.threshold) for value in filtered]
    tp = sum(p == 1 and r['label'] == 1 for p, r in zip(predictions, rows))
    tn = sum(p == 0 and r['label'] == 0 for p, r in zip(predictions, rows))
    fp = sum(p == 1 and r['label'] == 0 for p, r in zip(predictions, rows))
    fn = sum(p == 0 and r['label'] == 1 for p, r in zip(predictions, rows))
    focus_recall, relax_recall = tp / (tp + fn), tn / (tn + fp)
    flips = 0
    duration = 0.0
    for i in range(1, len(rows)):
        if rows[i]['label'] == rows[i - 1]['label']:
            flips += predictions[i] != predictions[i - 1]
            duration += rows[i]['time'] - rows[i - 1]['time']
    delays, penalties, preexisting, missed = [], [], 0, 0
    for i in range(1, len(rows)):
        if rows[i]['label'] == rows[i - 1]['label']:
            continue
        target = rows[i]['label']
        if predictions[i - 1] == target:
            preexisting += 1
            penalties.append(protocol.horizon_s)
            continue
        streak, delay = 0, None
        for j in range(i, len(rows)):
            if rows[j]['label'] != target or rows[j]['phase_time'] > protocol.horizon_s:
                break
            streak = streak + 1 if predictions[j] == target else 0
            if streak >= protocol.confirm_samples:
                delay = rows[j]['phase_time']
                break
        if delay is None:
            missed += 1
            penalties.append(protocol.horizon_s)
        else:
            delays.append(delay)
            penalties.append(delay)
    result = dict(n=len(rows), tp=tp, tn=tn, fp=fp, fn=fn,
                  accuracy=(tp + tn) / len(rows), focus_recall=focus_recall,
                  relax_recall=relax_recall, balanced_accuracy=(focus_recall + relax_recall) / 2,
                  flips=flips, flips_per_min=flips * 60 / duration if duration > 0 else 0,
                  duration_s=duration, confirmed_delay_s=statistics.mean(delays) if delays else None,
                  confirmed=len(delays), preexisting=preexisting, missed=missed,
                  transitions=len(penalties), delay_penalty_s=statistics.mean(penalties) if penalties else 0)
    result['score'] = (result['balanced_accuracy'] - protocol.delay_weight * result['delay_penalty_s']
                       - protocol.flip_weight * result['flips_per_min'])
    return result


def aggregate(metrics):
    """Session-macro means for quality metrics; sums for counts/exposure."""
    summed = {'n', 'tp', 'tn', 'fp', 'fn', 'flips', 'duration_s',
              'confirmed', 'preexisting', 'missed', 'transitions'}
    result = {}
    for key in metrics[0]:
        values = [m[key] for m in metrics if m[key] is not None]
        result[key] = sum(values) if key in summed else (statistics.mean(values) if values else None)
    return result


def select(indices, grid, cache):
    evaluated = [(c, aggregate([cache[c][i] for i in indices])) for c in grid]
    return max(evaluated, key=lambda pair: (pair[1]['score'], pair[1]['balanced_accuracy'],
               -pair[1]['delay_penalty_s'], -pair[1]['flips_per_min'],
               -grid.index(pair[0])))


def analyze(sessions, grid, protocol):
    if len(sessions) < 3:
        raise ValueError('at least 3 eligible sessions are required for whole-session validation')
    grid = list(dict.fromkeys(grid))
    if not grid:
        raise ValueError('empty parameter grid')
    baseline = Config('raw', 0, 50)
    cache, filters = {}, {}
    for config in list(dict.fromkeys([*grid, baseline])):
        cache[config] = []
        for i, session in enumerate(sessions):
            key = (i, config.method, config.parameter)
            if key not in filters:
                filters[key] = smooth([r['attention'] for r in session['rows']], config)
            cache[config].append(evaluate(session, config, protocol, filters[key]))
    folds = []
    strategies = {'selected': grid}
    for method in ('raw', 'ma', 'ema'):
        family = [c for c in grid if c.method == method]
        if family:
            strategies[method] = family
    for heldout, session in enumerate(sessions):
        train = [i for i in range(len(sessions)) if i != heldout]
        for strategy, candidates in {**strategies, 'baseline': [baseline]}.items():
            config, train_metrics = select(train, candidates, cache)
            folds.append(dict(session_id=session['id'], strategy=strategy,
                              train_session_ids=[sessions[i]['id'] for i in train],
                              config=asdict(config), train_score=train_metrics['score'],
                              metrics=cache[config][heldout]))
    all_indices = list(range(len(sessions)))
    recommendation, fit = select(all_indices, grid, cache)
    candidates = [dict(config=asdict(c), metrics=aggregate(cache[c])) for c in grid]
    candidates.sort(key=lambda c: (-c['metrics']['score'], -c['metrics']['balanced_accuracy']))
    validation = {strategy: aggregate([f['metrics'] for f in folds if f['strategy'] == strategy])
                  for strategy in [*strategies, 'baseline']}
    assessment = dict(score_delta_vs_baseline=validation['selected']['score'] - validation['baseline']['score'],
                      balanced_accuracy_delta_vs_baseline=validation['selected']['balanced_accuracy'] - validation['baseline']['balanced_accuracy'],
                      supports_replacing_baseline=validation['selected']['score'] > validation['baseline']['score'])
    return dict(protocol=asdict(protocol), candidates=candidates, folds=folds,
                validation=validation, validation_assessment=assessment,
                recommendation=dict(config=asdict(recommendation), fit_metrics=fit,
                                    status='calibration_only_requires_new_fixed_parameter_session'))


def write_outputs(result, sessions, excluded, output_dir):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    folder = Path(output_dir) / (stamp + '_' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    result.update(excluded=excluded, sessions=[{k: v for k, v in s.items() if k != 'rows'} for s in sessions],
                  created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
                  schema_version=1, grid_size=len(result['candidates']),
                  limitations=['Same-subject/same-day validation is not cross-subject generalization.',
                               'Instruction labels are not verified mental states.',
                               'Preparation and missing samples are unavailable; filters retain state across gaps.',
                               'Delay measures confirmation relative to instruction; preexisting targets are penalized.',
                               'Full-data candidate scores are exploratory, not held-out estimates.'])
    (folder / 'recommendation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for filename, records in [('candidates.csv', result['candidates']), ('folds.csv', result['folds'])]:
        flat = []
        for record in records:
            row = {**record['config'], **record['metrics']}
            if 'strategy' in record:
                row.update(session_id=record['session_id'], strategy=record['strategy'],
                           train_session_ids=';'.join(record['train_session_ids']), train_score=record['train_score'])
            flat.append(row)
        with (folder / filename).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    payload = {**result, 'traces': sessions}
    # Escape '<' so user-controlled metadata cannot terminate the JSON script element.
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = Path(__file__).with_name('attention_report.html').read_text(encoding='utf-8')
    (folder / 'report.html').write_text(template.replace('__REPORT_DATA__', embedded), encoding='utf-8')
    return folder


def number_list(value, kind=float):
    try:
        values = [kind(x.strip()) for x in value.split(',')]
        if not values or any(not math.isfinite(x) for x in values):
            raise ValueError()
        return list(dict.fromkeys(values))
    except ValueError:
        raise ValueError('grids must be nonempty comma-separated finite numbers') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Search causal attention filters; validate by held-out session; write offline HTML charts.')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR / 'half_half')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).with_name('reports'))
    parser.add_argument('--windows', default=','.join(map(str, range(2, 11))), help='trailing sample counts, comma-separated')
    parser.add_argument('--alphas', default=','.join(str(i / 10) for i in range(1, 10)))
    parser.add_argument('--thresholds', default=','.join(map(str, range(1, 100))))
    parser.add_argument('--delay-weight', type=float, default=.005, help='score penalty per second of delay')
    parser.add_argument('--flip-weight', type=float, default=.002, help='score penalty per flip/minute')
    parser.add_argument('--horizon-seconds', type=float, default=15)
    parser.add_argument('--confirm-samples', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        protocol = Protocol(args.delay_weight, args.flip_weight, args.horizon_seconds, args.confirm_samples)
        thresholds = number_list(args.thresholds)
        filters = [('raw', 0)] + [('ma', w) for w in number_list(args.windows, int)] + [('ema', a) for a in number_list(args.alphas)]
        grid = [Config(method, parameter, threshold) for method, parameter in filters for threshold in thresholds]
        if not args.data_dir.is_dir():
            raise ValueError('data directory does not exist')
        sessions, excluded = load_sessions(args.data_dir)
        result = analyze(sessions, grid, protocol)
        folder = write_outputs(result, sessions, excluded, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print('Report: ' + str((folder / 'report.html').resolve()))
    print('Recommendation: ' + json.dumps(result['recommendation']['config']))
    for name in ('selected', 'baseline'):
        m = result['validation'][name]
        print(f"Held-out {name}: balanced accuracy={m['balanced_accuracy']:.2%}, delay penalty={m['delay_penalty_s']:.2f}s, flips/min={m['flips_per_min']:.2f}")
    print(f"Sessions={len(sessions)}, samples={sum(len(s['rows']) for s in sessions)}, excluded files={len(excluded)}, candidates={len(grid)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
