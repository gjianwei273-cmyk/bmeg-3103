"""Compare focus classifiers, train a deployable model, and write reproducible results."""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime
from itertools import product
import json
from pathlib import Path
import statistics
from uuid import uuid4

import keras
import numpy as np

from .focus_pipeline import (DEFAULT_DATA, DEFAULT_SETTINGS, MODES, interval_metrics,
                            load_sessions, make_windows, model_inputs, replay_states,
                            validation_folds)


@dataclass(frozen=True)
class Candidate:
    mode: str
    window: int
    layers: int

    @property
    def name(self):
        return f'{self.mode}_w{self.window}_d{self.layers}'


def build_model(window, width, layers, filters, seed):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    return keras.Sequential([
        keras.Input(shape=(window, width)),
        *[keras.layers.Conv1D(filters, 3, padding='same', activation='relu')
          for _ in range(layers)],
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1, activation='sigmoid'),
    ])


def examples(sessions, candidate, gap, require_both=True):
    samples = [(frame.label, window) for session in sessions
               for _, frame, window in make_windows(session, candidate.mode, candidate.window, gap)
               if frame.label is not None]
    if not samples or (require_both and {label for label, _ in samples} != {0, 1}):
        ids = ', '.join(session.id for session in sessions)
        required = 'of both classes' if require_both else 'for validation'
        raise ValueError(f'{candidate.name} needs usable windows {required} in {ids}')
    return (model_inputs([window for _, window in samples], candidate.mode),
            np.asarray([label for label, _ in samples], dtype=np.float32).reshape(-1, 1))


def fit(candidate, train, valid, args):
    x, y = examples(train, candidate, args.max_gap)
    vx, vy = examples([valid], candidate, args.max_gap, require_both=False)
    mean = x.mean(axis=(0, 1))
    scale = np.maximum(x.std(axis=(0, 1)), 1e-6)
    x, vx = (x - mean) / scale, (vx - mean) / scale
    model = build_model(candidate.window, x.shape[-1], candidate.layers, args.filters, args.seed)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.003),
                  loss='binary_crossentropy', run_eagerly=True)
    rng = np.random.default_rng(args.seed)
    best_loss, best_epoch, weights = float('inf'), 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(len(x))
        for batch in np.array_split(indices, (len(x) + 63) // 64):
            model.train_on_batch(x[batch], y[batch])
        train_loss = float(keras.ops.mean(keras.losses.binary_crossentropy(y, model(x, training=False))))
        val_loss = float(keras.ops.mean(keras.losses.binary_crossentropy(vy, model(vx, training=False))))
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        if val_loss < best_loss:
            best_loss, best_epoch, weights = val_loss, epoch, model.get_weights()
        if epoch - best_epoch >= args.patience:
            break
    if weights is None:
        raise ValueError('model did not produce a finite validation loss')
    model.set_weights(weights)
    return model, mean, scale, history, best_epoch


def predict(session, candidate, gap, model, mean, scale):
    entries = list(make_windows(session, candidate.mode, candidate.window, gap))
    scores = [None] * len(session.frames)
    if entries:
        data = model_inputs([window for _, _, window in entries], candidate.mode)
        prediction = model((data - mean) / scale, training=False).numpy()[:, 0]
        for (index, _, _), score in zip(entries, prediction):
            scores[index] = float(score)
    return scores


def trend_reference(session, gap):
    """Trend EMA reference with explicit coefficients, independent of the PDF implementation."""
    level = trend = last_score = last_time = None
    scores = []
    for frame in session.frames:
        attention = frame.values[8] if frame.values is not None else None
        if attention is None:
            level = trend = last_score = last_time = None
            scores.append(None)
            continue
        value = attention / 100
        if last_time is None or frame.time - last_time >= gap:
            level, trend = value, 0.
        else:
            level += .5 * (value - level)
            trend += .25 * ((value - last_score) - trend)
        scores.append(min(1., max(0., level + trend)))
        last_score, last_time = value, frame.time
    return scores


def error_rate(metrics):
    seconds = metrics['evaluated_seconds']
    return (metrics['fp_seconds'] + metrics['fn_seconds']) / seconds if seconds else None


def calibration(session, scores, gap, optional=False):
    candidates = []
    for threshold, on_frames in product((.35, .4, .45, .5, .55, .6, .65, .7, .75), (1, 2, 3)):
        settings = dict(DEFAULT_SETTINGS, max_gap_s=gap, threshold=threshold, on_frames=on_frames)
        metrics = interval_metrics(session, scores, replay_states(session, scores, settings), gap)
        error = error_rate(metrics)
        if error is not None:
            candidates.append((error, abs(threshold - .5), on_frames, threshold, settings))
    if not candidates:
        if optional:
            return None
        raise ValueError(f'no scorable elapsed intervals in calibration session {session.id}')
    return min(candidates, key=lambda item: item[:-1])[-1]


def evaluate(session, scores, settings, allowed=None):
    states = replay_states(session, scores, settings)
    return interval_metrics(session, scores, states, settings['max_gap_s'], allowed)


def summarize(results):
    keys = ('tp', 'tn', 'fp', 'fn', 'fp_seconds', 'fn_seconds', 'evaluated_seconds',
            'flips', 'on_missed', 'off_missed', 'unobservable_transitions')
    totals = {key: sum(result[key] for result in results) for key in keys}
    tp, tn, fp, fn = (totals[key] for key in ('tp', 'tn', 'fp', 'fn'))
    count, denominator = tp + tn + fp + fn, 2 * tp + fp + fn
    delays = {key: [delay for result in results for delay in result[key]]
              for key in ('on_delays_s', 'off_delays_s')}
    return dict(totals, accuracy=(tp + tn) / count if count else None,
                f1=2 * tp / denominator if denominator else None,
                time_error_rate=error_rate(totals),
                on_delay_median_s=statistics.median(delays['on_delays_s']) if delays['on_delays_s'] else None,
                off_delay_median_s=statistics.median(delays['off_delays_s']) if delays['off_delays_s'] else None)


def common_mask(score_lists):
    return {i for i, scores in enumerate(zip(*score_lists)) if all(score is not None for score in scores)}


def compare(candidates, sessions, args):
    reference_name = 'trend_ema_reference'
    results = {candidate.name: [] for candidate in candidates}
    results[reference_name] = []
    for train, valid, test in validation_folds(sessions):
        print(f'CV held-out {test.id}; validation {valid.id}', flush=True)
        predictions, records = {}, {}
        for candidate in candidates:
            model, mean, scale, history, epoch = fit(candidate, train, valid, args)
            val_scores = predict(valid, candidate, args.max_gap, model, mean, scale)
            predictions[candidate.name] = predict(test, candidate, args.max_gap, model, mean, scale)
            records[candidate.name] = dict(best_epoch=epoch, history=history,
                                          settings=calibration(valid, val_scores, args.max_gap))
        reference_settings = calibration(valid, trend_reference(valid, args.max_gap),
                                         args.max_gap, optional=True)
        reference_scores = trend_reference(test, args.max_gap)
        if reference_settings and any(score is not None for score in reference_scores):
            predictions[reference_name] = reference_scores
            records[reference_name] = dict(settings=reference_settings)
        mask = common_mask(predictions.values())
        if not mask and reference_name in predictions:
            del predictions[reference_name]
            mask = common_mask(predictions.values())
        for name, scores in predictions.items():
            record = records[name]
            metrics = evaluate(test, scores, record['settings'], mask)
            results[name].append(dict(record, train=[session.id for session in train],
                                      validation=valid.id, test=test.id, metrics=metrics,
                                      available_frames=sum(score is not None for score in scores),
                                      total_frames=len(test.frames)))
    return {name: dict(summary=summarize([record['metrics'] for record in folds]), folds=folds)
            for name, folds in results.items()}


def best(candidates, results):
    def ranking(candidate):
        error = results[candidate.name]['summary']['time_error_rate']
        return float('inf') if error is None else error, candidate.layers, candidate.window, candidate.name
    winner = min(candidates, key=ranking)
    if results[winner.name]['summary']['time_error_rate'] is None:
        raise ValueError('comparison has no shared scorable intervals; check timestamp gaps and feature availability')
    return winner


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_curves(path, history, learning):
    with (path / 'training_curve.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(history)
    with (path / 'learning_curve.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['training_sessions', 'time_error_rate',
                                                    'accuracy', 'best_epoch'])
        writer.writeheader()
        writer.writerows(learning)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 3.5), layout='constrained')
    FigureCanvasAgg(figure)
    loss, volume = figure.subplots(ncols=2)
    epochs = [row['epoch'] for row in history]
    loss.plot(epochs, [row['train_loss'] for row in history], label='Training')
    loss.plot(epochs, [row['val_loss'] for row in history], label='Validation')
    loss.set(xlabel='Epoch', ylabel='Binary cross-entropy')
    loss.legend()
    points = [row for row in learning if row['time_error_rate'] is not None]
    volume.plot([row['training_sessions'] for row in points],
                [100 * row['time_error_rate'] for row in points], marker='o')
    volume.set(xlabel='Training recordings', ylabel='Validation timer error (%)',
               xticks=[row['training_sessions'] for row in points])
    figure.savefig(path / 'curves.png', dpi=140)


def save_comparison(path, comparisons):
    columns = ['stage', 'model', 'time_error_rate', 'evaluated_seconds', 'fp_seconds', 'fn_seconds',
               'accuracy', 'f1', 'on_delay_median_s', 'off_delay_median_s', 'flips',
               'on_missed', 'off_missed', 'unobservable_transitions']
    with (path / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for stage, results in comparisons.items():
            for name, result in results.items():
                writer.writerow(dict(stage=stage, model=name, **result['summary']))


def run(sessions, excluded, args, path):
    holdout = [session for session in sessions if session.day == args.holdout_day] if args.holdout_day else []
    development = [session for session in sessions if session not in holdout]
    if args.holdout_day and not holdout:
        raise ValueError(f'no recordings found for held-out day {args.holdout_day}')
    if len(development) < 3:
        raise ValueError('need at least three development recordings for train/validation/test')
    stage_inputs = [Candidate(mode, 8, 2) for mode in args.inputs]
    input_results = compare(stage_inputs, development, args)
    input_winner = best(stage_inputs, input_results)
    stage_windows = [Candidate(input_winner.mode, length, 2) for length in args.windows]
    window_results = compare(stage_windows, development, args)
    window_winner = best(stage_windows, window_results)
    stage_layers = [Candidate(window_winner.mode, window_winner.window, layers) for layers in args.layers]
    layer_results = compare(stage_layers, development, args)
    winner = best(stage_layers, layer_results)
    train, valid = development[:-1], development[-1]
    model, mean, scale, history, epoch = fit(winner, train, valid, args)
    val_scores = predict(valid, winner, args.max_gap, model, mean, scale)
    settings = dict(calibration(valid, val_scores, args.max_gap),
                    input_mode=winner.mode, window=winner.window)
    model.save(path / 'model.keras')
    save_json(path / 'model.json', dict(candidate=asdict(winner), settings=settings,
                                       mean=mean.tolist(), scale=scale.tolist()))
    learning = []
    sizes = sorted({1, len(train) // 2, len(train)} - {0})
    for size in sizes:
        try:
            if size == len(train):
                scores, point_epoch = val_scores, epoch
            else:
                point_model, point_mean, point_scale, _, point_epoch = fit(winner, train[:size], valid, args)
                scores = predict(valid, winner, args.max_gap, point_model, point_mean, point_scale)
        except ValueError as error:
            print(f'Learning curve skipped at {size} recordings: {error}', flush=True)
            continue
        point = summarize([evaluate(valid, scores, settings)])
        learning.append(dict(training_sessions=size, time_error_rate=point['time_error_rate'],
                             accuracy=point['accuracy'], best_epoch=point_epoch))
    save_curves(path, history, learning)
    reference = calibration(valid, trend_reference(valid, args.max_gap), args.max_gap, optional=True)
    final = []
    if holdout:
        # Reload the saved model because the learning-curve fits reset the Keras state.
        model = keras.saving.load_model(path / 'model.keras', compile=False)
        for test in holdout:
            scores = predict(test, winner, args.max_gap, model, mean, scale)
            baseline = trend_reference(test, args.max_gap)
            mask = common_mask([scores, baseline]) if reference else set()
            final.append(dict(test=test.id, model=evaluate(test, scores, settings, mask or None),
                              reference=evaluate(test, baseline, reference, mask) if mask else None,
                              model_all_available=evaluate(test, scores, settings)))
    comparisons = dict(inputs=input_results, windows=window_results, layers=layer_results)
    save_comparison(path, comparisons)
    save_json(path / 'report.json', dict(
        development=[dict(id=s.id, day=s.day, frames=len(s.frames)) for s in development],
        excluded=excluded, comparisons=comparisons,
        model=winner.name, training_sessions=[s.id for s in train], validation=valid.id,
        validation_best_epoch=epoch, settings=settings, holdout_day=args.holdout_day,
        heldout_results=final, learning_curve=learning,
        options=dict(inputs=args.inputs, windows=args.windows, layers=args.layers,
                     filters=args.filters, epochs=args.epochs, patience=args.patience, seed=args.seed)))
    print(f'Selected {winner.name}; results and model saved to {path}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--inputs', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--windows', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--layers', nargs='+', type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument('--filters', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-gap', type=float, default=DEFAULT_SETTINGS['max_gap_s'])
    parser.add_argument('--holdout-day', type=lambda value: date.fromisoformat(value).isoformat())
    parser.add_argument('--subject', help='subject_id for a personalized model')
    args = parser.parse_args(argv)
    if min(*args.windows, args.filters, args.epochs, args.patience) < 1 or not 0 < args.max_gap < float('inf'):
        parser.error('window, filters, epochs, patience and max gap must be positive')
    sessions, excluded = load_sessions(args.data_dir)
    if args.subject is not None:
        sessions = [session for session in sessions if session.subject == args.subject]
    if len({session.subject for session in sessions}) > 1:
        parser.error('multiple participant IDs found; select the demonstrator with --subject')
    if len(sessions) < 3:
        parser.error(f'need at least three real recordings; found {len(sessions)}; exclusions: {excluded}')
    path = args.output_dir or Path(__file__).with_name('reports') / (
        datetime.now().strftime('focus_%Y%m%d_%H%M%S_') + uuid4().hex[:6])
    try:
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise ValueError('output directory must be empty')
        run(sessions, excluded, args, path)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
