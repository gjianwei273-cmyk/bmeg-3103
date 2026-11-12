"""Train and evaluate lightweight CNNs on labelled EEG recordings."""
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path

from focus_runtime import CausalWindow, FocusTimer, select_values


BANDS = ('delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta',
         'low_gamma', 'middle_gamma')
MODES = ('attention', 'bands', 'combined')
DEFAULT_DATA = Path(__file__).with_name('data')
DEFAULT_SETTINGS = dict(max_gap_s=2.5, threshold=.5, hysteresis=.1, on_frames=1, off_frames=2)


@dataclass(frozen=True)
class Frame:
    time: float
    label: int | None
    values: tuple | None


@dataclass(frozen=True)
class Session:
    id: str
    day: str | None
    subject: str | None
    frames: list
    events: list
    legacy_phase_pause: bool = False


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def parse_label(value, transition=False):
    if transition:
        if value in (None, '', 'transition'):
            return None
        raise ValueError('transition frame cannot carry a binary label')
    if value in ('focus', '1', 1):
        return 1
    if value in ('relax', '0', 0):
        return 0
    raise ValueError(f'invalid label: {value!r}')


def feature_values(row):
    if row.get('valid', '1') != '1' or row.get('poor_signal', '0') not in ('', '0'):
        return None
    powers = [number(row.get(band)) for band in BANDS]
    attention = number(row.get('attention'))
    return tuple(value if value is not None and value >= 0 else None for value in powers) + (
        attention if attention is not None and 0 < attention <= 100 else None,)


def load_session(path):
    summary = path.with_suffix('.json')
    metadata = json.loads(summary.read_text(encoding='utf-8-sig')) if summary.exists() else {}
    if metadata.get('status', 'completed') != 'completed' or metadata.get('source', 'serial') != 'serial':
        raise ValueError('requires a completed real recording')
    frames, events = [], []
    session_id, subject, day = metadata.get('session_id'), metadata.get('subject_id'), None
    previous_time = -math.inf
    with path.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row.get('source', 'serial') != 'serial':
                raise ValueError('simulated recordings are excluded')
            row_id, row_subject = row.get('session_id'), row.get('subject_id')
            session_id, subject = session_id or row_id, subject or row_subject
            if (row_id and row_id != session_id) or (row_subject and row_subject != subject):
                raise ValueError('each file must contain a single recording of one participant')
            timestamp = row.get('timestamp') or metadata.get('started_at')
            wall_time = datetime.fromisoformat(timestamp) if timestamp else None
            if wall_time and day is None:
                day = wall_time.date().isoformat()
            now = number(row.get('elapsed_s'))
            if now is None and not row.get('elapsed_s') and row.get('timestamp'):
                now = wall_time.timestamp()
            if now is None or now < previous_time:
                raise ValueError('expected monotonic elapsed_s or ISO timestamps')
            transition = row.get('transition_label') == '1' or row.get('phase') == 'transition'
            label = parse_label(row.get('label'), transition)
            phase_time = number(row.get('phase_elapsed_s'))
            if not events or events[-1][1] != label:
                events.append((now - phase_time if phase_time is not None else now, label))
            frames.append(Frame(now, label, feature_values(row)))
            previous_time = now
    if not frames:
        raise ValueError('recording has no frames')
    if metadata.get('phase_events'):
        events = [(number(event['elapsed_s']), parse_label(
            event['label'], event.get('phase') == 'transition' or event.get('transition_label') == 1))
                  for event in metadata['phase_events']]
        events = [event for index, event in enumerate(events) if not index or event[1] != events[index - 1][1]]
    if any(t is None for t, _ in events) or any(a[0] > b[0] for a, b in zip(events, events[1:])):
        raise ValueError('expected monotonic phase events')
    legacy_pause = bool(metadata.get('prepare_seconds') and not metadata.get('phase_events'))
    return Session(session_id or path.stem, day, subject, frames, events, legacy_pause)


def load_sessions(directory):
    sessions, excluded, seen = [], [], set()
    for path in sorted(Path(directory).glob('*.csv')):
        try:
            session = load_session(path)
            if session.id in seen:
                raise ValueError('duplicate recording id')
            sessions.append(session)
            seen.add(session.id)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            excluded.append(dict(file=path.name, reason=str(exc)))
    return sessions, excluded


def make_windows(session, mode, length, max_gap):
    buffer = CausalWindow(dict(input_mode=mode, window=length, max_gap_s=max_gap))
    for index, frame in enumerate(session.frames):
        window = buffer.push(frame.time, frame.values)
        if window is not None:
            yield index, frame, window


def validation_folds(sessions):
    """Keep the test and early-stopping/calibration recordings separate."""
    for index, test in enumerate(sessions):
        valid = sessions[(index + 1) % len(sessions)]
        yield [item for i, item in enumerate(sessions) if i not in (index, (index + 1) % len(sessions))], valid, test


def replay_states(session, scores, settings):
    timer = FocusTimer(lambda window: window[-1][8],
                       dict(settings, input_mode='attention', window=1))
    # A precomputed model score can legitimately be zero. None signals unavailability.
    return [timer.update(frame.time, (None,) * 8 + (score,) if score is not None else None)['state']
            for frame, score in zip(session.frames, scores)]


def interval_metrics(session, scores, states, max_gap, allowed=None):
    frames, events = session.frames, session.events
    times = [frame.time for frame in frames]
    event_times = [event[0] for event in events]
    result = dict(fp_seconds=0., fn_seconds=0., evaluated_seconds=0., flips=0,
                  tp=0, tn=0, fp=0, fn=0, on_delays_s=[], off_delays_s=[],
                  on_missed=0, off_missed=0, unobservable_transitions=0)
    supported = {i for i, score in enumerate(scores) if score is not None}
    if allowed is not None:
        supported.intersection_update(allowed)
    for index, (frame, score, state) in enumerate(zip(frames, scores, states)):
        if index not in supported:
            continue
        if frame.label is not None:
            prediction = int(score >= .5)
            key = ('t' if prediction == frame.label else 'f') + ('p' if prediction else 'n')
            result[key] += 1
        if index + 1 == len(frames):
            continue
        next_frame = frames[index + 1]
        long_gap = next_frame.time - frame.time >= max_gap
        if long_gap and session.legacy_phase_pause and next_frame.label != frame.label:
            continue
        stop = min(next_frame.time, frame.time + max_gap)
        start = frame.time
        event_index = bisect_right(event_times, start)
        label = frame.label
        for t, next_label in events[event_index:bisect_left(event_times, stop)] + [(stop, label)]:
            seconds = t - start
            if label is not None:
                result['evaluated_seconds'] += seconds
                if int(state == 'focus') != label:
                    result['fn_seconds' if label else 'fp_seconds'] += seconds
            start, label = t, next_label
        if (not long_gap and index + 1 in supported and frame.label is not None
                and next_frame.label == frame.label):
            result['flips'] += int(states[index + 1] != state)
    for event_index, (onset, label) in enumerate(events[1:], 1):
        if label is None:
            continue
        direction = 'on' if label else 'off'
        target = 'focus' if label else 'relax'
        before = bisect_left(times, onset) - 1
        after = bisect_left(times, onset)
        if (before not in supported or after not in supported
                or times[after] - times[before] >= max_gap):
            result['unobservable_transitions'] += 1
            continue
        stop = events[event_index + 1][0] if event_index + 1 < len(events) else math.inf
        observed = None
        if events[event_index - 1][1] is None and states[before] == target and states[after] == target:
            observed = 0.
        elif states[before] in ('focus', 'relax') and states[before] != target:
            for i in range(after, bisect_left(times, stop)):
                if i not in supported or times[i] - times[i - 1] >= max_gap:
                    break
                if states[i] == target:
                    observed = times[i] - onset
                    break
        if observed is None:
            result[f'{direction}_missed'] += 1
        else:
            result[f'{direction}_delays_s'].append(observed)
    return result


def replay_metrics(session, scorer, settings):
    scores = [None] * len(session.frames)
    for index, _, window in make_windows(session, settings['input_mode'], settings['window'], settings['max_gap_s']):
        scores[index] = float(scorer(window))
    return interval_metrics(session, scores, replay_states(session, scores, settings), settings['max_gap_s'])


def model_inputs(windows, mode):
    import numpy as np
    data = np.asarray([[select_values(values, mode) for values in window] for window in windows],
                      dtype=np.float32)
    if len(data) and mode != 'attention':
        data[:, :, :8] = np.log1p(data[:, :, :8])
    return data


def load_model_bundle(path):
    """Load the trained Keras model and preprocessing/timer settings."""
    import keras
    import numpy as np
    path = Path(path)
    with (path / 'model.json').open(encoding='utf-8') as handle:
        info = json.load(handle)
    model = keras.saving.load_model(path / 'model.keras', compile=False)
    settings = info['settings']
    mean = np.asarray(info['mean'], dtype=np.float32)
    scale = np.asarray(info['scale'], dtype=np.float32)

    def score(window):
        data = model_inputs([window], settings['input_mode'])
        return float(model((data - mean) / scale, training=False).numpy()[0, 0])

    return score, settings
