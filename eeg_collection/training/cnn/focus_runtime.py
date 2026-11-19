"""Shared causal window and timer for offline replay and live inference."""
from collections import deque
import math


def select_values(values, mode):
    if values is None:
        return None
    if mode == 'attention':
        selected = values[8:]
    elif mode == 'bands':
        selected = values[:8]
    elif mode == 'combined':
        selected = values
    else:
        raise ValueError(f'unknown input mode: {mode}')
    if any(value is None for value in selected):
        return None
    return selected


class CausalWindow:
    def __init__(self, settings):
        self.mode = settings['input_mode']
        self.max_gap = settings['max_gap_s']
        self.frames = deque(maxlen=settings['window'])
        self.last_frame = None

    def expire(self, now):
        if self.last_frame is not None and now >= self.last_frame + self.max_gap:
            self.clear()
            return True
        return False

    def clear(self):
        self.frames.clear()
        self.last_frame = None

    def push(self, now, values):
        self.expire(now)
        if select_values(values, self.mode) is None:
            self.clear()
            return None
        self.last_frame = now
        self.frames.append(values)
        return tuple(self.frames) if len(self.frames) == self.frames.maxlen else None


class FocusTimer:
    def __init__(self, scorer, settings):
        self.scorer = scorer
        self.settings = settings
        self.window = CausalWindow(settings)
        self.state = 'warming'
        self.score = None
        self.time = None
        self.seconds = {'focus': 0., 'relax': 0.}
        self.pending = 0

    def snapshot(self):
        return dict(state=self.state, score=self.score, focus_seconds=self.seconds['focus'],
                    relax_seconds=self.seconds['relax'])

    def poll(self, timestamp_seconds):
        now = timestamp_seconds
        if not math.isfinite(now) or (self.time is not None and now < self.time):
            raise ValueError('expected monotonic finite timestamps')
        last = self.window.last_frame
        if last is not None and self.state in self.seconds:
            self.seconds[self.state] += min(now, last + self.window.max_gap) - self.time
        self.time = now
        if self.window.expire(now):
            self.disconnect()
        return self.snapshot()

    def disconnect(self):
        self.window.clear()
        self.state = 'disconnected'
        self.score = None
        self.pending = 0

    def update(self, timestamp_seconds, values):
        self.poll(timestamp_seconds)
        window = self.window.push(timestamp_seconds, values)
        if self.window.last_frame is None:
            self.disconnect()
        elif window is None:
            self.state = 'warming'
        else:
            self.score = float(self.scorer(window))
            threshold = self.settings['threshold']
            if self.state == 'focus':
                change = self.score < threshold - self.settings['hysteresis']
                required = self.settings['off_frames']
                target = 'relax'
            else:
                self.state = 'relax'
                change = self.score >= threshold
                required = self.settings['on_frames']
                target = 'focus'
            self.pending = self.pending + 1 if change else 0
            if self.pending >= required:
                self.state = target
                self.pending = 0
        return self.snapshot()

    def reset_counter(self, timestamp_seconds):
        self.poll(timestamp_seconds)
        self.seconds = {'focus': 0., 'relax': 0.}
        return self.snapshot()
