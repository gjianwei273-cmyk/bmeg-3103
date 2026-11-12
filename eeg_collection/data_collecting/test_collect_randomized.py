import csv
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

from . import collect_randomized as app


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class SerialFrames:
    def __init__(self, clock, frames):
        self.clock = clock
        self.frames = iter(frames)
        self.next_frame = next(self.frames, None)

    def reset_input_buffer(self):
        pass

    def read(self, size):
        if self.next_frame is None or self.clock.now < self.next_frame[0]:
            return b''
        _, data = self.next_frame
        self.next_frame = next(self.frames, None)
        if data is None:
            raise KeyboardInterrupt
        return data


class RandomizedCollectorTests(unittest.TestCase):
    def test_random_schedules_alternate_and_preserve_each_sixty_second_budget(self):
        schedules = set()
        for seed in range(30):
            schedule = app.make_schedule(random.Random(seed))
            schedules.add(tuple(schedule))
            active = schedule[::2]
            self.assertEqual(len(active), 6)
            self.assertEqual(schedule[1::2], [('transition', 5)] * 5)
            self.assertTrue(all(a[0] != b[0] for a, b in zip(active, active[1:])))
            self.assertTrue(all(15 <= seconds <= 25 for _, seconds in active))
            for phase in ('relax', 'focus'):
                self.assertEqual(sum(seconds for name, seconds in active if name == phase), 60)
        self.assertGreater(len(schedules), 1)

    def run_collection(self, frames):
        clock = Clock()
        stream = SerialFrames(clock, frames)
        schedule = [('relax', 1), ('transition', 5), ('focus', 1)]
        temp_root = Path(__file__).resolve().parent / 'data' / 'test_tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            result = app.collect(stream, Path(directory), schedule, prepare_seconds=0,
                                 clock=clock.monotonic, sleep=clock.sleep, console=io.StringIO())
            with result['csv_path'].open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(result['summary_path'].read_text(encoding='utf-8'))
        return rows, summary

    def test_transition_keeps_partial_and_invalid_rows_without_a_binary_label(self):
        rows, summary = self.run_collection([
            (0.1, b'1,2,3,4,5,6,7,8,42\n'),
            (0.8, b'1,2,3,4,'),
            (1.1, b'5,6,7,8,52\nmalformed\n'),
            (6.2, b'1,2,3,4,5,6,7,8,62\n'),
        ])
        self.assertEqual([(r['phase'], r['label'], r['transition_label'], r['attention'])
                          for r in rows],
                         [('relax', '0', '0', '42'), ('transition', '', '1', '52'),
                          ('transition', '', '1', ''), ('focus', '1', '0', '62')])
        self.assertEqual([r['training_eligible'] for r in rows], ['1', '0', '0', '1'])
        self.assertEqual(rows[2]['raw_line'], 'malformed')
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['transition_seconds'], 5)
        self.assertEqual([e['elapsed_s'] for e in summary['phase_events']], [0, 1, 6])

    def test_interrupt_during_transition_keeps_data_and_planned_schedule(self):
        rows, summary = self.run_collection([
            (0.1, b'1,2,3,4,5,6,7,8,42\n'),
            (1.2, b'1,2,3,4,5,6,7,8,52\n'),
            (2.0, None),
        ])
        self.assertEqual([r['transition_label'] for r in rows], ['0', '1'])
        self.assertEqual(summary['status'], 'interrupted')
        self.assertEqual(len(summary['phase_events']), 2)
        self.assertEqual(summary['schedule'][-1]['phase'], 'focus')
