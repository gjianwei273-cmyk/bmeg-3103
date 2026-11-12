import argparse
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from . import collect_data as app


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Stream:
    def __init__(self, clock, interrupt=False):
        self.clock = clock
        self.next = 0
        self.interrupt = interrupt

    def reset_input_buffer(self):
        pass

    def read(self, size):
        if self.interrupt and self.clock.now >= 0.5:
            raise KeyboardInterrupt
        if self.clock.now >= self.next:
            self.next = self.clock.now + 0.2
            return b'1,2,3,4,5,6,7,8,80\nwrong\n'
        return b''


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parent / 'data' / 'test_tmp'
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_parse_retains_all_nine_values(self):
        self.assertEqual(app.parse_sample('0,16777215,3,4,5,6,7,8,100'),
                         [0, 16777215, 3, 4, 5, 6, 7, 8, 100])

    def test_rejects_bad_columns_ranges_and_non_integer_values(self):
        for line in ['1,2', '1,2,3,4,5,6,7,8,101',
                     '-1,2,3,4,5,6,7,8,80',
                     '16777216,2,3,4,5,6,7,8,80',
                     '1.5,2,3,4,5,6,7,8,80']:
            with self.subTest(line=line), self.assertRaises(ValueError):
                app.parse_sample(line)

    def collect(self, directory, interrupt=False):
        clock = Clock()
        output = io.StringIO()
        result = app.collect(Stream(clock, interrupt), Path(directory),
                             subject='P01', notes='测试', port='FAKE',
                             relax_seconds=1, focus_seconds=1, prepare_seconds=0.2,
                             source='simulation', clock=clock.monotonic,
                             sleep=clock.sleep, console=output)
        with result['csv_path'].open(encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        return result, rows, output.getvalue()

    def test_one_round_labels_times_and_invalid_rows_are_saved(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            result, rows, messages = self.collect(directory)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual({r['phase'] for r in rows}, {'relax', 'focus'})
            self.assertEqual({r['label'] for r in rows if r['phase'] == 'relax'}, {'0'})
            self.assertEqual({r['label'] for r in rows if r['phase'] == 'focus'}, {'1'})
            self.assertTrue(all(0 <= float(r['phase_elapsed_s']) < 1 for r in rows))
            self.assertTrue(all(r['timestamp'].endswith('+08:00') for r in rows))
            self.assertTrue(any(r['valid'] == '0' and r['raw_line'] == 'wrong' for r in rows))
            self.assertTrue(all(r['delta'] == '1' for r in rows if r['valid'] == '1'))
            self.assertIn('放松', messages)
            self.assertIn('专注', messages)
            self.assertTrue(result['summary_path'].exists())

    def test_interrupt_keeps_written_data_and_summary(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            result, rows, _ = self.collect(directory, interrupt=True)
            self.assertEqual(result['status'], 'interrupted')
            self.assertTrue(rows)
            self.assertTrue(result['summary_path'].exists())

    def test_continuous_phases_keep_partial_lines_and_receive_timing_across_cues(self):
        class BoundaryStream(Stream):
            def __init__(self, clock):
                super().__init__(clock)
                self.resets = 0
                self.parts = [(0.15, b'1,2,3,4,5,6,7,8,81\n'),
                              (0.28, b'1,2,3,4,'),
                              (0.38, b'5,6,7,8,82\n'),
                              (0.63, b'1,2,3,4,5,6,7,8,83\n')]

            def reset_input_buffer(self):
                self.resets += 1

            def read(self, size):
                if self.parts and self.clock.now >= self.parts[0][0]:
                    return self.parts.pop(0)[1]
                return b''

        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            clock = Clock()
            stream = BoundaryStream(clock)
            messages = io.StringIO()
            result = app.collect(stream, directory,
                                 phases=[('focus', 0.25), ('relax', 0.25), ('focus', 0.30)],
                                 prepare_seconds=0.10, clock=clock.monotonic,
                                 sleep=clock.sleep, console=messages)
            with result['csv_path'].open(encoding='utf-8-sig') as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, app.FIELDS)
                rows = list(reader)
            summary = json.loads(result['summary_path'].read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(stream.resets, 1)
            self.assertEqual([(r['phase'], r['label'], r['attention']) for r in rows],
                             [('focus', '1', '81'), ('relax', '0', '82'), ('focus', '1', '83')])
            self.assertEqual([r['receive_interval_s'] for r in rows], ['', '0.250', '0.250'])
            self.assertEqual([r['phase_elapsed_s'] for r in rows], ['0.050'] * 3)
            events = summary['phase_events']
            self.assertEqual([(e['phase'], e['label'], e['duration_s']) for e in events],
                             [('focus', 1, 0.25), ('relax', 0, 0.25), ('focus', 1, 0.30)])
            for event, elapsed in zip(events, [0.10, 0.35, 0.60]):
                self.assertAlmostEqual(event['elapsed_s'], elapsed)
            self.assertIn('数呼吸', messages.getvalue())

    def test_phase_schedule_accepts_fractions_and_rejects_malformed_entries(self):
        self.assertEqual(app.parse_phases('focus:17.5, relax:23,focus:12'),
                         [('focus', 17.5), ('relax', 23.0), ('focus', 12.0)])
        for value in ['', 'focus', 'rest:5', 'focus:0', 'relax:inf', 'focus:1,']:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                app.parse_phases(value)

    def test_no_data_does_not_claim_success(self):
        class Empty(Stream):
            def read(self, size):
                return b''
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            clock = Clock()
            result = app.collect(Empty(clock), Path(directory), relax_seconds=0.2,
                                 focus_seconds=0.2, prepare_seconds=0,
                                 clock=clock.monotonic, sleep=clock.sleep,
                                 console=io.StringIO())
            self.assertEqual(result['status'], 'no_data')

    def test_fragmented_serial_lines_are_reassembled(self):
        class Fragments(Stream):
            def __init__(self, clock):
                super().__init__(clock)
                self.parts = iter([b'1,2,3,4,', b'5,6,7,8,0\r', b'\n'])

            def read(self, size):
                return next(self.parts, b'')

        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            clock = Clock()
            result = app.collect(Fragments(clock), directory, relax_seconds=0.2,
                                 focus_seconds=0.2, prepare_seconds=0,
                                 clock=clock.monotonic, sleep=clock.sleep,
                                 console=io.StringIO())
            with result['csv_path'].open(encoding='utf-8-sig') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['attention'], '0')
            self.assertEqual(rows[0]['valid'], '1')
            self.assertEqual(rows[0]['training_eligible'], '0')

    def test_disconnect_preserves_rows_and_reports_error(self):
        class Disconnect(Stream):
            def read(self, size):
                if self.clock.now > 0.1:
                    raise OSError('device disconnected')
                return super().read(size)

        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            clock = Clock()
            result = app.collect(Disconnect(clock), directory, prepare_seconds=0,
                                 clock=clock.monotonic, sleep=clock.sleep,
                                 console=io.StringIO())
            self.assertEqual(result['status'], 'error')
            with result['csv_path'].open(encoding='utf-8-sig') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_rejects_invalid_duration_arguments(self):
        for number in ['0', '-1', 'nan', 'inf']:
            with self.subTest(number=number), self.assertRaises(argparse.ArgumentTypeError):
                app.positive_seconds(number)


if __name__ == '__main__':
    unittest.main()
