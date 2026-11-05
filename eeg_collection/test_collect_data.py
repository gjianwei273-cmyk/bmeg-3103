import csv
import io
import tempfile
import unittest
from pathlib import Path

import collect_data as app


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
        import argparse
        for number in ['0', '-1', 'nan', 'inf']:
            with self.subTest(number=number), self.assertRaises(argparse.ArgumentTypeError):
                app.positive_seconds(number)


if __name__ == '__main__':
    unittest.main()
