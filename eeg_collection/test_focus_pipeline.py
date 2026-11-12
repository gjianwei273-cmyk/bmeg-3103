import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from focus_pipeline import (BANDS, Frame, Session, interval_metrics, load_sessions,
                            load_model_bundle, make_windows, replay_metrics, validation_folds)
from focus_runtime import FocusTimer


SETTINGS = dict(window=2, input_mode='bands', max_gap_s=2.5, threshold=.5,
                hysteresis=.1, on_frames=1, off_frames=2)
HIGH = (1,) * 8 + (None,)
LOW = (0,) * 8 + (None,)


class FocusTests(unittest.TestCase):
    def test_timer_warms_up_and_pauses_on_stale_or_invalid_input(self):
        timer = FocusTimer(lambda window: window[-1][0], SETTINGS)
        self.assertEqual(timer.update(1, HIGH)['state'], 'warming')
        self.assertEqual(timer.update(2, HIGH)['state'], 'focus')
        self.assertEqual(timer.poll(3)['focus_seconds'], 1)
        self.assertEqual(timer.update(3, LOW)['state'], 'focus')
        self.assertEqual(timer.update(4, LOW)['state'], 'relax')
        self.assertEqual(timer.update(4.5, None)['state'], 'disconnected')
        self.assertEqual(timer.update(5, HIGH)['state'], 'warming')
        self.assertEqual(timer.update(6, HIGH)['state'], 'focus')
        self.assertEqual(timer.poll(10)['state'], 'disconnected')
        self.assertEqual(timer.poll(10)['focus_seconds'], 4.5)
        self.assertEqual(timer.reset_counter(10)['focus_seconds'], 0)

    def test_labelled_csv_accepts_variable_blocks_and_attention_free_windows(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'sample.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, ['elapsed_s', 'label', 'attention', 'training_eligible', *BANDS])
                writer.writeheader()
                for t, label in [(1, 1), (2, 1), (3, 0), (4, 1), (10, 0), (11, 0)]:
                    writer.writerow(dict(elapsed_s=t, label=label, attention=0,
                                         training_eligible=0, **dict.fromkeys(BANDS, 1)))
            sessions, excluded = load_sessions(directory)
        self.assertFalse(excluded)
        self.assertEqual(sessions[0].events, [(1, 1), (3, 0), (4, 1), (10, 0)])
        windows = list(make_windows(sessions[0], 'bands', 2, 2.5))
        self.assertEqual([(i, row.label) for i, row, _ in windows], [(1, 1), (2, 0), (3, 1), (5, 0)])
        self.assertFalse(list(make_windows(sessions[0], 'combined', 2, 2.5)))

    def test_fold_partitions_are_disjoint(self):
        for train, valid, test in validation_folds(list(range(5))):
            self.assertEqual(len(train), 3)
            self.assertEqual(len({*train, valid, test}), 5)

    def test_replay_measures_both_edges_and_real_seconds(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'sample.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, ['elapsed_s', 'label', *BANDS])
                writer.writeheader()
                for t, label, signal in [(0, 0, 0), (1, 1, 0), (2, 1, 1), (3, 0, 1), (4, 0, 0), (5, 0, 0)]:
                    writer.writerow(dict(elapsed_s=t, label=label, **dict.fromkeys(BANDS, signal)))
            session = load_sessions(directory)[0][0]
        settings = dict(SETTINGS, window=1, off_frames=1)
        result = replay_metrics(session, lambda window: window[-1][0], settings)
        self.assertEqual(result['fp_seconds'], 1)
        self.assertEqual(result['fn_seconds'], 1)
        self.assertEqual(result['on_delays_s'], [1])
        self.assertEqual(result['off_delays_s'], [1])


    def test_cnn_configuration_accepts_every_frame_and_channel(self):
        from focus_train import build_model
        model = build_model(window=4, width=9, layers=3, filters=4, seed=0)
        self.assertEqual(model.input_shape, (None, 4, 9))
        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(sum(layer.__class__.__name__ == 'Conv1D' for layer in model.layers), 3)

    def test_comparison_masks_edges_and_flips_outside_shared_support(self):
        session = Session('s', None, None, [Frame(t, int(t >= 2), LOW) for t in range(5)],
                          [(0, 0), (2, 1)])
        result = interval_metrics(session, [.1, .1, .9, .9, .1],
                                  ['relax', 'relax', 'focus', 'focus', 'relax'], 2.5, {0, 1, 3})
        self.assertEqual(result['unobservable_transitions'], 1)
        self.assertFalse(result['on_delays_s'])
        self.assertEqual(result['flips'], 0)

    def test_known_phase_gap_counts_the_same_grace_as_the_timer(self):
        session = Session('s', None, None, [Frame(0, 0, HIGH), Frame(10, 0, HIGH)], [(0, 0)])
        result = replay_metrics(session, lambda window: 1., dict(SETTINGS, window=1))
        self.assertEqual(result['fp_seconds'], 2.5)

    def test_transition_rows_keep_signal_but_do_not_receive_binary_labels(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'session.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, ['elapsed_s', 'label', 'transition_label', *BANDS])
                writer.writeheader()
                for t, label in enumerate([0, '', '', 1, 1]):
                    writer.writerow(dict(elapsed_s=t, label=label, transition_label=int(label == ''),
                                         **dict.fromkeys(BANDS, int(t >= 2))))
            path.with_suffix('.json').write_text(json.dumps(dict(phase_events=[
                dict(elapsed_s=0, label=0), dict(elapsed_s=1, label=None, phase='transition'),
                dict(elapsed_s=3, label=1)])), encoding='utf-8')
            sessions, excluded = load_sessions(directory)
            self.assertFalse(excluded)
            session = sessions[0]
        windows = list(make_windows(session, 'bands', 2, 2.5))
        self.assertEqual([frame.label for _, frame, _ in windows], [None, None, 1, 1])
        result = replay_metrics(session, lambda window: window[-1][0], dict(SETTINGS, window=1))
        self.assertEqual(result['evaluated_seconds'], 2)
        self.assertEqual(result['tp'] + result['tn'], 3)
        self.assertEqual(result['on_delays_s'], [0])

    def test_band_only_training_exports_bundle_and_reserves_entire_day(self):
        from focus_train import main
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            data = Path(directory) / 'data'
            data.mkdir()
            output = Path(directory) / 'report'
            for index in range(4):
                day = 2 if index == 3 else 1
                with (data / f'session{index}.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, ['timestamp', 'elapsed_s', 'label', *BANDS])
                    writer.writeheader()
                    for t in range(48):
                        label = int(16 <= t < 34)
                        writer.writerow(dict(timestamp=f'2026-11-0{day}T12:00:{t:02}+08:00',
                                             elapsed_s=t, label=label,
                                             **dict.fromkeys(BANDS, 1 + label * 100)))
            main(['--data-dir', str(data), '--output-dir', str(output),
                  '--inputs', 'bands', '--windows', '4', '--layers', '1',
                  '--holdout-day', '2026-11-02', '--epochs', '1'])
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual([item['id'] for item in report['development']],
                             ['session0', 'session1', 'session2'])
            self.assertEqual(report['heldout_results'][0]['test'], 'session3')
            self.assertIsNone(report['heldout_results'][0]['reference'])
            self.assertTrue((output / 'curves.png').is_file())
            scorer, settings = load_model_bundle(output)
            self.assertEqual(settings['input_mode'], 'bands')
            self.assertTrue(0 <= scorer([HIGH] * settings['window']) <= 1)


if __name__ == '__main__':
    unittest.main()
