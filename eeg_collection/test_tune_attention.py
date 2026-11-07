import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import tune_attention as app


def session(name='one', values=(20, 20, 20, 80, 80, 80), labels=(0, 0, 0, 1, 1, 1)):
    rows = []
    phase_start = 0
    for i, (value, label) in enumerate(zip(values, labels)):
        if i and label != labels[i - 1]:
            phase_start = i
        rows.append(dict(time=float(i), phase_time=float(i - phase_start),
                         label=label, attention=float(value)))
    return dict(id=name, subject='P01', rows=rows)


def write_session(root, name='one', source='serial', status='completed'):
    data = session(name)
    fields = ['session_id', 'subject_id', 'elapsed_s', 'phase_elapsed_s', 'phase',
              'label', 'attention', 'source', 'valid', 'training_eligible']
    path = root / (name + '.csv')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in data['rows']:
            writer.writerow(dict(session_id=name, subject_id='P01', elapsed_s=r['time'],
                                 phase_elapsed_s=r['phase_time'], label=r['label'],
                                 phase='focus' if r['label'] else 'relax',
                                 attention=r['attention'], source=source, valid=1,
                                 training_eligible=1))
    path.with_suffix('.json').write_text(json.dumps(dict(session_id=name,
        subject_id='P01', source=source, status=status)), encoding='utf-8')
    return path


class FilterTests(unittest.TestCase):
    def test_known_causal_values_and_prefix_invariance(self):
        values = [30, 32, 31, 70, 33]
        self.assertEqual(app.smooth(values, app.Config('raw', 0, 50)), values)
        self.assertEqual(app.smooth(values, app.Config('ma', 3, 50)),
                         [30, 31, 31, 133 / 3, 134 / 3])
        ema = app.smooth([20, 80, 80], app.Config('ema', .5, 50))
        self.assertEqual(ema, [20, 50, 65])
        for method, parameter in [('raw', 0), ('ma', 3), ('ema', .3)]:
            c = app.Config(method, parameter, 50)
            self.assertEqual(app.smooth(values, c)[:3], app.smooth(values[:3], c))

    def test_filter_is_not_reset_at_known_phase_boundary(self):
        s = session(values=(20, 20, 20, 80, 80, 80))
        m = app.evaluate(s, app.Config('ma', 3, 50), app.Protocol(confirm_samples=2))
        self.assertEqual((m['tp'], m['tn'], m['fp'], m['fn']), (2, 3, 0, 1))
        self.assertEqual(m['confirmed_delay_s'], 2)

    def test_invalid_filter_parameters(self):
        for config in [('ma', 0, 50), ('ema', 0, 50), ('ema', 1.1, 50),
                       ('raw', 0, float('nan')), ('other', 2, 50)]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                app.Config(*config)


class MetricTests(unittest.TestCase):
    def test_threshold_inclusive_confusion_and_within_phase_flips(self):
        s = session(values=(20, 50, 20, 80, 20, 80))
        m = app.evaluate(s, app.Config('raw', 0, 50), app.Protocol(confirm_samples=1))
        self.assertEqual((m['tp'], m['tn'], m['fp'], m['fn']), (2, 2, 1, 1))
        self.assertAlmostEqual(m['balanced_accuracy'], 2 / 3)
        self.assertEqual(m['flips'], 4)
        self.assertEqual(m['confirmed_delay_s'], 0)

    def test_confirmation_uses_receive_time_and_full_run(self):
        s = session(values=(20, 20, 20, 80, 80, 80))
        s['rows'][4]['phase_time'] = 4
        s['rows'][5]['phase_time'] = 8
        m = app.evaluate(s, app.Config('raw', 0, 50), app.Protocol())
        self.assertEqual(m['confirmed_delay_s'], 8)

    def test_preexisting_target_and_missing_confirmation_are_penalized(self):
        for values, expected in [((20, 20, 80, 80, 80, 80), 'preexisting'),
                                 ((20, 20, 20, 80, 20, 80), 'missed')]:
            with self.subTest(values=values):
                m = app.evaluate(session(values=values), app.Config('raw', 0, 50), app.Protocol())
                self.assertIsNone(m['confirmed_delay_s'])
                self.assertEqual(m['delay_penalty_s'], 15)
                self.assertEqual(m[expected], 1)

    def test_confirmation_after_horizon_is_not_detected(self):
        s = session()
        s['rows'][5]['phase_time'] = 16
        m = app.evaluate(s, app.Config('raw', 0, 50), app.Protocol())
        self.assertEqual(m['missed'], 1)
        self.assertIsNone(m['confirmed_delay_s'])

    def test_balancing_weights_can_prefer_faster_candidate(self):
        sessions = [session('a'), session('b'), session('c')]
        result = app.analyze(sessions, [app.Config('ma', 3, 50), app.Config('raw', 0, 50)], app.Protocol())
        self.assertEqual(result['recommendation']['config']['method'], 'raw')

    def test_outer_choice_does_not_use_heldout_labels(self):
        sessions = [session('a'), session('b'), session('c')]
        grid = [app.Config('raw', 0, 30), app.Config('raw', 0, 70)]
        first = app.analyze(sessions, grid, app.Protocol())
        changed = deepcopy(sessions)
        for row in changed[2]['rows']:
            row['label'] = 1 - row['label']
        second = app.analyze(changed, grid, app.Protocol())
        def heldout(result):
            return next(f for f in result['folds'] if f['session_id'] == 'c' and f['strategy'] == 'selected')
        self.assertEqual(heldout(first)['config'], heldout(second)['config'])
        self.assertEqual(heldout(first)['train_session_ids'], ['a', 'b'])
        self.assertNotEqual(heldout(first)['metrics']['balanced_accuracy'],
                            heldout(second)['metrics']['balanced_accuracy'])


class InputAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parent / 'data' / 'test_tmp'
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_filters_simulation_incomplete_and_reports_bad_metadata(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            root = Path(d)
            write_session(root)
            write_session(root, 'sim', source='simulation')
            write_session(root, 'bad', status='interrupted')
            write_session(root, 'broken').with_suffix('.json').write_text('{', encoding='utf-8')
            sessions, exclusions = app.load_sessions(root)
            self.assertEqual([s['id'] for s in sessions], ['one'])
            self.assertEqual(len(exclusions), 3)

    def test_invalid_rows_are_skipped_but_bad_label_and_order_reject_session(self):
        for mutation in ['zero', 'label', 'time', 'nan', 'duplicate', 'phase_clock']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=self.temp_root) as d:
                root = Path(d)
                p = write_session(root)
                with p.open(encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    rows, fields = list(reader), reader.fieldnames
                if mutation == 'zero':
                    rows[0].update(attention='0', training_eligible='0')
                elif mutation == 'label':
                    rows[0]['label'] = '1'
                elif mutation == 'time':
                    rows[1]['elapsed_s'] = '-1'
                elif mutation == 'nan':
                    rows[0]['attention'] = 'nan'
                elif mutation == 'phase_clock':
                    rows[4]['phase_elapsed_s'] = '.001'
                    rows[5]['phase_elapsed_s'] = '.002'
                else:
                    (root / 'copy.csv').write_bytes(p.read_bytes())
                    (root / 'copy.json').write_bytes(p.with_suffix('.json').read_bytes())
                with p.open('w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                sessions, excluded = app.load_sessions(root)
                if mutation == 'zero':
                    self.assertEqual(len(sessions[0]['rows']), 5)
                    self.assertEqual(sessions[0]['skipped_rows'], 1)
                elif mutation == 'duplicate':
                    self.assertEqual(len(sessions), 1)
                    self.assertEqual(len(excluded), 1)
                else:
                    self.assertEqual(sessions, [])
                    self.assertEqual(len(excluded), 1)

    def test_cli_writes_offline_report_and_preserves_previous_run(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            root = Path(d)
            data = root / 'data'
            data.mkdir()
            for name in ['a', 'b', 'c']:
                write_session(data, name)
            write_session(data, 'sim', source='simulation')
            cmd = [sys.executable, str(Path(app.__file__)), '--data-dir', str(data),
                   '--output-dir', str(root / 'reports'), '--windows', '3',
                   '--alphas', '0.5', '--thresholds', '30,50,70']
            for _ in range(2):
                run = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
            reports = list((root / 'reports').iterdir())
            self.assertEqual(len(reports), 2)
            result = json.loads((reports[0] / 'recommendation.json').read_text(encoding='utf-8'))
            self.assertEqual(len(result['folds']), 15)
            self.assertEqual(len(result['excluded']), 1)
            self.assertEqual(result['validation']['baseline']['balanced_accuracy'], 1)
            self.assertFalse(result['validation_assessment']['supports_replacing_baseline'])
            self.assertTrue((reports[0] / 'candidates.csv').exists())
            html = (reports[0] / 'report.html').read_text(encoding='utf-8')
            self.assertNotIn('src="http', html)
            self.assertIn('<svg', html)
            self.assertIn('application/json', html)
            if shutil.which('node'):
                js = subprocess.run(['node', str(Path(app.__file__).with_name('test_attention_report.mjs')),
                                     str(reports[0] / 'report.html')], capture_output=True, text=True, encoding='utf-8')
                self.assertEqual(js.returncode, 0, js.stdout + js.stderr)

    def test_embedded_subject_cannot_terminate_json_script(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            sessions = [session('a'), session('b'), session('c')]
            for s in sessions:
                s['subject'] = '</script><script>alert(1)</script>'
            result = app.analyze(sessions, [app.Config('raw', 0, 50)], app.Protocol())
            folder = app.write_outputs(result, sessions, [], d)
            html = (folder / 'report.html').read_text(encoding='utf-8')
            self.assertNotIn('</script><script>alert(1)', html)
            embedded = html.split('id="reportData">', 1)[1].split('</script>', 1)[0]
            self.assertEqual(json.loads(embedded)['traces'][0]['subject'], sessions[0]['subject'])

    def test_cli_rejects_insufficient_sessions_and_bad_parameters(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            root = Path(d)
            write_session(root)
            for extra in [[], ['--alphas', 'nan'], ['--confirm-samples', '0'],
                          ['--delay-weight', '-1'], ['--thresholds', '']]:
                result = subprocess.run([sys.executable, app.__file__, '--data-dir', d, *extra],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
