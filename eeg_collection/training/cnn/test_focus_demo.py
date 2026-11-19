import csv
from http.client import HTTPResponse
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from ...data_collecting.collect_data import FEATURES
from .focus_demo import (
    Dashboard, DemoSession, LocalHandler, LocalServer, SerialFrames, read_replay, run_replay,
)
from .focus_runtime import FocusTimer


FRAME = (1, 2, 3, 4, 5, 6, 7, 8, 80)
SETTINGS = dict(window=1, input_mode='attention', max_gap_s=2.5, threshold=0.5,
                hysteresis=0.1, on_frames=1, off_frames=1)


def replay_csv(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['elapsed_s', 'valid', *FEATURES])
    writer.writerows(rows)
    stream.seek(0)
    return stream


def new_session(clock, source='serial'):
    timer = FocusTimer(lambda window: window[-1][8] / 100, SETTINGS)
    return DemoSession(timer, source, clock=clock)


class SerialFramesTests(unittest.TestCase):
    def test_partial_and_invalid_frames(self):
        frames = SerialFrames()
        self.assertEqual(frames.feed(b'1,2,3'), [])
        self.assertEqual(frames.feed(b',4,5,6,7,8,80\r\ninvalid\n'), [FRAME, None])

    def test_oversized_line_is_rejected_through_its_newline(self):
        frames = SerialFrames()
        self.assertEqual(frames.feed(b'x' * 4097), [])
        self.assertEqual(frames.feed(b'1,2,3,4,5,6,7,8,80\n1,2,3,4,5,6,7,8,80\n'),
                         [None, FRAME])


class ReplayTests(unittest.TestCase):
    def test_preserves_recorded_gaps_and_invalid_rows(self):
        stream = replay_csv([[5, 1, *FRAME], [5.75, 0], [11, 1, *FRAME]])
        self.assertEqual(read_replay(stream), [(0, FRAME), (0.75, None), (6, FRAME)])

    def test_rejects_nonmonotonic_times(self):
        stream = replay_csv([[5, 1, *FRAME], [4, 1, *FRAME]])
        with self.assertRaisesRegex(ValueError, 'elapsed_s'):
            read_replay(stream)

    def test_accepts_labelled_iso_recordings_with_only_bands(self):
        stream = io.StringIO(
            'timestamp,label,poor_signal,' + ','.join(FEATURES[:8]) + '\n'
            '2026-11-05T18:00:00.000+08:00,relax,0,1,2,3,4,5,6,7,8\n'
            '2026-11-05T18:00:01.750+08:00,focus,25,1,2,3,4,5,6,7,8\n'
        )
        self.assertEqual(read_replay(stream), [(0, FRAME[:8] + (None,)), (1.75, None)])

    def test_replay_waits_for_start_and_uses_relative_offsets(self):
        class ReplayClock:
            now = 10
            def __call__(self):
                return self.now
            def wait(self, seconds):
                self.now += seconds
                return False

        clock = ReplayClock()
        session = new_session(clock, 'replay')
        self.assertFalse(session.started.is_set())
        session.start()
        run_replay(session, [(0, FRAME), (1.75, FRAME)], wait=clock.wait)
        snapshot = session.snapshot()
        self.assertAlmostEqual(snapshot['focus_seconds'], 1.75)
        self.assertEqual(snapshot['state'], 'disconnected')
        self.assertEqual(snapshot['source_status'], 'finished')
        self.assertEqual(session.start(), snapshot)


class SessionTests(unittest.TestCase):
    def test_score_history_uses_inference_times_and_marks_signal_gaps(self):
        now = [100]
        session = new_session(lambda: now[0])
        session.receive(FRAME)
        now[0] = 101
        session.receive(FRAME[:-1] + (30,))
        now[0] = 105
        result = session.snapshot()
        self.assertEqual(result['threshold'], SETTINGS['threshold'])
        self.assertEqual(result['chart_seconds'], 5)
        self.assertEqual(result['scores'], [(0, 0.8), (1, 0.3), (3.5, None)])
        now[0] = 106
        session.receive(FRAME)
        self.assertEqual(session.snapshot()['scores'][-1], (6, 0.8))
        now[0] = 167
        self.assertEqual(session.snapshot()['scores'], [])

    def test_stop_freezes_both_timers_until_new_round(self):
        now = [10]
        session = new_session(lambda: now[0])
        session.receive(FRAME)
        session.start()
        now[0] = 11
        session.receive(FRAME[:-1] + (30,))
        now[0] = 15
        result = session.stop()
        self.assertEqual(result['round_status'], 'ended')
        self.assertEqual(result['elapsed_seconds'], 5)
        self.assertEqual(result['focus_seconds'], 1)
        self.assertEqual(result['relax_seconds'], 2.5)
        self.assertEqual(result['prediction_state'], 'disconnected')
        now[0] = 90
        session.receive(FRAME)
        frozen = session.snapshot()
        for field in ('scores', 'chart_seconds', 'score', 'elapsed_seconds', 'focus_seconds',
                      'relax_seconds', 'prediction_state'):
            self.assertEqual(frozen[field], result[field], field)
        self.assertEqual(session.stop()['elapsed_seconds'], 5)
        result = session.start()
        self.assertEqual(result['round_status'], 'running')
        self.assertEqual(result['elapsed_seconds'], 0)
        self.assertEqual(result['focus_seconds'], 0)
        self.assertEqual(result['relax_seconds'], 0)
        self.assertEqual(result['prediction_state'], 'focus')
        self.assertEqual(result['scores'], [])

    def test_relax_time_excludes_warmup_invalid_frames_and_signal_gaps(self):
        now = [10]
        timer = FocusTimer(lambda window: window[-1][8] / 100, dict(SETTINGS, window=2))
        session = DemoSession(timer, 'serial', clock=lambda: now[0])
        session.start()
        for instant, frame in [(11, FRAME[:-1] + (30,)), (12, FRAME[:-1] + (30,)),
                               (13, None), (17, FRAME[:-1] + (30,)), (18, FRAME[:-1] + (30,)),
                               (19, FRAME)]:
            now[0] = instant
            session.receive(frame)
        now[0] = 20
        result = session.stop()
        self.assertEqual(result['elapsed_seconds'], 10)
        self.assertEqual(result['relax_seconds'], 2)
        self.assertEqual(result['focus_seconds'], 1)
        self.assertEqual(result['prediction_state'], 'focus')

    def test_waiting_for_first_serial_frame_shows_disconnected(self):
        session = new_session(lambda: 10)
        self.assertEqual(session.snapshot()['state'], 'disconnected')

    def test_zero_attention_disconnects_attention_model(self):
        session = new_session(lambda: 10)
        session.receive(FRAME[:-1] + (0,))
        self.assertEqual(session.snapshot()['state'], 'disconnected')

    def test_zero_attention_keeps_bands_available(self):
        timer = FocusTimer(lambda window: window[-1][0], dict(SETTINGS, input_mode='bands'))
        session = DemoSession(timer, 'serial', clock=lambda: 10)
        session.receive(FRAME[:-1] + (0,))
        self.assertEqual(session.snapshot()['state'], 'focus')

    def test_start_and_reset_keep_pretrained_window_but_clear_time(self):
        now = [10]
        session = new_session(lambda: now[0])
        session.receive(FRAME)
        now[0] = 11
        session.receive(FRAME)
        self.assertEqual(session.snapshot()['focus_seconds'], 0)
        self.assertEqual(session.start()['state'], 'focus')
        now[0] = 12
        session.receive(FRAME)
        self.assertAlmostEqual(session.snapshot()['focus_seconds'], 1)
        self.assertEqual(session.start()['focus_seconds'], 0)
        now[0] = 20
        self.assertEqual(session.snapshot()['state'], 'disconnected')


class SourceSelectionTests(unittest.TestCase):
    def test_switch_to_recording_and_restart_it_after_early_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / 'demo.csv'
            recording.write_text(replay_csv([[0, 1, *FRAME], [600, 1, *FRAME]]).getvalue(),
                                 encoding='utf-8')
            dashboard = Dashboard(lambda mode: new_session(lambda: 100, mode), [recording],
                                  serial_port_list=lambda: [])
            self.addCleanup(dashboard.close)
            self.assertEqual(dashboard.snapshot()['source_status'], 'idle')
            options = dashboard.sources()
            self.assertEqual(options['recordings'], [{'id': '0', 'name': str(recording)}])
            with self.assertRaises(ValueError):
                dashboard.select_source('replay', '../arbitrary.csv')
            dashboard.select_source('replay', '0')
            self.assertEqual(dashboard.snapshot()['source_status'], 'ready')
            dashboard.start()
            worker = dashboard.worker
            self.assertEqual(dashboard.stop()['round_status'], 'ended')
            self.assertFalse(worker.is_alive())
            self.assertEqual(dashboard.start()['round_status'], 'running')
            self.assertIsNot(worker, dashboard.worker)

    def test_connect_listed_port_and_close_it_when_switching_sources(self):
        connected = threading.Event()
        closed = threading.Event()

        class FakeSerial:
            @staticmethod
            def Serial(port, baudrate, timeout):
                if (port, baudrate, timeout) != ('COM7', 57600, 0.1):
                    raise AssertionError('unexpected serial settings')
                return FakeSerial()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                closed.set()

            def read(self, size):
                connected.set()
                return b'1,2,3,4,5,6,7,8,80\n'

        dashboard = Dashboard(lambda mode: new_session(lambda: 100, mode), [],
                              serial=FakeSerial, serial_port_list=lambda: [('COM7', 'Arduino')])
        self.addCleanup(dashboard.close)
        self.assertEqual(dashboard.sources()['ports'], [{'id': 'COM7', 'name': 'Arduino'}])
        with self.assertRaises(ValueError):
            dashboard.select_source('serial', 'COM8')
        dashboard.select_source('serial', 'COM7')
        self.assertTrue(connected.wait(1))
        status = dashboard.snapshot()
        self.assertEqual((status['source'], status['source_id'], status['source_status']),
                         ('serial', 'COM7', 'connected'))
        dashboard.select_source('serial', None)
        self.assertTrue(closed.is_set())
        self.assertEqual(dashboard.snapshot()['source_status'], 'idle')

    def test_replay_can_start_again_after_end_of_file(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / 'short.csv'
            recording.write_text(replay_csv([[0, 1, *FRAME]]).getvalue(), encoding='utf-8')
            dashboard = Dashboard(lambda mode: new_session(lambda: 100, mode), [recording],
                                  serial_port_list=lambda: [])
            self.addCleanup(dashboard.close)
            dashboard.select_source('replay', '0')
            for _ in range(2):
                self.assertEqual(dashboard.start()['round_status'], 'running')
                dashboard.worker.join(1)
                status = dashboard.snapshot()
                self.assertEqual(status['source_status'], 'finished')
                self.assertEqual(status['round_status'], 'ended')
                self.assertEqual(status['scores'], [(0, 0.8)])


class LocalServerTests(unittest.TestCase):
    def request(self, server, request_line, content_type=None, body=None):
        headers = f'Host: 127.0.0.1:{server.server_port}\r\n'
        if content_type:
            headers += f'Content-Type: {content_type}\r\n'
        payload = json.dumps(body).encode('utf-8') if body is not None else b''
        if payload:
            headers += f'Content-Length: {len(payload)}\r\n'
        packet = (request_line + ' HTTP/1.1\r\n' + headers + '\r\n').encode('ascii') + payload

        class MemoryConnection:
            def __init__(self, data):
                self.data = data
                self.sent = bytearray()

            def makefile(self, mode, *args):
                return io.BytesIO(self.data)

            def sendall(self, data):
                self.sent.extend(data)

        connection = MemoryConnection(packet)
        LocalHandler(connection, ('127.0.0.1', 0), server)
        response = HTTPResponse(MemoryConnection(connection.sent))
        response.begin()
        self.addCleanup(response.close)
        return response

    def test_serves_dashboard_status_and_start(self):
        session = new_session(lambda: 10)
        with LocalServer(0, session) as server:
            self.assertEqual(server.server_address[0], '127.0.0.1')
            page = self.request(server, 'GET /')
            self.assertEqual(page.headers.get_content_type(), 'text/html')
            self.assertFalse(json.load(self.request(server, 'GET /api/status'))['started'])
            self.assertEqual(self.request(server, 'POST /api/start').status, 415)
            result = self.request(server, 'POST /api/start', 'application/json')
            self.assertTrue(json.load(result)['started'])
            self.assertEqual(self.request(server, 'POST /api/stop').status, 415)
            result = self.request(server, 'POST /api/stop', 'application/json')
            self.assertEqual(json.load(result)['round_status'], 'ended')


    def test_source_api_validates_payload_and_reports_unavailable_ports(self):
        dashboard = Dashboard(lambda mode: new_session(lambda: 10, mode), [],
                              serial_port_list=lambda: [])
        with LocalServer(0, dashboard) as server:
            self.assertEqual(json.load(self.request(server, 'GET /api/sources')),
                             {'serial_available': False, 'ports': [], 'recordings': []})
            for payload in ({'mode': 'serial', 'id': []}, {'mode': 'serial', 'id': 'COM4'},
                            {'mode': 'replay', 'id': '../outside.csv'}):
                response = self.request(server, 'POST /api/source', 'application/json', payload)
                self.assertEqual(response.status, 400)
                self.assertIn('error', json.load(response))
            response = self.request(server, 'POST /api/source', 'application/json',
                                    {'mode': 'serial', 'id': None})
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response)['source_status'], 'idle')


if __name__ == '__main__':
    unittest.main()
