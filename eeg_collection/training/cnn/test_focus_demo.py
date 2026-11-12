import csv
from http.client import HTTPResponse
import io
import json
import unittest

from ...data_collecting.collect_data import FEATURES
from .focus_demo import DemoSession, LocalHandler, LocalServer, SerialFrames, read_replay, run_replay
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


class LocalServerTests(unittest.TestCase):
    def request(self, server, request_line, content_type=None):
        headers = f'Host: 127.0.0.1:{server.server_port}\r\n'
        if content_type:
            headers += f'Content-Type: {content_type}\r\n'
        packet = (request_line + ' HTTP/1.1\r\n' + headers + '\r\n').encode('ascii')

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


if __name__ == '__main__':
    unittest.main()
