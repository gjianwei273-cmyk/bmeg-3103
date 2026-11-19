"""Serve a loopback dashboard for live TGAM frames or a recorded session."""

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ...data_collecting.collect_data import FEATURES, parse_sample
from .focus_pipeline import feature_values, number


CHART_WINDOW_SECONDS = 60
DATA_DIRECTORY = Path(__file__).resolve().parents[2] / 'data_collecting' / 'data'


class SerialFrames:
    """Keep partial reads until a full Arduino line is received."""

    def __init__(self):
        self.pending = bytearray()
        self.discarding = False

    def feed(self, chunk):
        frames = []
        *lines, tail = chunk.split(b'\n')
        for line in lines:
            self._append(line)
            if self.discarding:
                frames.append(None)
            else:
                try:
                    frames.append(tuple(parse_sample(self.pending.decode('ascii'))))
                except ValueError:
                    frames.append(None)
            self.pending.clear()
            self.discarding = False
        self._append(tail)
        return frames

    def _append(self, fragment):
        if self.discarding:
            return
        if len(self.pending) + len(fragment) > 4096:
            self.discarding = True
            self.pending.clear()
        else:
            self.pending.extend(fragment)


def read_replay(stream):
    reader = csv.DictReader(stream)
    fields = set(reader.fieldnames or ())
    if not fields.intersection({'elapsed_s', 'timestamp'}) or not fields.intersection(FEATURES):
        raise ValueError('回放 CSV 需要 elapsed_s 或 timestamp，以及 EEG 特征字段。')
    rows = []
    first, previous = 0, -math.inf
    time_field = None
    for row in reader:
        if time_field is None:
            time_field = 'elapsed_s' if row.get('elapsed_s') else 'timestamp'
        try:
            if time_field == 'elapsed_s':
                instant = number(row.get(time_field))
            else:
                timestamp = row.get(time_field) or ''
                if timestamp.endswith('Z'):
                    timestamp = timestamp[:-1] + '+00:00'
                parsed = datetime.fromisoformat(timestamp)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                instant = parsed.timestamp()
            if instant is None or instant < previous or (time_field == 'elapsed_s' and instant < 0):
                raise ValueError
        except (OSError, OverflowError, ValueError) as exc:
            raise ValueError(f'CSV 第 {reader.line_num} 行 {time_field} 无效。') from exc
        if not rows:
            first = instant
        previous = instant
        rows.append((instant - first, feature_values(row)))
    if not rows:
        raise ValueError('回放 CSV 没有数据。')
    return rows


class DemoSession:
    """Serialize timer calls from the input worker and the HTTP server."""

    def __init__(self, timer, source, clock=time.monotonic):
        self.timer = timer
        self.source = source
        self.clock = clock
        self.started = threading.Event()
        self.source_status = 'ready' if source == 'replay' else 'live'
        self.lock = threading.Lock()
        self.chart_origin = self.clock()
        self.scores = deque()
        self.round_started_at = None
        self.round_result = None
        self.replay_finished_at = None
        self.timer.update(self.chart_origin, None)

    def _record_score(self, now, score):
        if score is not None or (self.scores and self.scores[-1][1] is not None):
            self.scores.append((now - self.chart_origin, score))
        self._trim_scores(now)

    def _trim_scores(self, now):
        cutoff = now - self.chart_origin - CHART_WINDOW_SECONDS
        while self.scores and (self.scores[0][0] < cutoff or self.scores[0][1] is None):
            self.scores.popleft()

    def _poll(self, now):
        last_frame = self.timer.window.last_frame
        result = self.timer.poll(now)
        if last_frame is not None and result['state'] == 'disconnected':
            self._record_score(last_frame + self.timer.window.max_gap, None)
        self._trim_scores(now)
        return result

    def _status(self, now, result):
        started = self.round_started_at is not None
        status = dict(result, started=started, source=self.source, source_status=self.source_status,
                      round_status='running' if started else 'idle',
                      elapsed_seconds=now - self.round_started_at if started else 0,
                      focus_seconds=result['focus_seconds'] if started else 0,
                      relax_seconds=result['relax_seconds'] if started else 0,
                      prediction_state=result['state'],
                      threshold=self.timer.settings['threshold'],
                      chart_seconds=now - self.chart_origin,
                      chart_window_seconds=CHART_WINDOW_SECONDS, scores=list(self.scores))
        if self.round_result is not None:
            status.update(self.round_result)
        return status

    def set_source_status(self, status):
        with self.lock:
            self.source_status = status

    def receive(self, values):
        if values is not None and values[8] == 0:
            values = values[:8] + (None,)
        with self.lock:
            now = self.clock()
            self._poll(now)
            result = self.timer.update(now, values)
            self._record_score(now, result['score'])

    def _snapshot(self):
        now = self.clock() if self.replay_finished_at is None else self.replay_finished_at
        return self._status(now, self._poll(now))

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def start(self):
        with self.lock:
            if self.replay_finished_at is not None:
                return self._snapshot()
            now = self.clock()
            self._poll(now)
            result = self.timer.reset_counter(now)
            self.round_started_at = now
            self.round_result = None
            self.scores.clear()
            self.started.set()
            if self.source_status == 'ready':
                self.source_status = 'playing'
            return self._status(now, result)

    def _end_round(self, now, result):
        if self.round_started_at is not None and self.round_result is None:
            self.round_result = dict(round_status='ended', elapsed_seconds=now - self.round_started_at,
                                     focus_seconds=result['focus_seconds'],
                                     relax_seconds=result['relax_seconds'], score=result['score'],
                                     prediction_state=result['state'],
                                     chart_seconds=now - self.chart_origin, scores=list(self.scores))

    def stop(self):
        with self.lock:
            now = self.clock() if self.replay_finished_at is None else self.replay_finished_at
            result = self._poll(now)
            self._end_round(now, result)
            return self._status(now, result)

    def finish(self):
        with self.lock:
            now = self.clock()
            self._end_round(now, self._poll(now))
            self._record_score(now, None)
            self.timer.disconnect()
            self.replay_finished_at = now
            self.source_status = 'finished'


def run_replay(session, rows, *, wait):
    """Play the recording once the dashboard's Start button is pressed."""
    while not session.started.is_set():
        if wait(0.1):
            return
    started = session.clock()
    for offset, values in rows:
        if wait(max(0, started + offset - session.clock())):
            return
        session.receive(values)
    session.finish()


def run_serial(session, serial, port, baudrate, stop):
    while not stop.is_set():
        try:
            with serial.Serial(port, baudrate, timeout=0.1) as stream:
                session.set_source_status('connected')
                frames = SerialFrames()
                while not stop.is_set():
                    for values in frames.feed(stream.read(4096)):
                        session.receive(values)
        except OSError as exc:
            session.receive(None)
            session.set_source_status('retrying')
            print(f'串口 {port} 断开，正在重试：{exc}', file=sys.stderr, flush=True)
            stop.wait(2)


class Dashboard:
    """Own one active data source and release its worker before switching."""

    def __init__(self, new_session, recordings, *, serial=None, serial_port_list, baudrate=57600):
        self.new_session = new_session
        self.recordings = {str(index): path for index, path in enumerate(recordings)}
        self.serial = serial
        self.serial_port_list = serial_port_list
        self.baudrate = baudrate
        self.lock = threading.Lock()
        self.session = new_session('serial')
        self.session.set_source_status('idle')
        self.source_id = None
        self.rows = None
        self.worker = None
        self.stop_worker = None

    def sources(self):
        return dict(
            serial_available=self.serial is not None,
            ports=[dict(id=port, name=label) for port, label in self.serial_port_list()],
            recordings=[dict(id=key, name=str(path.relative_to(DATA_DIRECTORY))
                             if path.is_relative_to(DATA_DIRECTORY) else str(path))
                        for key, path in self.recordings.items()],
        )

    def _shutdown_worker(self):
        if self.worker is not None:
            self.stop_worker.set()
            self.worker.join()
            self.worker = None
            self.stop_worker = None

    def _launch(self, run):
        self.stop_worker = threading.Event()
        self.worker = threading.Thread(target=run, args=(self.stop_worker,), daemon=True)
        self.worker.start()

    def _status(self, result):
        return dict(result, source_id=self.source_id)

    def snapshot(self):
        with self.lock:
            return self._status(self.session.snapshot())

    def select_source(self, mode, source_id, *, allow_unlisted=False):
        with self.lock:
            if self.session.snapshot()['round_status'] == 'running':
                raise ValueError('请先结束本轮练习。')
            rows = None
            if mode == 'replay':
                path = self.recordings.get(source_id)
                if path is None:
                    raise ValueError('请选择列表中的回放文件。')
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    rows = read_replay(stream)
            elif mode == 'serial':
                if source_id is not None:
                    if self.serial is None:
                        raise ValueError('Arduino 模式需要安装 pyserial。')
                    if not allow_unlisted and source_id not in dict(self.serial_port_list()):
                        raise ValueError('找不到该串口，请刷新串口列表。')
            else:
                raise ValueError('未知的数据来源。')

            self._shutdown_worker()
            self.session = self.new_session(mode)
            self.source_id = source_id
            self.rows = rows
            if mode == 'serial':
                self.session.set_source_status('idle' if source_id is None else 'connecting')
                if source_id is not None:
                    self._launch(lambda stop: run_serial(
                        self.session, self.serial, source_id, self.baudrate, stop))
            return self._status(self.session.snapshot())

    def start(self):
        with self.lock:
            current = self.session.snapshot()
            if current['round_status'] == 'running':
                return self._status(current)
            if self.session.source == 'serial':
                if self.source_id is None or current['source_status'] != 'connected':
                    raise ValueError('请先连接 Arduino。')
                return self._status(self.session.start())

            self._shutdown_worker()
            self.session = self.new_session('replay')
            result = self.session.start()
            self._launch(lambda stop: run_replay(self.session, self.rows, wait=stop.wait))
            return self._status(result)

    def stop(self):
        with self.lock:
            result = self.session.stop()
            if self.session.source == 'replay':
                self._shutdown_worker()
            return self._status(result)

    def close(self):
        with self.lock:
            self._shutdown_worker()


class LocalHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _respond(self, payload, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, value, status=200):
        self._respond(json.dumps(value, allow_nan=False).encode('utf-8'), 'application/json', status)

    def _select_source(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                raise ValueError
            payload = json.loads(self.rfile.read(size))
        except ValueError as exc:
            raise ValueError('无效的数据来源设置。') from exc
        if not isinstance(payload, dict) or not isinstance(payload.get('mode'), str):
            raise ValueError('无效的数据来源设置。')
        source_id = payload.get('id')
        if source_id is not None and not isinstance(source_id, str):
            raise ValueError('无效的数据来源设置。')
        return self.server.session.select_source(payload['mode'], source_id)

    def _local_host(self):
        port = self.server.server_port
        return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

    def do_GET(self):
        if not self._local_host():
            self.send_error(403)
        elif self.path == '/':
            self._respond(self.server.page, 'text/html; charset=utf-8')
        elif self.path == '/api/status':
            self._json(self.server.session.snapshot())
        elif self.path == '/api/sources':
            try:
                self._json(self.server.session.sources())
            except OSError:
                self._json({'error': '无法读取本机串口列表。'}, 503)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self._local_host():
            self.send_error(403)
            return
        actions = {'/api/start': self.server.session.start, '/api/stop': self.server.session.stop,
                   '/api/source': self._select_source}
        if self.path not in actions:
            self.send_error(404)
        elif self.headers.get('Content-Type') != 'application/json':
            self.send_error(415)
        else:
            try:
                self._json(actions[self.path]())
            except ValueError as exc:
                self._json({'error': str(exc)}, 400)
            except OSError:
                self._json({'error': '无法打开选定的数据来源。'}, 503)


class LocalServer(ThreadingHTTPServer):
    def __init__(self, port, session):
        self.session = session
        self.page = Path(__file__).with_suffix('.html').read_bytes()
        super().__init__(('127.0.0.1', port), LocalHandler)


def main():
    from .focus_pipeline import load_model_bundle
    from .focus_runtime import FocusTimer

    parser = argparse.ArgumentParser(description='在本机浏览器查看实时专注判定与专注时长。')
    parser.add_argument('--model', required=True, type=Path, help='训练导出的模型目录')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--port', help='Arduino 串口，例如 COM4')
    source.add_argument('--replay', type=Path, help='用已采集的 CSV 按原时间回放')
    parser.add_argument('--baudrate', type=int, default=57600)
    parser.add_argument('--http-port', type=int, default=8765, help='本地网页端口')
    args = parser.parse_args()
    if args.baudrate < 1 or not 0 <= args.http_port <= 65535:
        parser.error('波特率必须大于 0，网页端口必须在 0–65535 之间。')

    try:
        scorer, settings = load_model_bundle(args.model)
        try:
            import serial
            from serial.tools import list_ports
        except ImportError:
            serial = None

        def serial_port_list():
            if serial is None:
                return []
            return [(port.device, port.description) for port in sorted(list_ports.comports())]

        def new_session(mode):
            return DemoSession(FocusTimer(scorer, settings), mode)

        recordings = sorted(DATA_DIRECTORY.rglob('*.csv'))
        if args.replay:
            recording = args.replay.resolve()
            if recording in recordings:
                recordings.remove(recording)
            recordings.insert(0, recording)
        dashboard = Dashboard(new_session, recordings, serial=serial,
                              serial_port_list=serial_port_list, baudrate=args.baudrate)
        try:
            if args.replay:
                dashboard.select_source('replay', '0')
            elif args.port:
                dashboard.select_source('serial', args.port, allow_unlisted=True)
            with LocalServer(args.http_port, dashboard) as server:
                print(f'请打开 http://127.0.0.1:{server.server_port} （Ctrl+C 退出）', flush=True)
                server.serve_forever()
        finally:
            dashboard.close()
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as exc:
        parser.exit(2, f'无法启动演示：{exc}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
