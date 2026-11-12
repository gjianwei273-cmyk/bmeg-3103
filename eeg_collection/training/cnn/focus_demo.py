"""Serve a loopback dashboard for live TGAM frames or a recorded session."""

import argparse
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
        self.final_result = None
        self.timer.update(self.clock(), None)

    def _status(self, result):
        started = self.started.is_set()
        return dict(result, started=started, source=self.source, source_status=self.source_status,
                    focus_seconds=result['focus_seconds'] if started else 0)

    def receive(self, values):
        if values is not None and values[8] == 0:
            values = values[:8] + (None,)
        with self.lock:
            self.timer.update(self.clock(), values)

    def snapshot(self):
        with self.lock:
            result = self.final_result
            if result is None:
                result = self.timer.poll(self.clock())
            return self._status(result)

    def start(self):
        with self.lock:
            if self.final_result is not None:
                return self._status(self.final_result)
            result = self.timer.reset_counter(self.clock())
            self.started.set()
            if self.source_status == 'ready':
                self.source_status = 'playing'
            return self._status(result)

    def finish(self):
        with self.lock:
            self.final_result = dict(self.timer.poll(self.clock()), state='disconnected', score=None)
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
                frames = SerialFrames()
                while not stop.is_set():
                    for values in frames.feed(stream.read(4096)):
                        session.receive(values)
        except OSError as exc:
            session.receive(None)
            print(f'串口 {port} 断开，正在重试：{exc}', file=sys.stderr, flush=True)
            stop.wait(2)


class LocalHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _respond(self, payload, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, value):
        self._respond(json.dumps(value, allow_nan=False).encode('utf-8'), 'application/json')

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
        else:
            self.send_error(404)

    def do_POST(self):
        if not self._local_host():
            self.send_error(403)
        elif self.path != '/api/start':
            self.send_error(404)
        elif self.headers.get('Content-Type') != 'application/json':
            self.send_error(415)
        else:
            self._json(self.server.session.start())


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--port', help='Arduino 串口，例如 COM4')
    source.add_argument('--replay', type=Path, help='用已采集的 CSV 按原时间回放')
    parser.add_argument('--baudrate', type=int, default=57600)
    parser.add_argument('--http-port', type=int, default=8765, help='本地网页端口')
    args = parser.parse_args()
    if args.baudrate < 1 or not 0 <= args.http_port <= 65535:
        parser.error('波特率必须大于 0，网页端口必须在 0–65535 之间。')

    stop = threading.Event()
    try:
        scorer, settings = load_model_bundle(args.model)
        timer = FocusTimer(scorer, settings)
        if args.replay:
            with args.replay.open(encoding='utf-8-sig', newline='') as stream:
                rows = read_replay(stream)
            session = DemoSession(timer, 'replay')
            worker = threading.Thread(target=run_replay, args=(session, rows),
                                      kwargs={'wait': stop.wait}, daemon=True)
        else:
            try:
                import serial
            except ImportError:
                parser.error('请先安装 pyserial：python -m pip install -r eeg_collection/data_collecting/requirements.txt')
            session = DemoSession(timer, 'serial')
            worker = threading.Thread(target=run_serial,
                                      args=(session, serial, args.port, args.baudrate, stop),
                                      daemon=True)
        with LocalServer(args.http_port, session) as server:
            worker.start()
            try:
                print(f'请打开 http://127.0.0.1:{server.server_port} （Ctrl+C 退出）', flush=True)
                server.serve_forever()
            finally:
                stop.set()
                worker.join(timeout=3)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as exc:
        parser.exit(2, f'无法启动演示：{exc}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
