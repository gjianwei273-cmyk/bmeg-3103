"""Collect one labelled TGAM round from the existing nine-column Arduino sketch."""
import argparse
import csv
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


FEATURES = ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta',
            'high_beta', 'low_gamma', 'middle_gamma', 'attention']
FIELDS = ['session_id', 'subject_id', 'round', 'sample_index', 'timestamp',
          'elapsed_s', 'phase', 'label', 'phase_elapsed_s', 'receive_interval_s',
          'source', 'port', 'baudrate', 'valid', 'training_eligible', 'error',
          *FEATURES, 'raw_line', 'notes']
SHANGHAI = timezone(timedelta(hours=8))


def parse_sample(line):
    parts = line.strip().split(',')
    if len(parts) != 9:
        raise ValueError('expected_9_columns')
    if any(not part.strip().isascii() or not part.strip().isdigit() for part in parts):
        raise ValueError('expected_unsigned_integers')
    values = [int(part) for part in parts]
    if any(value > 0xFFFFFF for value in values[:8]):
        raise ValueError('power_out_of_range')
    if not 0 <= values[8] <= 100:
        raise ValueError('attention_out_of_range')
    return values


def collect(stream, output_dir, *, subject='P01', notes='', port='', baudrate=57600,
            relax_seconds=60, focus_seconds=60, prepare_seconds=5, source='serial',
            clock=time.monotonic, sleep=time.sleep, console=sys.stdout):
    """Persist every complete active-phase line; injected time/stream allow offline tests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now(SHANGHAI).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    csv_path = output_dir / (session_id + '.csv')
    summary_path = output_dir / (session_id + '.json')
    started_at = datetime.now(SHANGHAI).isoformat(timespec='milliseconds')
    started = clock()
    counts = {phase: {'valid': 0, 'invalid': 0, 'training_eligible': 0}
              for phase in ('relax', 'focus')}
    status = 'completed'
    index = 0
    error = ''
    print('一轮采集：放松 → 专注。保持静止、自然呼吸；Ctrl+C 可停止并保存。', file=console, flush=True)
    try:
        with csv_path.open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            handle.flush()
            for phase, label, duration, instruction in (
                ('relax', 0, relax_seconds, '放松：自然呼吸，放松身体，避免刻意集中注意力。'),
                ('focus', 1, focus_seconds, '专注：保持注视固定目标，持续做心算，尽量避免身体运动。'),
            ):
                print('\n请准备' + instruction, file=console, flush=True)
                deadline = clock() + prepare_seconds
                last_tick = None
                while clock() < deadline:
                    remaining = math.ceil(deadline - clock())
                    if remaining != last_tick:
                        print(f'准备倒计时 {remaining} 秒（此时不保存数据）', file=console, flush=True)
                        last_tick = remaining
                    stream.read(4096)  # Drain old data throughout preparation.
                    sleep(0.05)
                stream.reset_input_buffer()
                pending = b''
                phase_start = clock()
                previous = None
                last_tick = None
                print('\a开始' + instruction, file=console, flush=True)
                while clock() - phase_start < duration:
                    remaining = math.ceil(duration - (clock() - phase_start))
                    if remaining != last_tick:
                        count = counts[phase]['valid']
                        warning = '；暂无有效数据，请检查佩戴/串口' if clock() - phase_start > 3 and count == 0 else ''
                        print(f'{"放松" if label == 0 else "专注"} 剩余 {remaining} 秒 | 有效数据 {count} 条{warning}',
                              file=console, flush=True)
                        last_tick = remaining
                    chunk = stream.read(4096)
                    received = clock()
                    # Do not assign data read after the deadline to the preceding phase.
                    if received - phase_start >= duration:
                        break
                    pending += chunk
                    while b'\n' in pending:
                        raw, pending = pending.split(b'\n', 1)
                        line = raw.rstrip(b'\r').decode('ascii', errors='replace')
                        row = dict(session_id=session_id, subject_id=subject, round=1,
                                   sample_index=index + 1,
                                   timestamp=datetime.now(SHANGHAI).isoformat(timespec='milliseconds'),
                                   elapsed_s=f'{received - started:.3f}', phase=phase, label=label,
                                   phase_elapsed_s=f'{received - phase_start:.3f}',
                                   receive_interval_s='' if previous is None else f'{received - previous:.3f}',
                                   source=source, port=port, baudrate=baudrate,
                                   valid=0, training_eligible=0, error='', raw_line=line, notes=notes)
                        try:
                            values = parse_sample(line)
                            row.update(zip(FEATURES, values))
                            row['valid'] = 1
                            # TGAM attention=0 is unavailable/unreliable, not a relaxed label.
                            row['training_eligible'] = int(values[8] > 0)
                            counts[phase]['valid'] += 1
                            counts[phase]['training_eligible'] += row['training_eligible']
                        except ValueError as exc:
                            row['error'] = str(exc)
                            counts[phase]['invalid'] += 1
                        writer.writerow(row)
                        handle.flush()
                        index += 1
                        previous = received
                    if len(pending) > 65536:
                        raise ValueError('串口长时间没有换行，输入格式不符合九列 CSV 协议。')
                    sleep(0.05)
    except KeyboardInterrupt:
        status = 'interrupted'
    except (OSError, ValueError) as exc:
        status, error = 'error', str(exc)
    if status == 'completed' and any(c['valid'] == 0 for c in counts.values()):
        status = 'no_data'
    summary = dict(session_id=session_id, subject_id=subject, notes=notes,
                   started_at=started_at, ended_at=datetime.now(SHANGHAI).isoformat(timespec='milliseconds'),
                   status=status, error=error, source=source, port=port, baudrate=baudrate,
                   relax_seconds=relax_seconds, focus_seconds=focus_seconds,
                   prepare_seconds=prepare_seconds, counts=counts,
                   timestamp_meaning='PC receive time, Asia/Shanghai; not sensor sampling time',
                   signal_quality='Arduino emits only quality=0 packets; quality is not transmitted',
                   csv_file=csv_path.name)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n采集状态：{status}\nCSV：{csv_path.resolve()}\n摘要：{summary_path.resolve()}', file=console, flush=True)
    print(f'各阶段数量：{counts}', file=console, flush=True)
    if error:
        print(error, file=console, flush=True)
    if status == 'no_data':
        print('至少一个阶段没有有效数据，本轮不能作为完整训练数据。', file=console, flush=True)
    return dict(status=status, csv_path=csv_path, summary_path=summary_path)


class SimulatedStream:
    """Explicitly synthetic data, for trying the workflow without a headset."""
    def __init__(self):
        self.next_sample = time.monotonic()
        self.random = random.Random(3103)

    def read(self, size):
        if time.monotonic() < self.next_sample:
            return b''
        self.next_sample = time.monotonic() + 1
        values = [self.random.randint(1, 500000) for _ in range(8)]
        values.append(self.random.randint(20, 90))
        return (','.join(map(str, values)) + '\n').encode('ascii')

    def reset_input_buffer(self):
        self.next_sample = time.monotonic()

    def close(self):
        pass


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('时长必须是大于 0 的有限数字')
    return number


def preparation_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('准备时间必须是大于等于 0 的有限数字')
    return number


def main():
    parser = argparse.ArgumentParser(description='一轮 EEG 采集：放松 + 专注，自动标注并保存 CSV。')
    parser.add_argument('--port', help='Arduino 串口，如 COM3；省略时交互选择')
    parser.add_argument('--list-ports', action='store_true', help='列出串口后退出')
    parser.add_argument('--subject', default='P01', help='参与者编号')
    parser.add_argument('--notes', default='', help='本轮备注')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--relax-seconds', type=positive_seconds, default=60)
    parser.add_argument('--focus-seconds', type=positive_seconds, default=60)
    parser.add_argument('--prepare-seconds', type=preparation_seconds, default=5)
    parser.add_argument('--simulate', action='store_true', help='使用模拟数据；不能用于训练或报告真实结果')
    parser.add_argument('--yes', action='store_true', help='跳过开始前的 Enter 确认')
    args = parser.parse_args()
    serial = None
    if not args.simulate or args.list_ports:
        try:
            import serial
            from serial.tools import list_ports
        except ImportError:
            parser.exit(2, '请先安装依赖：python -m pip install -r eeg_collection/requirements.txt\n')
        ports = list(list_ports.comports())
        if args.list_ports or not args.port:
            for index, item in enumerate(ports, 1):
                print(f'{index}. {item.device}: {item.description}')
            if args.list_ports:
                return 0
            if not ports:
                parser.exit(2, '没有发现串口，请连接 Arduino 并安装对应驱动。\n')
            choice = input('请输入串口编号：').strip()
            if not choice.isdigit() or not 1 <= int(choice) <= len(ports):
                parser.exit(2, '串口编号无效。\n')
            args.port = ports[int(choice) - 1].device
    stream = None
    try:
        if args.simulate:
            print('【模拟模式】生成的数据仅用于验证操作流程。')
            stream = SimulatedStream()
        else:
            stream = serial.Serial(args.port, 57600, timeout=0)
            print('串口已打开，等待 Arduino 重启……')
            time.sleep(2)
        if not args.yes:
            input('戴好头环并关闭其他串口工具；准备好后按 Enter 开始一轮测试。')
        result = collect(stream, args.output_dir, subject=args.subject, notes=args.notes,
                         port='SIMULATED' if args.simulate else args.port,
                         relax_seconds=args.relax_seconds, focus_seconds=args.focus_seconds,
                         prepare_seconds=args.prepare_seconds,
                         source='simulation' if args.simulate else 'serial')
        return {'completed': 0, 'interrupted': 130}.get(result['status'], 2)
    except KeyboardInterrupt:
        print('\n已取消。')
        return 130
    except (OSError, ValueError) as exc:
        print(f'无法采集：{exc}', file=sys.stderr)
        return 2
    finally:
        if stream is not None:
            stream.close()


if __name__ == '__main__':
    sys.exit(main())
