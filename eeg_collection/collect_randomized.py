"""Record randomized, alternating relax and focus blocks with labeled transitions."""

import argparse
import csv
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from collect_data import FIELDS, FEATURES, SHANGHAI, SimulatedStream, parse_sample, preparation_seconds


LABELS = {'relax': 0, 'focus': 1, 'transition': None}
NAMES = {'relax': '放松', 'focus': '专注', 'transition': '转换'}
INSTRUCTIONS = {'relax': '自然呼吸，放松身体。', 'focus': '注视固定目标，持续做心算。'}
CSV_FIELDS = FIELDS[:FIELDS.index('label') + 1] + ['transition_label'] + FIELDS[FIELDS.index('label') + 1:]
BAUDRATE = 57600


def make_schedule(rng):
    # Every ordered triple is equally likely, including ones with repeated durations.
    triples = [(a, b, 60 - a - b) for a in range(15, 26) for b in range(15, 26)
               if 15 <= 60 - a - b <= 25]
    first, second = ('relax', 'focus') if rng.randrange(2) == 0 else ('focus', 'relax')
    schedule = []
    for first_seconds, second_seconds in zip(rng.choice(triples), rng.choice(triples)):
        if schedule:
            schedule.append(('transition', 5))
        schedule.extend([(first, first_seconds), ('transition', 5), (second, second_seconds)])
    return schedule


def collect(stream, output_dir, schedule, *, subject='P01', notes='', port='', source='serial',
            prepare_seconds=5, clock=time.monotonic, sleep=time.sleep, console=sys.stdout):
    """Timestamp completed lines using the PC clock and keep the stream open across cues."""
    schedule = [dict(phase=phase, duration_s=seconds, label=LABELS[phase],
                     transition_label=int(phase == 'transition')) for phase, seconds in schedule]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(SHANGHAI)
    session_id = started_at.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    csv_path = output_dir / (session_id + '.csv')
    summary_path = output_dir / (session_id + '.json')
    counts = {phase: dict(valid=0, invalid=0, training_eligible=0) for phase in LABELS}
    durations = {phase: sum(step['duration_s'] for step in schedule if step['phase'] == phase)
                 for phase in LABELS}
    events = []
    status, error = 'completed', ''
    missing_samples = False
    pending = b''
    previous = None
    index = 0
    started = clock()

    try:
        with csv_path.open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            handle.flush()
            print(f'正式采集 {sum(durations.values())} 秒，包含转换。Ctrl+C 停止并保存。',
                  file=console, flush=True)
            print(f"首段为{NAMES[schedule[0]['phase']]}，请准备。", file=console, flush=True)
            prepare_end = clock() + prepare_seconds
            while (remaining := prepare_end - clock()) > 0:
                print(f'准备倒计时 {math.ceil(remaining)} 秒', file=console, flush=True)
                sleep(min(1, remaining))
            stream.reset_input_buffer()
            for segment, step in enumerate(schedule):
                phase, duration = step['phase'], step['duration_s']
                transition = step['transition_label']
                name = NAMES[phase]
                instruction = (f"准备切换到{NAMES[schedule[segment + 1]['phase']]}，听到提示再开始。"
                               if transition else INSTRUCTIONS[phase])
                phase_start = clock()
                events.append(dict(step, elapsed_s=phase_start - started))
                print(f'\a开始{name}。{instruction}', file=console, flush=True)
                deadline = phase_start + duration
                last_tick = None
                prior_valid = counts[phase]['valid']
                while (remaining := deadline - clock()) > 0:
                    tick = math.ceil(remaining)
                    if tick != last_tick:
                        count = counts[phase]['valid'] - prior_valid
                        print(f'{name}剩余 {tick} 秒 | 有效数据 {count} 条', file=console, flush=True)
                        last_tick = tick
                    pending += stream.read(4096)
                    received = clock()
                    received_at = datetime.now(SHANGHAI)
                    while b'\n' in pending:
                        raw, pending = pending.split(b'\n', 1)
                        line = raw.rstrip(b'\r').decode('ascii', errors='replace')
                        index += 1
                        row = dict(session_id=session_id, subject_id=subject, round=1,
                                   sample_index=index, timestamp=received_at.isoformat(timespec='milliseconds'),
                                   elapsed_s=f'{received - started:.3f}', phase=phase, label=step['label'],
                                   transition_label=transition, phase_elapsed_s=f'{received - phase_start:.3f}',
                                   receive_interval_s='' if previous is None else f'{received - previous:.3f}',
                                   source=source, port=port, baudrate=BAUDRATE, valid=0,
                                   training_eligible=0, error='', raw_line=line, notes=notes)
                        try:
                            values = parse_sample(line)
                        except ValueError as exc:
                            row['error'] = str(exc)
                            counts[phase]['invalid'] += 1
                        else:
                            row.update(zip(FEATURES, values))
                            row['valid'] = 1
                            row['training_eligible'] = int(not transition and values[-1] > 0)
                            counts[phase]['valid'] += 1
                            counts[phase]['training_eligible'] += row['training_eligible']
                        writer.writerow(row)
                        handle.flush()
                        previous = received
                    if len(pending) > 65536:
                        raise ValueError('串口长时间没有换行，请检查九列 CSV 输出。')
                    sleep(max(0, min(0.05, deadline - clock())))
                missing_samples |= counts[phase]['valid'] == prior_valid
    except KeyboardInterrupt:
        status = 'interrupted'
    except (OSError, ValueError) as exc:
        status, error = 'error', str(exc)

    if status == 'completed' and missing_samples:
        status = 'no_data'
    summary = dict(session_id=session_id, subject_id=subject, notes=notes, source=source,
                   port=port, baudrate=BAUDRATE, status=status, error=error, csv_file=csv_path.name,
                   started_at=started_at.isoformat(timespec='milliseconds'),
                   ended_at=datetime.now(SHANGHAI).isoformat(timespec='milliseconds'),
                   relax_seconds=durations['relax'], focus_seconds=durations['focus'],
                   transition_seconds=durations['transition'], prepare_seconds=prepare_seconds,
                   schedule=schedule, phase_events=events, counts=counts)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'采集状态：{status}\nCSV：{csv_path.resolve()}\n摘要：{summary_path.resolve()}',
          file=console, flush=True)
    if error:
        print(error, file=console, flush=True)
    if status == 'no_data':
        print('至少有一段未收到有效数据，请检查头环和串口。', file=console, flush=True)
    return dict(status=status, csv_path=csv_path, summary_path=summary_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='交替采集放松与专注各 60 秒，记录每次 5 秒转换。')
    parser.add_argument('--port', help='Arduino 串口，如 COM3')
    parser.add_argument('--list-ports', action='store_true', help='列出串口后退出')
    parser.add_argument('--subject', default='P01', help='参与者编号')
    parser.add_argument('--notes', default='', help='本轮备注')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).with_name('data') / 'randomized')
    parser.add_argument('--prepare-seconds', type=preparation_seconds, default=5)
    parser.add_argument('--seed', type=int, help='指定随机种子以复现排程')
    parser.add_argument('--simulate', action='store_true', help='使用模拟数据试运行')
    parser.add_argument('--yes', action='store_true', help='跳过开始前的 Enter 确认')
    args = parser.parse_args(argv)
    stream = None
    try:
        if not args.simulate or args.list_ports:
            try:
                import serial
                from serial.tools import list_ports
            except ImportError:
                parser.error('请先运行 python -m pip install -r eeg_collection/requirements.txt')
            if args.list_ports or not args.port:
                ports = list(list_ports.comports())
                for index, item in enumerate(ports, 1):
                    print(f'{index}. {item.device}: {item.description}')
                if args.list_ports:
                    return 0
                if not ports:
                    parser.error('没有发现串口，请连接 Arduino。')
                choice = input('请输入串口编号：').strip()
                if not choice.isdecimal() or not 1 <= int(choice) <= len(ports):
                    parser.error('串口编号无效。')
                args.port = ports[int(choice) - 1].device
        if args.simulate:
            print('【模拟模式】数据仅用于试运行。')
            stream = SimulatedStream()
        else:
            stream = serial.Serial(args.port, BAUDRATE, timeout=0)
            print('串口已打开，等待 Arduino 重启……')
            time.sleep(2)
        if not args.yes:
            input('戴好头环并关闭其他串口工具，准备好后按 Enter。')
        result = collect(stream, args.output_dir, make_schedule(random.Random(args.seed)),
                         subject=args.subject, notes=args.notes, prepare_seconds=args.prepare_seconds,
                         port='SIMULATED' if args.simulate else args.port,
                         source='simulation' if args.simulate else 'serial')
        return {'completed': 0, 'interrupted': 130}.get(result['status'], 2)
    except KeyboardInterrupt:
        return 130
    except (OSError, EOFError) as exc:
        print(f'无法采集：{exc}', file=sys.stderr)
        return 2
    finally:
        if stream is not None:
            stream.close()


if __name__ == '__main__':
    sys.exit(main())
