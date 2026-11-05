# 一轮脑波采集

配合 `demo_code/BMEG3103/Arduino/Arduino.ino` 使用，接收其 57600 波特率九列 CSV 输出。
Python 3.9+；真实采集需要 pyserial。无需修改 Arduino 代码。

在项目根目录执行：

```powershell
python -m pip install -r eeg_collection/requirements.txt
python eeg_collection/collect_data.py
```

按提示选择串口，准备好后按 Enter。每次只执行一轮：准备 5 秒 → 放松 60 秒 → 准备 5 秒 → 专注 60 秒。
放松时自然呼吸；专注时保持注视固定目标、持续做心算，尽量少动。阶段开始有终端响铃（是否响取决于终端设置）和中文文字提示，每秒显示剩余时间。
关闭 Arduino 串口监视器和其他串口工具，以免占用端口。

也可指定串口和参与者：

```powershell
python eeg_collection/collect_data.py --list-ports
python eeg_collection/collect_data.py --port COM3 --subject P01 --notes "第一次测试"
```

可调整 `--relax-seconds`、`--focus-seconds`、`--prepare-seconds` 和 `--output-dir`。
默认结果保存在本脚本旁的 `data/`，每次创建独立 CSV 和 JSON 摘要，不覆盖旧结果。
CSV 使用 UTF-8 BOM，便于 Excel 打开。Ctrl+C 会保留已写入的数据，并标记摘要为 interrupted；串口故障标记为 error。
若任一阶段没有有效行则标记 no_data。终端会报告各阶段实际收到的数量，不会补造缺失样本。

## CSV 字段

| 字段 | 含义 |
|---|---|
| session_id / subject_id / round / sample_index | 本次采集编号、参与者编号、轮次（固定 1）、收到的行序号 |
| timestamp | 电脑收到该行的日期时间，含毫秒及 +08:00 时区；不是头环采样时间 |
| elapsed_s / phase_elapsed_s | 距本轮开始 / 当前正式阶段开始的秒数，使用单调时钟计时 |
| phase / label | relax=放松、标签 0；focus=专注、标签 1；来自实验指令，不是根据注意力值猜测 |
| receive_interval_s | 与同阶段上一完整接收行的时间间隔；首行为空；批量收到时可能为 0 |
| delta / theta / low_alpha / high_alpha | Arduino 输出的前四个频段数值 |
| low_beta / high_beta / low_gamma / middle_gamma | Arduino 输出的后四个频段数值 |
| attention | 设备给出的注意力分数（0–100），0 视为不可用，保留但不建议训练 |
| valid / training_eligible | 格式、范围是否有效；可用于训练还要求 attention > 0 |
| error / raw_line | 格式错误原因、原始串口行；异常行的特征字段留空 |
| source / port / baudrate / notes | serial 或 simulation、串口、波特率、备注 |

所有完整的正式阶段行都会保存，包括异常行；准备阶段数据不保存，切换时清空串口缓存和未完成的行。
训练时筛选 `source=serial AND training_eligible=1`，并检查摘要是否 completed、两阶段数据量是否足够。
阶段边缘有设备与串口延迟，训练前可进一步排除每阶段头几秒，避免状态切换的过渡影响。
准备与清缓存期间可能丢弃部分行，后续格式错误会被记录，不会填补或插值。

当前 Arduino 已过滤信号质量不为 0 的数据，但未输出质量字段；缺数据可能是接触不好。
本程序无法恢复原始 EEG 波形、放松度、设备采样时间或被 Arduino 丢弃的数据。
频段数值保留设备原值，不宣称是微伏。标签是要求执行的任务，不保证当时的真实心理状态。
JSON 摘要包含实际有效/异常/可训练行数、时长配置、开始结束时间、模式及完成状态。

## 无硬件试运行与测试

```powershell
python eeg_collection/collect_data.py --simulate --yes --relax-seconds 2 --focus-seconds 2 --prepare-seconds 0
python -m unittest discover -s eeg_collection -v
```

模拟模式无需 pyserial，生成的数据明确标记为 simulation，不能用于真实训练或准确率报告。
自动测试覆盖字段解析、异常数据、一轮标签、时间、CSV 输出、中断保存和无数据状态。
