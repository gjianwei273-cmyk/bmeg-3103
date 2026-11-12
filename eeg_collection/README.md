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

## 自动搜索 attention 的平滑与阈值

在项目根目录运行，无需安装额外依赖（只用 Python 标准库）：

```powershell
python eeg_collection/tune_attention.py
```

自动读取 `eeg_collection/data/` 的 CSV 与同名 JSON 摘要，仅使用 completed、serial
且 valid=1、training_eligible=1 的数据。模拟、未完成、缺摘要或格式异常的文件会排除并记录原因。
每轮必须包含先放松后专注的两个阶段，至少需要 3 轮；现有 6 轮会执行 6 次“5 轮选参、1 轮验证”。
不打乱相邻样本，不根据阶段标签重置滤波器，不修改采集文件。

默认联合搜索 1,881 组参数：

- 不平滑，以及最近 2–10 个有效样本的移动平均。
- 指数平滑 α=0.1–0.9：`s[t] = α*x[t] + (1-α)*s[t-1]`。
- 每种滤波设置搭配 1–99 的整数阈值；平滑值 ≥ 阈值判断专注。

滤波器在每轮首个有效值初始化；移动平均在窗口未填满时平均已有值。
窗口是样本数，只有接收接近 1 Hz 时才近似为秒；缺失与准备阶段数据不会插值，跨阶段保留滤波状态。

每次在 `eeg_collection/reports/` 创建独立目录，保留之前的结果：

- `report.html`：双击可离线打开。含各方法整轮留出验证、参数取舍散点图、
  阈值/滤波参数热力图，以及可切换轮次和参数的原始/平滑/阈值时间序列。
  点击散点、热力图或前 12 组参数表中的“查看”，可检查相应参数轨迹。
- `candidates.csv`：全部参数在全数据上的探索结果，方便 Excel 打开。
- `folds.csv`：每次留出所选参数、训练轮次、留出指标与固定阈值 50 基准。
- `recommendation.json`：全数据校准候选、网格、协议、留出结果、输入 SHA-256 和排除记录。

### 准确率、响应与稳定性的平衡

默认综合评分（指标以 0–1 计）：

```text
平衡准确率 = (专注识别率 + 放松识别率) / 2
评分 = 平衡准确率 - 0.005 × 延迟惩罚秒数 - 0.002 × 阶段内跳变次数/分钟
```

默认多等 1 秒扣 0.5 个百分点，每分钟多跳变一次扣 0.2 个百分点。
这是可调整的工程权衡；评分相同时优先准确率高、延迟低、跳变少的参数，再按网格顺序确定。
质量指标按轮等权平均；误报、漏报、确认和样本数量直接相加。

响应指标要求指令切换后连续 3 个有效样本判断正确，使用最后一个确认样本相对指令的时间，
观察窗口为 15 秒。这不会额外改变阈值预测，只用于评价。若窗口内未确认，或切换前已经误判为目标状态，
延迟惩罚取完整 15 秒，并分别记录 missed / preexisting；confirmed_delay_s 只统计有效确认，未确认时为空。
阶段内跳变不包含跨指令边界的变化，分钟数只计同阶段相邻有效观测的时间差。

例如更重视响应速度，或缩小搜索范围：

```powershell
python eeg_collection/tune_attention.py --delay-weight 0.01
python eeg_collection/tune_attention.py --windows 2,3,5 --alphas 0.3,0.5,0.7 --thresholds 40,45,50,55,60
python eeg_collection/tune_attention.py --horizon-seconds 10 --confirm-samples 2 --flip-weight 0.003
```

也支持 `--data-dir` 和 `--output-dir`。更改权重或网格后应重新运行，并保留报告里的协议，避免混用结果。

### 怎样判断结果可信

各方法在每次留出时都只用其余轮次选择参数；“自动选择”还会在方法之间选。
`validation` 衡量的是选参流程；全数据推荐配置和 candidates.csv 的拟合成绩不是该固定配置的独立测试结果。
报告会比较自动选参和固定基准，若综合评分没有改善，会明确提示暂不替换基准。

当前数据来自同一天、同一参与者，且只包含放松→专注切换；不能据此保证其他日期、其他人或反向切换效果。
标签来自实验指令，准备阶段没有保存数据，响应时间包含人的反应与设备/接收影响。
下一步应固定参数、新采独立数据验证，避免在同一批数据上反复改评分直到结果好看。

### 验证搜索与图表

```powershell
python -m unittest discover -s eeg_collection -v
node eeg_collection/test_attention_report.mjs eeg_collection/reports/你的报告目录/report.html
```

Python 测试覆盖因果滤波、跨阶段状态、阈值边界、混淆矩阵、连续确认、未确认/提前误判惩罚、
留出隔离、数据过滤、时钟一致性、HTML 数据转义与重复运行不覆盖。
如 PATH 中存在 Node.js，Python 集成测试也会运行图表脚本检查；Node 单独检查实际报告的图表生成与交互逻辑，
不替代浏览器中的视觉布局检查。

## 呼吸专注计时与 CNN 对比

安装训练和演示依赖。训练使用本机 CPU，串口采集和原 Attention 搜索仍可单独使用原来的依赖。

```powershell
python -m pip install -r eeg_collection/requirements-focus.txt
```

### 采集不定长的标注数据

```powershell
python eeg_collection/collect_data.py --port COM4 --subject P01 --phases relax:22,focus:37,relax:19,focus:28,relax:24 --output-dir eeg_collection/data/breath
```

每个阶段用 `标签:秒数`，按给定顺序连续记录。可以传入小数时长、交换顺序或重复切换。
正式开始前准备一次，之后不清串口缓冲。提示出现时开始数呼吸或自然休息。
CSV 沿用原字段，JSON 的 `phase_events` 保存实际提示时间。记录时尽量保持姿势、睁闭眼条件一致。
不传 `--phases` 时仍按原模式采集。

训练可以读现有 CSV 和同名 JSON，也接受每文件一场采集的外部 CSV。外部 CSV 至少需要
`label`、`elapsed_s`（秒）或带时区的 ISO `timestamp`，以及所选输入对应的列。
标签接受 `0/1` 或 `relax/focus`，时长和切换顺序由时间和标签决定。
可选 `valid`、`source`、`session_id`、`subject_id` 和 `poor_signal` 延用同样含义。
导入旧 CSV 时，模型读取实际八频段值，八频段模式不会因 Attention 为 0 被排除。
连续录制有 `phase_events` 时使用提示时间，否则从标签及可用的阶段相对时间推断。
如果采集脚本用 `transition_label=1` 标记过渡且留空二分类标签，过渡帧仍进入实时窗口，
但不会作为训练目标或计时对错的标注区间。
选择跨日留出时，每份记录需要能从 CSV 时间戳或 JSON 开始时间识别采集日期。

### 训练、比较与跨日测试

```powershell
# 先尝试现有同日数据
python eeg_collection/focus_train.py

# 之后改为新采的数呼吸数据，日期换成专门留作测试的采集日
python eeg_collection/focus_train.py --data-dir eeg_collection/data/breath --subject P01 --holdout-day 2026-11-15
```

默认先比较 Attention、八频段、九项组合，再比较 4/8 帧，最后比较 1/2/3 层。需要加测
12 帧用 `--windows 4 8 12`。老师若要求模型只用八频段，用 `--inputs bands`。
每折以整场为单位，训练、早停及触发校准、计分使用不同录制。选模型时比较同一组有模型输出的时段。
`--holdout-day` 会把指定日期排除在全部训练、预处理拟合、早停、调参和模型选择之外。
同日录制的探索成绩不能用于判断跨日表现。旧模式阶段之间的准备空白也无法用于测量连续切换延迟。

输出在 `eeg_collection/reports/focus_*/`。其中：

- `comparison.csv` 是各阶段对照。计时错误率是误计与漏计秒数之和除以评估时长。准确率、F1 与混淆计数基于模型分数阈值 0.5。计时则使用校准后的触发规则。
- `report.json` 保留每折设置、计分覆盖、验证轨迹、两种切换方向的延迟、漏检和无法观测的切换数。不同阶段的可比较时段可能不同。
- `training_curve.csv`、`learning_curve.csv` 和 `curves.png` 展示训练轮数与独立录制数量的影响。数据量曲线用于探索同一验证轮上增加训练录制的效果。
- `model.keras` 和 `model.json` 一起组成部署模型，包含预处理及计时设置。输出路径可用 `--output-dir` 指定空目录。

`trend_ema_reference` 是本代码中的 Attention 趋势平滑参照，水平 EMA 系数 0.5、相邻差分 EMA 系数
0.25，再把两者相加并裁剪到 0–1。它通过同一回放流程重新校准阈值。已有 PDF 的 `trend_b25` 实现
未随该工作区提供，因此报告分别保留这个参照的名称和成绩。

### 启动本地网页

把下方路径换成训练命令打印的模型目录。使用串口前先关闭正在占用它的采集脚本或串口监视器。

```powershell
python eeg_collection/focus_demo.py --model eeg_collection/reports/focus_existing_data_final --port COM4
python eeg_collection/focus_demo.py --model eeg_collection/reports/focus_existing_data_final --replay eeg_collection/data/20261105_183912_45e58863.csv
```

浏览器打开 [本地计时页面](http://127.0.0.1:8765)。实时模式先用串口数据预热，按开始后累计。
回放模式按开始后用录制的时间间隔播放。信号超过间隔限制时暂停计时，恢复后重新填充窗口。
