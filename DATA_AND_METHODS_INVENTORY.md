# 数据与方法清单

由 `scratchpad/make_inventory.py` 从产物本身生成,不是手写的。
对应运行:`data\outputs\v13_runs\20260808T2100Z_colobus_bouts`

---

## 1. 用了哪些数据

扩增前 **30,651** 个片段,扩增后 **37,893** 行。

### 按来源

| 行数 | 类别 | 来源标签 | 人工审核 | 磁盘路径 |
|---:|---|---|---|---|
| 16,826 | Background | `birdnet:<每个鸟种一个文件夹>` | 否 | `D:/Gabon BirdNET segments Birds/<鸟种>` |
| 3,143 | Background | `auto_flagged_fp` | 否 | `data/outputs/auto_cleanup/auto_flagged_fp 等 8 个目录` |
| 2,535 | Cernic | `review:confirmed_call` | 是 | `data/outputs/detected_clips/Cernic/IPA10ST/20210222 等 61 个目录` |
| 2,370 | Background | `review:ipa4st_intruder` | 是 | `data/outputs/detected_clips/Cernic/IPA4ST/20210222 等 4 个目录` |
| 1,401 | Background | `reference:background noise Clips 5sec` | 是 | `data/background/background noise Clips 5sec` |
| 1,284 | Background | `review:false_positive` | 是 | `data/outputs/detected_clips/Cernic/IPA10ST/20210223 等 58 个目录` |
| 701 | Colobus_guereza | `reference:Colobus guereza bouts` | 是 | `data/species/Colobus guereza bouts` |
| 654 | Colobus_confuser | `reference:Colobus_confuser` | 是 | `data/species/Colobus_confuser` |
| 342 | Background | `reference:Pan troglodytes Clips 5sec` | 是 | `data/background/Pan troglodytes Clips 5sec` |
| 317 | Background | `auto_flagged_fp:confirmed_fp` | 是 | `data/outputs/auto_cleanup/auto_flagged_fp/mahal 等 2 个目录` |
| 253 | Colobus_confuser | `colobus_field_fp` | 是 | `data/outputs/detected_clips/Colobus_guereza/IPA10ST/20210224 等 25 个目录` |
| 188 | Background | `reference:Cercocebus torquatus Clips 5s` | 是 | `data/background/Cercocebus torquatus Clips 5s` |
| 150 | C_pogonias | `reference:C_pogonias` | 是 | `data/species/C_pogonias` |
| 124 | Cernic | `reference:CERNIC keks` | 是 | `data/species/CERNIC keks` |
| 101 | Cernic | `reference:CERNIC hacks` | 是 | `data/species/CERNIC hacks` |
| 101 | Cernic | `reference:CERNIC pyows` | 是 | `data/species/CERNIC pyows` |
| 70 | Cernic | `auto_flagged_fp:RECOVERED_CALL` | 是 | `data/outputs/auto_cleanup/auto_flagged_fp/mahal 等 2 个目录` |
| 27 | Cernic | `reference:CERNIC putty-nose 5s` | 是 | `data/species/CERNIC putty-nose 5s` |
| 25 | Cernic | `reference:CERNIC putty-nose 2s` | 是 | `data/species/CERNIC putty-nose 2s` |
| 20 | Background | `reference:wrong classified` | 是 | `data/background/wrong classified` |
| 19 | Cernic | `reference:CERNIC field_confirmed` | 是 | `data/species/CERNIC field_confirmed` |

### 按类别,以及能不能被 LOSO 评估

| 类别 | 扩增前 | 扩增后 | 有站点归属 | 人工审核的 detection | 能被 LOSO 打分 |
|---|---:|---:|---:|---:|---|
| Background | 25,891 | 25,891 | 9,166 | 3,654 | **能** |
| C_pogonias | 150 | 3,000 | 0 | 0 | **不能** |
| Cernic | 3,002 | 3,002 | 2,618 | 2,535 | **能** |
| Colobus_confuser | 907 | 3,000 | 2,967 | 0 | **不能** |
| Colobus_guereza | 701 | 3,000 | 0 | 0 | **不能** |

**LOSO 评估池共 6,189 行**:Background 3,654,Cernic 2,535。

> Colobus_guereza 和 C_pogonias 在 16 折验证里**一行都没有被打过分**。表里的 precision 是纯 Cernic 的数字。

## 2. 用了哪些方法

| 环节 | 取值 / 做法 |
|---|---|
| 采样率 | 44100 Hz |
| 分析窗 | 2.0 s |
| Mel 频带 | 128,20–8000 Hz |
| 输入图像 | 224×224×3 |
| 主干 | VGG19 ImageNet 预训练,**冻结**,取 `block4_conv4` |
| 头部 | 频率坐标(CoordConv)通道 → 4 个频带各自 Conv1D → 合并 → 跨带 Conv1D → BiLSTM |
| 短片段填充 | 嵌入真实环境底噪,SNR (-6.0, 9.0) dB(**不是补零**) |
| 扩增:背景混合 | SNR (-5, 10) dB |
| 扩增:时间/频率裁剪 | 各裁 5%–10% |
| 扩增:频率平移 | ±9 mel(128 行的 7%) |
| 扩增策略 | 按**目标数 3,000**,不是固定倍数(Sun et al.) |
| 扩增范围 | 只对目标类;Background 不扩增 |
| 类别权重 | sklearn `balanced` |
| 交叉验证 | 16 折留一站点(LOSO) |
| 切分单位 | 按**源录音**分组,副本与原件同侧 |
| 阈值 | 在另外 15 个站点上拟合到 95% 召回,**不是 oracle** |
| 检测分组 | {'Cernic': 'Cernic', 'Colobus_guereza': 'Colobus_guereza', 'Colobus_confuser': 'Background', 'C_pogonias': 'C_pogonias', 'Background': 'Background'} |
| 检测阈值(默认) | 0.4 |
| NMS IoU | 0.5 |
| 低频门控 | 开启,1500 Hz 以下能量占比 ≥ 0.4,**只作用于 Colobus_guereza** |
| 部署时间窗 | 05:00–19:00 |
| 负样本筛查 | 模型分数 ≥0.5 的剪掉,**但只剪机器标注且明确未审核的行** |

## 3. 没有用的方法(以及为什么)

| 方法 | 状态 | 原因 |
|---|---|---|
| 降噪(spectral gating) | **弃用** | 两次测量 roar 频带能量占比都下降(0.629→0.559、0.487→0.415),说明门控把 call 本身削掉了 |
| `background/random_forest` 背景源 | **撤回** | 用 4 类模型对 5 类配置读取导致误判,已在 config 里注释掉 |
| 补零填充短片段 | **替换** | 会让全短片段的类靠那段数字静音被识别,已改为嵌入真实底噪 |
| 未分组的随机切分 | **替换** | 同一录音的片段会同时出现在训练和验证两侧,模拟出 79% 泄漏 |
| oracle 阈值 | **替换** | 在留出站点自己的答案上选阈值,新站点无法复现 |
| 275 个 BirdNET 高分片段 | **不用** | 未经人工审核,你明确要求不要 |
| Colobus 高频干扰扩增(V12) | 未在本轮启用 | 本轮用的是按目标数扩增;该方法仍在论文里作为 V12 记载 |
| GPU 训练 | **不可用** | TensorFlow 2.20 在原生 Windows 上没有 GPU 支持,全部 CPU |

## 4. 结果:两个模型,只差 Colobus 的切分方式

| 指标(gated = 部署时间窗,每站点计一次) | 单脉冲版 | bout 版 |
|---|---:|---:|
| v12 基线 precision | 0.6953 | 0.6953 |
| v13 precision | 0.9209 | 0.9161 |
| 保留的 call | 95.0% | 91.2% |
| 移除的假阳 | 78.6% | 78.9% |
| 拟合阈值中位 | 0.944 | 0.954 |
| **9 个野外确认 clip 触发数**(单脉冲版) | **2/9** | |
| **9 个野外确认 clip 触发数**(bout 版) |  | **1/9** |

> 这张表的前五行**只测 Cernic** —— LOSO 评估池里 Colobus 和 C_pogonias 一行都没有。
> 唯一测到 Colobus 的是最后两行,而它显示 bout 版的野外灵敏度**更差**。
> `data/outputs/field_control_scores.csv` 是逐个 clip 的分数。

## 5. 已知的开口(需要你判断的)

1. **16,826 个 BirdNET 背景片段没有人听过。** 只用模型分数筛过 —— 模型给自己打分。
2. **Colobus 和 C_pogonias 没有任何留出验证。** 只有阳性对照和你的听感。
3. **precision 测的是重排序,不是扫描。** 只有人工听 detection 能测后者。
4. **阈值不稳定**(0.044–0.995),单一常数不可迁移。
5. **低频门控 0.40 是用旧模型标定的**,新模型没重标;它在 IPA4ST 上删掉了全部 7 个 Colobus 检测。
6. **Colobus 训练数据是档案录音,部署是野外录音。**
