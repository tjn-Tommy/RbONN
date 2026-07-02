# 基于给定强度初值、OSA 与无梯度优化的通用对称编码方案

## 1. 目标与范围

本方案接收一组给定的15 px对称强度轮廓作为初值，再在真实 Santec SLM 与 OSA 系统上直接优化8个独立强度比例。初值可以来自仿真、人工设计或历史实测结果，但初值的生成方式不属于真机优化流程。

目标是学习一组对所有 channel 共用的修正参数，而不是为每个 channel 单独保存一套编码：

- active width：`N = 15 px`；
- padding：`M = 5 px`；
- channel pitch：`N + M = 20 px`；
- 编码关于中心像素对称，因此只有 `ceil(15/2) = 8` 个独立参数；
- OSA 每次只扫描中心及 `±1、±2` 共五个 channel；
- 使用中心、`-10 channel`、`+10 channel` 三个位置检查空间泛化；
- fidelity 在振幅域评价，串扰和吞吐率在功率域评价。

整体采用交替流程：

```text
给定的8维强度初值
  -> Stage 1：满量程 one-hot 串扰/效率优化
  -> Stage 2：固定轮廓，建立粗振幅 LUT
  -> Stage 3：固定 LUT，小范围优化 modulation fidelity
  -> 必要时再交替一轮
  -> 固定最终轮廓，重建精细 LUT
  -> 全 channel 验证
```

本方案不尝试从 OSA 恢复光学相位。OSA 只提供光谱功率，因此能够优化的是光场振幅的模、带内功率、串扰和吞吐率。

## 2. 变量与物理约定

### 2.1 振幅与功率

全文统一使用：

- `a`：目标归一化振幅，范围 `[0, 1]`；
- `q = a²`：目标归一化功率；
- `l_j`：第 `j` 个像素的归一化强度比例，范围 `[0, 1]`；
- `u`：送入像素级 intensity-to-level LUT 前的标量强度命令，范围 `[0, 1]`；
- `v_j = u * l_j`：第 `j` 个像素实际请求的归一化强度；
- `p_j = sqrt(v_j)`：仅在需要描述光场时使用的等效像素振幅；
- `I(lambda)`：OSA 在线性单位 W 下的光谱功率数据；
- 所有 dBm 数据必须先转成 W，再做背景扣除、平均和积分。

`l_j` 在全文中始终表示强度比例，不表示振幅比例。不能把目标振幅 `a` 直接送入实测 intensity LUT。对固定轮廓 `l`，运行时的映射为：

```text
目标channel振幅 a
    -> channel-level LUT: u = T_l(a)
    -> 像素目标强度: v_j = u * l_j
    -> 像素级intensity LUT: level_j = level_for_j(v_j)
```

在尚未建立 channel-level LUT 时，可用 `u = a²` 作为初始近似。建立 LUT 后，`u = T_l(a)` 一般不必严格等于 `a²`。任何情况下都不直接把 `l_j` 或 `a*l_j` 乘到灰度 level 上。

### 2.2 Channel 区域

active encoding 占 15 px，两个 active encoding 之间有 5 px padding。连续坐标下，每个 channel 的分配区域为：

```text
[channel center - 10 px, channel center + 10 px)
```

真实光谱上优先使用相邻校准中心的中点作为 bin 边界：

```text
B_i = [(lambda_(i-1) + lambda_i)/2,
       (lambda_i + lambda_(i+1))/2)
```

五个20 px channel bins 总宽度约为 `0.5625 nm`，所以 `0.8 nm` OSA span 足够，并能为两端留出余量。

## 3. 编码参数化

### 3.1 给定初始静态轮廓

真机优化器只接收一个8维初值 `l_init`。该接口不读取初值文件或外部模型，也不关心初值来自何种算法；传入值必须已经是归一化强度比例。

优化变量是固定的8维对称强度比例：

```text
l = [l1, l2, l3, l4, l5, l6, l7, l8]
v_j(a, l) = T_l(a) * l_j
```

Stage 1 尚无 channel-level LUT，且只测满量程 one-hot，因此此时 `u=1`，中心 channel 第 `j` 个像素的目标强度就是 `v_j=l_j`。

8个比例镜像为15 px：

```text
[l1, l2, l3, l4, l5, l6, l7, l8,
 l7, l6, l5, l4, l3, l2, l1]
```

`l_init` 是优化任务的必需输入。若调用方希望使用 flat profile，应显式传入：

```text
l_init = [1, 1, 1, 1, 1, 1, 1, 1]
```

优化器不对初值做隐式平方、开方或单位转换，以免混淆振幅与强度。

### 3.2 参数边界

```text
0 <= l_j <= 1
```

## 4. OSA 信号处理

### 4.1 固定测量设置

一次优化运行中保持以下设置不变：

- span：`0.8 nm`；
- sensitivity：固定档位(可调knob)；
- sampling points：固定，不使用运行中变化的点数；
- trace 单位：可以采集 LOG，但必须转成线性 W 后处理；
- SLM 更新后的 settle time：固定；

### 4.2 固定 channel 中心

每个 anchor 在正式搜索前使用 flat pattern 标定五个 channel 的中心与 bin 边界。一次优化运行中，所有候选使用相同边界。一开始可以使用calibration提供的中心，但需要根据实际 OSA trace 调整，确保 OSA 扫描中心在中心 channel 的中心附近。

### 4.3 背景扣除与积分

对 channel bin `B_i`，先做带符号的背景扣除积分，最后再截断到0：

```text
E_i = max(integral_Bi (I_on(lambda) - I_dark(lambda)) dlambda, 0)
```

积分值的绝对单位近似为 W*nm；本方案只使用同一 OSA 设置下的比值，因此不依赖绝对单位。


## 5. Stage 2：串扰轮廓确定后建立粗振幅 LUT

Stage 1 搜索串扰时不建立 channel-level LUT，只使用 calibration 提供的像素级 intensity-to-level LUT 实现满量程静态轮廓。得到串扰最优轮廓 `l_stage1` 后将其冻结，再建立粗振幅 LUT。

### 5.1 LUT 基准环境

使用左右邻居目标振幅均为0.5的环境构建 `l_stage1` 的粗 LUT。选择五个目标振幅采样点：

```text
a_k in {0, 0.25, 0.5, 0.75, 1}
```

首版 channel-level LUT 尚未建立，因此使用平方关系生成初始标量强度命令：

```text
u_k = a_k²
u_k in {0, 0.0625, 0.25, 0.5625, 1}
u_half^(0) = 0.5² = 0.25
```

发送到三个 channel 的 raw intensity commands 为：

```text
[u_left, u_center, u_right] = [u_half, u_k, u_half]
```

每个 channel 内第 `j` 个像素再使用 `v_j = u * l_stage1,j`，并通过该像素的 intensity-to-level calibration 转成 SLM level。这里 `a_k` 是目标振幅，`u_k` 是强度命令，两者不能混用。

对固定轮廓 `l_stage1` 和 anchor `r`，定义：

```text
E_blank = E_center([u_half, 0, u_half])
E_full  = E_center([u_half, 1, u_half])

A_meas(u_k) = sqrt(max((E_center([u_half, u_k, u_half]) - E_blank)
                       / (E_full - E_blank), 0))
```

对五个 `(u_k, A_meas(u_k))` 点做单调包络，再反向插值得到：

```text
target amplitude x -> scalar command u
```

`A_meas` 的上端不强制截断到1，以便暴露超调；只在开平方前对负值截断。

左右邻居第一次使用 raw intensity command `u_half=0.25`。建立首版 LUT 后，令 `u_half=T_stage1(0.5)`，再测一次五点曲线，使“邻居目标振幅为0.5”近似自洽；最多迭代一次，避免追逐噪声。

Stage 3 的一次 COBYQA 运行中固定使用该 LUT：

```text
T_stage1: target amplitude x -> scalar command u
```

LUT 固定端点 `T_stage1(0)=0`、`T_stage1(1)=1`。这里的输出 `u` 始终是归一化强度命令。

不为 Stage 3 的每一个候选重新建 LUT，否则每次评价的测量成本和噪声都会显著增加。


## 6. 测试场景

### 6.1 Stage 3 modulation fidelity 场景

Stage 3 固定使用 `T_stage1`。场景中的所有数值均表示目标振幅。对场景 `[a_left, a_center, a_right]`，三个 channel 都必须在实际显示前转换为强度命令：

```text
u_left   = T_stage1(a_left)
u_center = T_stage1(a_center)
u_right  = T_stage1(a_right)

v_channel,j = u_channel * l_candidate,j
```

中心目标振幅：

```text
x in {0.25, 0.5, 0.75, 1.0}
```

对每个 `x` 测量：

```text
[0.5, x, 0.5]  # LUT基准背景
[0.0, x, 0.0]  # 邻居全关
[1.0, x, 1.0]  # 邻居全开
```

额外测量两个左右非对称场景：

```text
[1.0, 0.5, 0.0]
[0.0, 0.5, 1.0]
```

共14个 modulation 场景。Stage 3 的候选轮廓不同于 `l_stage1`，因此即使场景与 Stage 2 相同，也必须为候选重新测量；只有同一个候选内部完全相同的 pattern 才能复用缓存。

### 6.2 Stage 1 串扰与效率场景

使用中心 one-hot 满量程：

```text
[0, 1, 0]
```

在0和1两个端点，目标振幅与归一化强度命令数值相同，因此该场景也可直接理解为 raw intensity commands。中心 channel 的15 px强度轮廓为 `l`，相邻 channel 的请求强度为0。

输入窗口外的 `±2 channel` 保持关闭，但 OSA 同时积分输出的 `-2、-1、0、+1、+2` 五个20 px bins。

Stage 1 只需要这个场景，因此一个 anchor 每个候选只需一次主要 OSA sweep。Stage 3 中该场景已经包含在 `[0, x, 0]`、`x=1` 的 modulation 场景内，直接复用同一 trace计算串扰与效率。

## 7. 核心指标

### 7.1 振幅编码误差

Stage 3 对所有候选使用 Stage 2 在 `l_stage1`、半开邻居环境下测得的固定 `E_blank_stage1` 和 `E_full_stage1`：

```text
A_hat(r, s) = sqrt(max((E_center(r, s) - E_blank_stage1(r))
                       / (E_full_stage1(r) - E_blank_stage1(r)), 0))
```

固定参考可以让一次 COBYQA 运行中的 objective 保持不变，并让轮廓变化造成的增益或衰减真实反映到 fidelity 中。Stage 3 结束后会为新轮廓重建最终 LUT，因此这里不为每个候选重新归一化满量程。

目标中心振幅为 `x_s`，anchor `r` 的 RMSE 为：

```text
RMSE_a(r) = sqrt(mean_s((A_hat(r, s) - x_s)**2))
```

分母使用场景总数，不使用 `N_case - 1`。这里计算的是确定性预测误差，不是样本方差的无偏估计。

同时保存：

- bias；
- MAE；
- P95 absolute error；
- max absolute error；
- 按邻居状态分组的 RMSE。

### 7.2 串扰

中心 one-hot 场景下：

```text
C_i(r) = E_i(r) / E_0(r), i in {-2, -1, +1, +2}
```

默认总串扰：

```text
C_total(r) = C_-2 + C_-1 + C_+1 + C_+2
```

同时报告：

```text
C_nearest = C_-1 + C_+1
C_next    = C_-2 + C_+2
XT_i_dB   = 10*log10(C_i)
```

FWHM 只作为诊断指标，不参与串扰计算。

### 7.3 强度效率

使用同一 anchor、相同固定 bin 和 one-hot 满量程场景：

```text
eta(r) = E_0_candidate(r) / E_0_flat(r)
```

最终验收要求每个 anchor 都满足：

```text
eta(r) >= 0.85
```

优化时使用 `0.87` 作为带安全余量的软阈值。

## 8. 损失函数

### 8.1 效率惩罚项

```text
L_eta(r) = (max(0, 0.87 - eta(r))/0.02)**2
```

优化阈值使用0.87，最终验收使用0.85。这样给OSA噪声和漂移保留约2个百分点的安全余量。

### 8.2 Stage 1：串扰优先目标

在搜索前测量给定的初始强度轮廓 `l_init`：

```text
C_init(r)
eta_init(r)
```

以串扰为唯一主要性能目标：

```text
C_norm(r) = C_total(r) / max(C_init(r), C_floor)

L_stage1 = mean_r(C_norm(r))
           + beta_eta * mean_r(L_eta(r))
```

初始经验值：

```text
beta_eta      = 1.0
```

Stage 1 不包含 modulation RMSE，也不构建 channel-level LUT。效率惩罚防止优化器通过整体压低轮廓来虚假降低串扰。

Stage 1 不加入平滑项、初值先验项或其他形状正则，损失函数严格使用以上两项。

### 8.3 Stage 3：固定 LUT 后的 fidelity 目标

Stage 2 建立 `T_stage1` 后，先用 `l_stage1 + T_stage1` 测得：

```text
RMSE_stage1(r)
C_stage1(r)
```

Stage 3 主要降低振幅编码误差，同时限制串扰不能明显回退：

```text
E_norm(r) = RMSE_a(r) / max(RMSE_stage1(r), sigma_A)

L_crosstalk_guard(r) =
    (max(0, C_total(r)/max(C_stage1(r), C_floor) - 1.05)/0.05)**2

L_stage3 = mean_r(E_norm(r))
           + beta_c * mean_r(L_crosstalk_guard(r))
           + beta_eta * mean_r(L_eta(r))
```

初始经验值：

```text
beta_c        = 1.0
beta_eta      = 1.0
```

`1.05` 表示 Stage 3 最多允许串扰相对 Stage 1 结果回退约5%。如果串扰是绝对优先指标，可把该上限改成1.00或使用显式约束。

Stage 3 同样不加入平滑项或初值先验项，损失函数严格使用 fidelity、串扰回退惩罚和效率惩罚三项。

### 8.4 最终可行性过滤

无论优化器返回什么，最终候选必须满足：

- 三个 anchor 各自 `eta >= 0.85`；
- held-out channel 的振幅 RMSE 不高于给定初始轮廓的允许上限；
- 改善量大于重复测量噪声的两倍；
- 不接受仅靠某一个 anchor 大幅改善、其他位置明显退化的候选。

## 9. 优化与复测流程

### 阶段0：仪器准备

1. 光源、OSA、SLM 预热并稳定。
2. 加载最新 calibration，确认所选三个 anchor 都位于可靠校准区域。
3. 固定 OSA RBW、span、points、sensitivity 和 averaging。
4. 接收并校验外部传入的8维强度初值 `l_init`，要求形状为 `(8,)`、全部为有限数且位于 `[0,1]`；不读取任何模型或 checkpoint。
5. 对每个 anchor 测 dark、flat reference 和 `l_init` reference。
6. `l_init` 的 one-hot 场景重复测量至少10次，估计串扰和效率噪声以及 `C_floor`。

### 阶段1A：单 anchor 串扰搜索

先只在光谱中心附近的代表性 anchor 使用 one-hot 满量程场景优化 `L_stage1`。本阶段不建立 channel-level LUT，也不测 modulation cases。

COBYQA 初始配置：

```text
x0                  = l_init
bounds              = [(0.0, 1.0)] * 8
maxfev              = 150-250
initial_tr_radius   = 0.10-0.15
final_tr_radius     = 0.01
scale               = True
```

上述 bounds 和 trust-region radius 都以归一化强度为单位。

每个候选只需：

- 通过像素级 LUT 生成 `[0,1,0]` 静态轮廓；
- 一次 OSA sweep 同时得到 `C_total` 和 `eta`；
- dark 和 flat reference 按固定周期复测，不为每个候选重复。

200次评价约为200次主要 sweep，不含周期性 reference，因此 Stage 1 可以使用较充分的 COBYQA 搜索预算。

运行过程中保存所有候选，而不是只保存 COBYQA 最后的结果。

### 阶段1B：多 anchor 串扰重排

从阶段1A选择目标值最好的10-20个候选，在：

```text
center channel
center - 10 channels
center + 10 channels
```

三个 anchor 上测量 one-hot 串扰和效率。每个候选至少重复3次，并按三处综合指标重新排序，得到 `l_stage1`。

### 阶段2：固定 `l_stage1` 建立粗 LUT

在三个 anchor 上固定 `l_stage1`，测量：

```text
target amplitudes: [0.5, a_k, 0.5]
a_k in {0, 0.25, 0.5, 0.75, 1}

initial raw intensity commands:
[0.25, a_k², 0.25]
```

生成每个 anchor 的 `T_stage1`。若需要更稳定的插值，可改用9点。完成 LUT 后重复测量 `l_stage1 + T_stage1` 的14个 modulation cases，得到：

```text
RMSE_stage1(r)
C_stage1(r)
sigma_A
```

这些值是 Stage 3 的固定基准。

### 阶段3A：固定 LUT 的局部 fidelity 搜索

固定 `T_stage1`，从 `l_stage1` 启动 COBYQA，优化 `L_stage3`。参数只允许在 Stage 1 结果附近小范围变化：

```text
x0 = l_stage1
bounds_j = [max(0, l_stage1,j - 0.10),
            min(1, l_stage1,j + 0.10)]
initial_tr_radius = 0.05
final_tr_radius   = 0.01
maxfev            = 80-150
```

这里的 `±0.10` 和 trust-region radius 都表示归一化强度的绝对变化，不是振幅变化。

每个候选测量14个 modulation cases，其中 `[0,1,0]` 同时提供串扰和效率。100次评价约为1400次主要 sweep。

### 阶段3B：多 anchor fidelity 重排

选择 Stage 3A 最好的5个候选，在中心和 `±10 channel` 三个 anchor 上完整复测，每个候选至少重复3次。使用 Stage 3 的 fidelity、串扰回退限制和效率约束重新排序，得到 `l_stage3`。


### 阶段4：交替检查与离散精修

1. 检查 `max(abs(l_stage3 - l_stage1))`，差值单位为归一化强度。
2. 若强度绝对变化不超过0.03，直接进入最终 LUT。
3. 若强度绝对变化超过0.03，为 `l_stage3` 重建五点粗 LUT，并允许再进行一次更短的局部 fidelity 搜索。
4. 最多执行两轮“粗 LUT -> 局部轮廓优化”，避免追逐 OSA 噪声。
5. 在最终精细 LUT 之前，对8个比例做能引起实际 pattern 变化的小范围离散精修。
6. 对产生完全相同 SLM pattern 的候选使用缓存，不重复测量。

任何轮廓变化都会使旧 LUT 失效，因此离散精修之后必须再建最终 LUT。

### 阶段5：建立最终精细 LUT

1. 固定最终轮廓 `l_final`，不再修改8个比例。
2. 在每个验证位置测量11-21点传递曲线。
3. 生成最终单调逆 LUT `T_final`。
4. 使用 `l_final + T_final` 重新测量 modulation、串扰和效率。

### 阶段6：全系统验证

在全部可用 channel 上，以随机交错顺序比较：

```text
flat + 自己的LUT
给定初始强度轮廓 + 自己的LUT
实机优化静态轮廓 + 自己的LUT
```

至少执行：

- 所有 channel 的 one-hot 串扰扫描；
- 不参与优化的 channel 位置；
- 随机多通道目标；
- 不同时间段重复测量；
- 平均值、P95和最差 channel 统计。



## 10. 数据与恢复设计

每次夜间运行创建独立目录，例如：

```text
data/osa_optimization/2026-07-01_run01/
```

保存：

```text
run_config.json        # OSA、SLM、calibration、目标函数和优化器配置
baseline.json          # flat和给定初始强度轮廓基线
candidates.csv         # 每次评价的l、指标、时间戳、状态
best_so_far.json       # 当前最佳可行候选
traces/                # 原始OSA trace，建议npz压缩
patterns/              # 唯一SLM pattern及hash
references/            # dark/flat/初始轮廓漂移参考
optimizer_state.json   # 可恢复的搜索记录或重启信息
```

每个 candidate 至少记录：

- 8维 `l`；
- 四个目标振幅点对应的15 px目标强度 `v_j=T_l(a)*l_j`；
- 实际15列 SLM levels；
- pattern hash；
- 每个 anchor 的 RMSE、bias、crosstalk、eta；
- dark/reference 版本；
- OSA settings；
- sweep 起止时间；
- 是否满足最终约束。

优化进程中每完成一个候选就落盘，保证仪器或程序中断后能够从已有候选继续，而不是丢失整夜数据。

## 11. 与现有代码的建议接口

可以复用：

- `OSAController`：配置、单次 sweep、trace 下载和线性域平均；
- `SLMController`：发送完整 pattern；
- `CalibrationResult` / `EncodingChannel`：坐标、波长和 intensity-level 数据；
- `analysis.py` 的 trace 数据结构和 CSV/NPZ 保存思路。

真机优化入口直接接收 `l_init: array[8]`。初值生成、模型加载及振幅到强度的预处理均由调用方负责，不属于本模块职责。

优化 objective 不应直接调用当前 `_channel_metrics()`，因为当前 modulation-error 逻辑仍包含与本方案不同的积分和归一化定义。建议把新逻辑放在独立模块，例如：

```text
src/slm_module/optimization.py
```

建议的职责划分：

```text
OptimizationConfig
    - 几何、anchors、场景、OSA参数、权重、阈值

FixedChannelBins
    - 固定中心和中点边界

CandidateEncoder
    - 固定8维强度比例 + 对称镜像 + local LUT；运行时不依赖外部模型

OSAEvaluator
    - dark subtraction、bin积分、粗LUT、场景测量、缓存

CandidateMetrics
    - amplitude RMSE、crosstalk、eta、噪声和约束

OptimizationRunner
    - COBYQA、checkpoint、resume、top-K复测
```

`optimize_from_osa()` 可以作为 GUI 入口，但不应承担所有采集、指标和持久化逻辑。

## 12. 验收标准

最终方案至少满足：

1. 三个训练 anchor 和所有 held-out anchor 的 `eta >= 0.85`；
2. 实机优化静态轮廓的振幅 RMSE 不劣于给定初始轮廓，或满足预先定义的最大退化比例；
3. 最近邻与 `±2` 串扰相对给定初始轮廓有超过 `2 sigma` 的稳定改善；
4. 全 channel 的最差值没有被平均值掩盖；
5. 在独立时间段重复测量仍保持同方向改善；
6. 最终结果同时报告振幅 fidelity、功率串扰、吞吐率和 FWHM，不用单一指标代替全部性能。

## 13. 关键禁止事项

- 不直接对 OSA 每个采样点开平方后积分；应先积分功率，再开平方得到归一化振幅。
- 不把目标振幅、强度比例或任何比例直接乘 SLM 灰度 level；必须先计算目标强度 `v_j=u*l_j`，再查询像素级 intensity LUT。
- 不把 `l_j` 当成振幅比例；`l_j` 在输入、优化、保存和边界判断中始终表示归一化强度。
- 不混淆目标振幅 `a` 和标量强度命令 `u=T_l(a)`。
- 不使用候选自身的 peak 移动 channel bins。
- 不用 FWHM 代替串扰积分。
- 不把邻居本身点亮后的全部邻道功率解释成串扰；串扰必须使用 one-hot 场景。
- 不使用 `beta*(1-eta)` 表示85%阈值；必须使用 hinge penalty 或显式约束。
- 不在同一次 COBYQA 运行中改变 LUT 点数、测试场景、OSA 设置或损失定义。
- 不只根据优化器最后一点选结果；必须对历史 top-K 候选重新测量。
