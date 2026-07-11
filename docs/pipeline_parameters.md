# RbONN 校准 Pipeline 参数完全指南

本文档说明 `src/slm_module/pipeline.py` 中统一校准流水线的**每一个参数**：它的物理含义、取值范围、默认值、对测量速度/精度的影响，以及每个阶段的**算法流程**与**数值运行例子**。

代码入口是 [`run_pipeline`](../src/slm_module/pipeline.py)，GUI 入口是 [`pipeline_page.py`](../src/slm_module/gui/pipeline_page.py)。

---

## 1. 总体架构

流水线把整条校准链拆成 5 个 **stage**，按固定顺序执行：

```
wl_map (Step 1+2)  →  intensity (Step 3)  →  tpa_center  →  pair_eta (Step 6)  →  comb_phase (Step 7)
   ↑ OSA+SLM           ↑ OSA+SLM              ↑ monitor+SLM   ↑ monitor+SLM         ↑ monitor+SLM
   产出 wl_map          产出 intensity_calib   产出 center_fit  产出 pair_etas        产出 comb_phase
```

### 1.1 stage 之间怎么传数据

每个 stage 声明自己需要的 **输入 artifact**（`requires` 必需 / `optional` 可选）和产出的 artifact（`produces`）。输入来源有两种（`InputSpec.source`）：

| 来源 | 含义 |
|------|------|
| `"memory"` | 用**同一次运行里前面某个已启用 stage** 刚产出的结果（在内存里，不落盘再读） |
| `"file"` | 从磁盘 JSON/CSV 加载（`path` 必填），用于**只跑后半段**、复用历史校准 |

阶段依赖关系（`STAGES` 表，[pipeline.py:117](../src/slm_module/pipeline.py#L117)）：

| stage_id | 需要输入 | 可选输入 | 产出 | 用到的仪器 |
|----------|----------|----------|------|-----------|
| `wl_map` | — | — | `wl_map` | osa, slm |
| `intensity` | `wl_map` | — | `intensity_calib` | osa, slm |
| `tpa_center` | `intensity_calib` | — | `center_fit` | monitor, slm |
| `pair_eta` | `intensity_calib` | `center_fit` | `pair_etas` | monitor, slm |
| `comb_phase` | `intensity_calib`, `pair_etas` | `center_fit` | `comb_phase` | monitor, slm |

### 1.2 两个重要保证

1. **输出即时落盘**：每个 stage 一算完就写自己的输出文件。后面任何 stage 失败或被中止，前面完成的结果都已在磁盘上（`PipelineAborted` / `PipelineStageError` 都带 `saved_files`）。
2. **顺序 & 校验前置**：`validate_request` 在开跑前一次性检查阶段顺序、输入是否有生产者、文件是否存在、输出路径是否冲突、输出目录是否存在 —— 避免跑到一半才因为一个路径打错而丢失几十分钟的测量。

---

## 2. 全局共享配置

这几组参数不属于单个 stage，而是被多个 stage 共享。

### 2.1 `LayoutConfig` —— 通道几何（[pipeline.py:153](../src/slm_module/pipeline.py#L153)）

描述在 778 nm 附近怎么在 SLM 上摆放对称的 x/w 通道对。**measurement 和 encoding 两边都用它**，所以两边永远一致。

| 参数 | 默认 | 含义 |
|------|------|------|
| `n_channels` | 20 | **每侧**通道数（x 侧 20 + w 侧 20 = 20 个 pair） |
| `channel_width_px` | 15 | 每个通道点亮窗口宽度（像素） |
| `gap_px` | 5 | 相邻通道之间的暗间隔（像素）。pitch = width + gap = 20 px |
| `center_gap_px` | None | 中心暗区宽度；None = 传统半 pitch 起始。设了就把最靠中心的通道往外推，给 778 nm 留更宽的暗带 |
| `center_wl` | 778.0 | 中心波长（nm），对称轴 |
| `guard_bands` | `((780,0.1),(776,0.1))` | Rb 吸收线保护带 `(中心nm, 半宽nm)`，落在带内的通道会被跳过 |

> **几何细节**：通道摆在 `center ± (k+0.5)·pitch`。第 k 个左/右通道要能配成波长对称的 pair，`guard_bands` 必须关于 `center_wl` 镜像对称，否则 `build_channel_calibration_grid` 直接报错（`require_symmetric_guard_bands`）。默认的 780/776 正好关于 778 对称。

### 2.2 `OSAStageSettings` —— OSA 扫描参数（[pipeline.py:190](../src/slm_module/pipeline.py#L190)）

`wl_map` 和 `intensity` 各自带一份，转成 `MeasurementSettings` 下发给 OSA。

| 参数 | 默认 | 含义 |
|------|------|------|
| `center_wl` | `"778nm"` | OSA 扫描中心波长 |
| `span` | `"8nm"` | 扫描跨度。要能覆盖整个通道排布 + 保护带 |
| `sensitivity` | `"HIGH2"` | OSA 灵敏度档位（越高越慢越准） |
| `sampling_points` | `"AUTO"` | 采样点数；AUTO 让 OSA 按 span 自动定 |
| `reference_level` | `"10uW"` | 参考电平（Y 轴量程），线性 `y_unit="LINear"` |

### 2.3 `PipelineRequest` 顶层字段（[pipeline.py:303](../src/slm_module/pipeline.py#L303)）

| 参数 | 默认 | 含义 |
|------|------|------|
| `stages` | — | `StagePlan` 列表，实际要跑哪些 stage、各自的 config/输入/输出 |
| `layout` | `LayoutConfig()` | 上面的通道几何 |
| `col_ratio` | None | 每列的编码 taper（幅度整形）。TPA 三个 stage 编码 pattern 时透传，保证校准时的通道形状 = 部署时的形状。None = 平顶带 |
| `use_center_fit` | True | 是否把有效的 `tpa_center` 拟合结果喂给下游做 layout 中心。见下 |

> **`use_center_fit` 的作用**（[pipeline.py:511](../src/slm_module/pipeline.py#L511) `ctx.center_wl`）：`pair_eta` 和 `comb_phase` 建 layout 时要一个中心波长。如果 `use_center_fit=True` 且提供了一个 `valid` 的 `center_fit`，就用测出来的 TPA 共振中心（`fit.center_wl_nm`）；否则退回 `layout.center_wl`（778）。这让整个 x/w 排布对齐到**真正的双光子共振**而不是名义中心。

### 2.4 `OutlierRemeasurePolicy` —— 离群点自动复测（[outliers.py:26](../src/slm_module/calibration/outliers.py#L26)）

`wl_map` 和 `intensity` 可选启用。扫完一遍后，把数据拟合到理论模型（Step 2 是线性 coord→λ 映射，Step 3 是 sin² 透过曲线），残差超过 `k_sigma` 个稳健标准差（基于 MAD，离群点自己不会抬高阈值）的点被标记、重新点亮、重测，该点取多次重测的**中位数**。

| 参数 | 默认 | 含义 |
|------|------|------|
| `k_sigma` | 4.0 | 残差阈值，单位是稳健 sigma（MAD×1.4826）。越小越激进 |
| `max_retries` | 3 | 每遍最多复测轮数 |
| `min_points` | 8 | 少于这么多点就不敢拟合，跳过复测 |

None（默认）= 关闭。

---

## 3. Stage 1+2 —— `wl_map`（`WlMapConfig`）

**目的**：① 全屏灰度扫一遍找出输出功率的 min/max 灰度（Step 1）；② 用一个亮窗在 SLM 上横向滑动，每个位置测 OSA 峰值波长，建立 **SLM x 坐标 → 波长** 的映射（Step 2）。

### 3.1 参数（[pipeline.py:211](../src/slm_module/pipeline.py#L211)）

| 参数 | 默认 | 含义 |
|------|------|------|
| `levels` | `range(0,1024,64)` = `[0,64,...,960]` | Step 1 全屏扫的灰度档位，用来找 min/max level |
| `window_size` | 8 | Step 2 亮窗宽度（像素）。窗越宽信号越强但波长分辨率越粗 |
| `coordinate_stride` | 1 | **Step 2 提速**：每隔 N 列才测一次，其余列由拟合曲线补出（见 §3.3） |
| `peak_half_window_nm` | None | 峰值质心的 ± 波长窗（nm）。给了就按物理 nm 宽度取质心，与 OSA 采样密度无关；None 用固定样本数 |
| `region` | None | `(x_start, x_end)` 只扫这段 SLM 列。光源只照亮部分孔径时有用（如 6 nm 脉冲打在 20 nm 孔径上）。None = 全宽 |
| `osa` | `OSAStageSettings()` | OSA 扫描参数 |
| `outlier_policy` | None | 离群点复测（§2.4） |

### 3.2 算法流程

1. **Step 1**（`find_min_max_intensity_levels`）：对 `levels` 里每个灰度，全屏显示 → OSA 测一条 trace → 取平均功率。记录功率最小/最大对应的灰度 = `min_level` / `max_level`。
2. **参考帧**：全屏 `min_level` 测暗背景 trace；全屏 `max_level` 测亮参考 trace。
3. **Step 2 扫描**（`wavelength_calibration`）：亮窗从 `region_lo` 滑到 `region_hi`，每个窗位置：暗底 + 窗内 `max_level` → OSA 测 trace → 背景扣除 + 归一化 → `local_peak_centroid` 质心定峰得波长 → 记 `(coordinate=x_start+window//2, wavelength)`。
4. **（可选）离群复测** → **多项式拟合**：`_fit_wavelength_mapping` 用**三次多项式**（点少时降阶）拟合 coord→λ，返回拟合后的波长与系数。

### 3.3 `coordinate_stride` 提速原理（本次新增）

coord→λ 映射几乎线性，而拟合用的是三次多项式，所以**没必要逐列测**。`stride=N` 时只测第 0, N, 2N… 列，并**始终追加区域最后一列作为锚点**（保证拟合覆盖整个区间、补全是内插而非外推）。拟合出系数后，在区域内**每一列**求值，返回的稠密网格与 `stride=1` 完全同构，下游无感知。

- 加速比 ≈ stride（OSA 每次测量是主要开销）。
- 建议：保证至少剩 10–20 个实测点，三次拟合才稳、离群剔除才有统计意义。例：1200 px 区域，stride 不超过 60–100。

### 3.4 运行例子

**配置**：`region=(0,1200)`, `window_size=8`, `coordinate_stride=20`, `peak_half_window_nm=1.0`。

```
Step 1: levels=[0,64,...,960] 共 16 次全屏测量 → min_level=位于暗端, max_level=位于亮端
参考帧: 2 次测量 (暗底 + 亮参考)
Step 2: 窗起始 0,20,40,...,1180 + 锚点1192 → 约 61 次测量 (而非 ~1193 次, 提速 ~20×)
        每次: x=610 → 778.0 nm, x=630 → 778.9 nm, ...
拟合: 三次多项式, 例如 λ(x) ≈ 0.0417·x + 752.6 (近线性)
补全: 对 x = 4,5,...,1196 全部求值 → 返回 1193 个 (coordinate, wavelength)
```

产出 `CalibrationResult`：`coordinates`(稠密), `wavelength`(拟合值), `min_level`, `max_level`, `wavelength_fit_coefficients`。

---

## 4. Stage 3 —— `intensity`（`IntensityConfig`）

**目的**：为每个通道建立**灰度 → 归一化输出强度**的透过曲线（transfer curve）。流水线用的是 **batch** 版本 `batch_intensity_calibration`：一条 OSA trace 同时点亮多个**互不相邻**的通道，大幅减少测量次数。

### 4.1 参数（[pipeline.py:221](../src/slm_module/pipeline.py#L221)）

| 参数 | 默认 | 含义 |
|------|------|------|
| `levels` | `range(400,901,10)` = `[400,410,...,900]` | 每个通道扫的灰度档位（transfer curve 的横轴） |
| `window_size` | 15 | 通道点亮窗宽（GUI 里绑定到 `LayoutConfig.channel_width_px`） |
| `wavelength_window_nm` | None | 在通道波长附近取强度时的平均窗（nm）。None 用固定样本数 `average_half_window=2` |
| `group_skip_channels` | 2 | **batch 分组间隔**：一条 trace 内相邻两个点亮通道之间隔几个通道 pitch，减少串扰。2 = 一组测通道 0,3,6,…（见 §4.3） |
| `refine_center` | True | 先用 OSA 精修 778 nm 对应的中心坐标（`refine_center_coordinate_with_osa`），再据此建通道网格 |
| `refine_wavelength` | False | 用 Step 3 更细的峰重新拟合 coord→λ 映射（batch 路径下用最亮档 trace 的质心） |
| `osa` | `OSAStageSettings()` | OSA 扫描参数 |
| `outlier_policy` | None | 离群点复测，对每个通道的 sin² 曲线做（§2.4） |

### 4.2 算法流程

1. **（可选）精修中心**：`refine_center=True` 时，用 Step 2 线性映射预测 778 nm 的粗坐标，点亮那个窗，OSA 测一次，质心定出实测峰，把中心坐标修正 `(target−measured)/slope`。
2. **建通道网格**：`build_channel_calibration_grid` 在中心两侧按 `center ± (k+0.5)·pitch` 摆 `n_channels` 个通道，跳过与 `guard_bands` 重叠的通道，越界或撞保护带就往外顺延。得到 40 个通道坐标（20 pair）。
3. **参考帧**：暗底 + 亮参考（亮参考里保护带列强制暗）。
4. **分组扫描**：按 `group_skip_channels` 把通道分成若干组，每组内的通道**同时点亮**。对每组 × 每个 `level` 测一条 trace，在每个激活通道的波长处取强度（背景扣除 = raw，除以亮参考 = normalized）。
5. **（可选）离群复测**，输出 `intensity_levels`(归一化) 和 `raw_intensity_levels`(瓦特)，形状都是 `(40 通道, len(levels))`。

### 4.3 `group_skip_channels` 例子

40 个通道、`group_skip_channels=2`（group_step=3）：

```
组 0: 通道 0, 3, 6, 9, ...   ← 同一条 trace 一起点亮
组 1: 通道 1, 4, 7, 10, ...
组 2: 通道 2, 5, 8, 11, ...
```

测量次数 = 组数 × len(levels) = 3 × 51 = 153 次（+2 参考帧），而不是 40 × 51 = 2040 次。相邻点亮通道间隔 2 个 pitch，OSA 峰彼此分得开、串扰小。

### 4.4 运行例子

```
配置: levels=[400,...,900] (51档), window_size=15, group_skip_channels=2, refine_center=True
精修: 778 nm 预测在 x=610, 实测峰 777.98 nm → 中心修正到 x=610.5
建网格: 40 通道坐标, 跳过 776/780 保护带附近
参考帧: 2 次
扫描: 3 组 × 51 档 = 153 次测量
产出: intensity_levels[40,51], 每行一个通道的 sin²-形透过曲线, 用于 encode_to_pattern 反查灰度
```

> 注意：narrow-sweep 路径的 `sweep_span_nm` / `coordinate_stride`（在 `intensity_calibration` 里）**流水线不用**，流水线走 batch。

---

## 5. `tpa_center`（`TPACenterConfig`）

**目的**：扫描 layout 的中心波长，点亮**一个** pair，读荧光 monitor，用**加权二次拟合**定位真正的 TPA 双光子共振中心。

### 5.1 参数（[pipeline.py:233](../src/slm_module/pipeline.py#L233)）

| 参数 | 默认 | 含义 |
|------|------|------|
| `scan_center_nm` | None | 扫描中心；None = 用 `LayoutConfig.center_wl`(778) |
| `scan_halfspan_nm` | 0.05 | 扫描半宽（nm），实际扫 `center ± halfspan` |
| `n_points` | 11 | 扫描点数（≥3，二次拟合需要） |
| `pair_index` | 0 | 点亮哪个 pair |
| `drive_level` | 1.0 | 该 pair 的驱动强度 x=w=drive_level |
| `n_trials` | 1 | 整个扫描重复几遍（每个波长点多次采样求 SEM） |
| `repeats` | 1 | 每次 monitor 读取内部平均几次 |
| `settle` | 0.15 | 换 pattern 后等待稳定的秒数（GUI 里绑定到全局 monitor settle） |
| `subtract_background` | True | 每个波长点先测一次全暗背景再扣除（漂移抑制，代价是测量数 ×2） |

### 5.2 算法流程

1. 生成 `centers = linspace(center−halfspan, center+halfspan, n_points)`。
2. 对每个 `trial` × 每个 `center_wl`：以该中心重建 layout → （可选）先测背景 → 点亮 `pair_index`（x=w=`drive_level`）→ `settle` → 读 monitor（内部 `repeats` 次）→ 净信号 = signal − background。
3. 相同波长的多次读数求均值和 SEM（`average_trace_points`）→ 加权二次拟合 `fit_center_trace`：`signal = a·λ² + b·λ + c`，峰位 `center = −b/2a`。
4. 拟合带有效性判据（凹/峰在窗内），`fit.valid` 决定是否被下游采用。

### 5.3 运行例子

```
配置: scan_halfspan_nm=0.05, n_points=11, pair_index=0, drive_level=1.0, subtract_background=True
扫描: λ = 777.95, 777.96, ..., 778.05 nm (11点)
每点: 测背景(all-off) + 测 pair[0]点亮 → 净信号
读数(mV): 777.95→2.1, ..., 778.01→3.8(峰), ..., 778.05→2.4
测量次数: 1 trial × 11 点 × 2(含背景) = 22 次 monitor 读
拟合: 二次曲线峰在 778.012 ± 0.004 nm → center_fit.valid=True
下游: use_center_fit=True 时, pair_eta/comb_phase 用 778.012 建 layout
```

---

## 6. Stage 6 —— `pair_eta`（`PairEtaConfig`）

**目的**：标定每个 pair 的双光子效率 η。对每个 pair 在 (x, w) 上扫一个网格（含 x=0 / w=0 轴），拟合完整模型

```
Y = η²·(x·w) + a_x·x + q_x·x² + a_w·w + q_w·w² + d
    └双光子交叉项┘ └── x 单光子 ──┘ └── w 单光子 ──┘ └暗┘
```

模型对 `b:=η², a_x, q_x, a_w, q_w, d` 是**线性**的，加权最小二乘解出后 `η=√b`。

### 6.1 参数（[pipeline.py:246](../src/slm_module/pipeline.py#L246)）

| 参数 | 默认 | 含义 |
|------|------|------|
| `pair_indices` | `[]` | 要标定的 pair 列表；`[]` = 所有 pair |
| `sweep_min` | 0.3 | 每侧扫描的最小强度 |
| `sweep_max` | 1.0 | 每侧扫描的最大强度 |
| `n_points` | 5 | 每侧扫描点数（不含额外的 0 轴点） |
| `reduced_points` | True | **True** = 只测 1-D 曲线（x-only / w-only / cross），**False** = 完整 2-D 外积网格（见 §6.3） |
| `n_trials` | 5 | 每个网格点重复几遍（给每个 cell 一个经验 SEM） |
| `repeats` | 1 | 每次 monitor 读内部平均次数 |
| `settle` | 0.15 | 换 pattern 后等待秒数 |

### 6.2 算法流程

1. 建 layout（中心用 `ctx.center_wl`，即可能来自 `tpa_center`）。
2. `sweep = [0] + linspace(sweep_min, sweep_max, n_points)`（前置 0 提供 x=0/w=0 轴点，用来钉住单光子项）。
3. 对每个 pair × 每个网格点 × 每个 trial：只点亮该 pair 的 x/w 通道 → 读 monitor。
4. 每个 (x,w) cell 跨 trials 求均值/SEM → 加权最小二乘拟合 → `η ± err`（χ²/dof>1 时 Birge 放大误差）。

### 6.3 `reduced_points` —— 少测很多点

完整网格是 `(n_points+1)²` 个点。`reduced_points=True` 时只测让每个拟合项可辨识的那几条线（`build_pair_points`）：

```
dark    (0, 0)      钉 offset d          共 1 点
x-only  (r, 0)      钉 a_x, q_x          共 n_points 点
w-only  (0, r)      钉 a_w, q_w          共 n_points 点
cross   (1, r)      x 固定=1, w 扫        共 n_points 点 ← 唯一 x·w≠0 的线, 钉 η
```

`r` 走 `linspace(sweep_min, sweep_max, n_points)`，去重后约 `3·n_points` 个点，而非 `(n_points+1)²`。例：`n_points=5` → reduced 约 16 点 vs 完整 36 点。

### 6.4 运行例子

```
配置: pair_indices=[] (全部20对), sweep_min=0.3, sweep_max=1.0, n_points=5, reduced_points=True, n_trials=5
每对测点: dark + x-only(5) + w-only(5) + cross(5), 去重 ≈ 16 点
测量数: 5 trials × 20 pairs × 16 点 = 1600 次 monitor 读
拟合(pair 0): η=0.82±0.03, a_x=0.11, q_x=-0.02, ..., χ²/dof=1.2
产出: TPAPairResult, 每对一个 PairFit(η...), 供 Step 7 当 PairModel
```

---

## 7. Stage 7 —— `comb_phase`（`CombPhaseConfig`）

**目的**：测每个 target pair 相对 reference pair 的**梳齿相位差 ΔΦ_comb**。固定参考 pair，扫 target pair 的相位（通过 SLM 灰度控制 `x=sin²(φ/2)`），读荧光干涉条纹，拟合出 ΔΦ_comb。

### 7.1 参数（[pipeline.py:258](../src/slm_module/pipeline.py#L258)）

| 参数 | 默认 | 含义 |
|------|------|------|
| `ref_index` | 0 | 参考 pair 序号 |
| `tgt_indices` | `[3]` | 要测的 target pair 列表（每个各出一条条纹 + 一份 CSV/JSON） |
| `sweep_points` | 15 | 相位扫描点数 |
| `phi_start_deg` | 0.0 | target 相位起点（度） |
| `phi_stop_deg` | 180.0 | target 相位终点（度）。0–180 是可达的半个条纹周期 |
| `ref_phase_deg` | 180.0 | 参考 pair 固定相位（180° ↔ 强度 1，全开） |
| `n_trials` | 10 | 整个扫描重复几遍 |
| `repeats` | 1 | 每次 monitor 读内部平均次数 |
| `settle` | 0.15 | 换 pattern 后等待秒数 |
| `bound_frac` | 1.0 | 拟合约束：数字 = 把 a:b 锁到 Step 6 的 η_ref:η_tgt 比、共享尺度限制在 ±frac；**None = 无约束闭式拟合** |
| `single_beam_bg` | True | 把两个 pair 的 Step 6 单光子响应作为固定背景折进拟合 |
| `measure_dark` | True | 测 all-off 暗读数做逐行扣除（漂移抑制） |
| `dark_per_trial` | True | 每个 trial 开头都重测一次暗（跟踪慢漂移）；False = 全程只在开头测一次 |

### 7.2 算法流程

1. 建 layout + 从 `pair_etas` 取出 target/reference 的 `PairModel`（含 η、单光子系数）。
2. `build_phase_sweep`：target 对称驱动 `φ^x=φ^w=φ` 扫 `[phi_start, phi_stop]`，参考固定在 `ref_phase_deg`。生成 `(x_t,w_t,x_r,w_r)` 强度元组，`x=sin²(φ/2)`。
3. 对每个 trial（可选先测暗）× 每个 drive 点：只点亮 target+reference → 读 monitor → 存行（含该 trial 的 dark）。
4. `fit_result`：拟合 `Y = |a·e^{iΦ_r} + b·e^{iΦ_t}|²`-型条纹，解出 `dphi_comb`（wrap 到 (−π,π]），受 `frac` / `single_beam_bg` 控制。

### 7.3 运行例子

```
配置: ref_index=0, tgt_indices=[3,7], sweep_points=15, phi 0→180°, ref_phase=180°,
      n_trials=10, bound_frac=1.0, single_beam_bg=True, measure_dark=True, dark_per_trial=True
每个 target 扫描: 每 trial = 1 暗 + 15 相位点 = 16 读; ×10 trials = 160 读
两个 target: 320 次 monitor 读
拟合(pair 3): a=0.51, b=0.48, ΔΦ_comb = +42.3 ± 1.8°, χ²/dof=1.1
产出: comb_phase JSON(每对 dphi_comb_deg/err/a/b/chi2) + 每对 _pair{k}.csv/.json
```

---

## 8. 端到端完整例子

一次从头到尾的全链运行（GUI「Run」等价的 `PipelineRequest`）：

```python
from slm_module.pipeline import (
    PipelineRequest, StagePlan, InputSpec, LayoutConfig,
    WlMapConfig, IntensityConfig, TPACenterConfig, PairEtaConfig, CombPhaseConfig,
    run_pipeline, PipelineInstruments,
)
from slm_module.calibration.outliers import OutlierRemeasurePolicy

request = PipelineRequest(
    layout=LayoutConfig(n_channels=20, channel_width_px=15, gap_px=5, center_wl=778.0),
    use_center_fit=True,
    stages=[
        StagePlan("wl_map", WlMapConfig(
            region=(0, 1200), coordinate_stride=20,     # ← 提速: 每 20 列测一次
            peak_half_window_nm=1.0,
            outlier_policy=OutlierRemeasurePolicy(k_sigma=4.0),
        ), inputs={}, output_path="out/wl_map.json"),

        StagePlan("intensity", IntensityConfig(
            levels=list(range(400, 901, 10)), group_skip_channels=2, refine_center=True,
        ), inputs={"wl_map": InputSpec("memory")},       # ← 用上一步内存结果
           output_path="out/intensity.json",
           extra_outputs={"csv": "out/intensity.csv"}),

        StagePlan("tpa_center", TPACenterConfig(
            scan_halfspan_nm=0.05, n_points=11, pair_index=0,
        ), inputs={"intensity_calib": InputSpec("memory")},
           output_path="out/center.json"),

        StagePlan("pair_eta", PairEtaConfig(
            sweep_min=0.3, sweep_max=1.0, n_points=5, reduced_points=True, n_trials=5,
        ), inputs={"intensity_calib": InputSpec("memory"),
                   "center_fit": InputSpec("memory")},
           output_path="out/pair_eta.json",
           extra_outputs={"csv": "out/pair_eta.csv"}),

        StagePlan("comb_phase", CombPhaseConfig(
            ref_index=0, tgt_indices=[3, 7], sweep_points=15, n_trials=10,
        ), inputs={"intensity_calib": InputSpec("memory"),
                   "pair_etas": InputSpec("memory"),
                   "center_fit": InputSpec("memory")},
           output_path="out/comb_phase.json"),
    ],
)

instruments = PipelineInstruments(slm=slm, osa=osa, monitor=daq)
outcome = run_pipeline(request, instruments, stop_event=stop, progress_callback=cb)
print(outcome.summaries)   # 每个 stage 一行结果摘要
```

**只跑后半段**（复用磁盘上的历史校准）：把前面 stage 从 `stages` 去掉，把内存输入换成文件：

```python
StagePlan("pair_eta", PairEtaConfig(...),
    inputs={"intensity_calib": InputSpec("file", path="out/intensity.json"),
            "center_fit":       InputSpec("file", path="out/center.json")},
    output_path="out/pair_eta.json")
```

---

## 9. 参数速查：想提速改哪里

| 想加速的阶段 | 调这个参数 | 代价 |
|------|------|------|
| Step 2 (wl_map) | `coordinate_stride` ↑ | 空间密度换速度；靠线性拟合补全，几乎无损（§3.3） |
| Step 2 (wl_map) | `region` 收窄 | 只扫真正被照亮的列 |
| Step 3 (intensity) | `group_skip_channels` ↓ | 每条 trace 挤更多通道，但串扰风险↑ |
| Step 3 (intensity) | `levels` 稀疏化 | transfer curve 点变少 |
| tpa_center | `subtract_background=False` | 少一半读数，但漂移抑制变弱 |
| pair_eta | `reduced_points=True` | 1-D 曲线代替 2-D 网格（§6.3） |
| pair_eta / comb_phase | `n_trials` ↓ | SEM 变大、误差棒变宽 |
| 所有 monitor 阶段 | `settle` ↓ | 前提是 SLM 已稳定，否则读到过渡态 |
