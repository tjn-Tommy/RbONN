
# Two-pass Phase-aware Calibration and Compensation LUT for RbONN

## 1. 项目背景

RbONN 的实验目标是利用 Rb 原子双光子吸收实现光学乘法 / 累加。对于一组关于 778 nm 对称的频率通道，理想情况下，每个 pair 的双光子贡献满足

$$Q_k \propto w_k x_k,$$

并且所有 pair 的双光子振幅同相叠加后，荧光信号满足

$$F-B \propto \left(\sum_k w_kx_k\right)^2.$$

因此，在非负输入和非负权重情况下，可以通过 $\sqrt{F-B}$ 读出 dot product。当前项目的物理目标正是基于 Rb TPA 实现这种光学乘法 / 累加机制。

但是现有 SLM + PBS 强度编码并非纯粹的 intensity modulation。单程 crossed-PBS 编码会引入灰度相关的场相位，使得不同 wavelength pair 的 TPA 路径无法简单地按强度相加。原始 intensity-only multiplier 因此会受到 phase-amplitude coupling 的限制。

为解决这一问题，本项目计划实现 two-pass SLM–PBS 联合编码：第一程和第二程分别使用 SLM 上的两个 line / region，通过联合选择两程灰度值，实现对每个 channel 或 pair 的有效复场控制。项目最终目标是建立 empirical compensation LUT，使 corrected encoding 后的 Rb TPA 信号恢复 coherent MAC 行为。

---

## 2. 当前代码基础

当前 RbONN 代码库已经具备以下基础模块：

### 2.1 Step 2 / Step 3 calibration

现有 calibration 模块已经支持：

1. 通过 bright-window sweep 建立 SLM 横坐标到 wavelength 的映射；
2. 对每个 calibrated coordinate 扫描 grayscale level；
3. 得到每个坐标的 grayscale transfer curve；
4. 保存 `level_range`、`intensity_levels`、`raw_intensity_levels` 等字段。

这些功能已经由 `CalibrationResult` 和 `calibration_new.py` 中的 Step 2 / Step 3 calibration 流程承载。`CalibrationResult` 当前包含 wavelength、coordinates、min/max level、level_range、intensity_levels、raw_intensity_levels 以及 wavelength fit coefficients。

### 2.2 Channel layout and intensity encoder

现有 `encoding.py` 已经定义了：

* `EncodingChannel`
* `ChannelLayout`
* `compute_channel_geometry`
* `build_channel_layout`
* `encode_to_pattern`

其中 `build_channel_layout` 会基于 Step 2 的 wavelength map，在 778 nm 附近构造对称的 x/w channel pairs，并避开 Rb guard bands。

`encode_to_pattern` 当前接受 `x_vals` 和 `w_vals`，再通过每个 channel 的 measured transfer curve 将目标值映射成 SLM grayscale pattern。

### 2.3 TPA center scan

代码库中已经有 `tpa_center.py`，用于扫描 center wavelength、重建 layout、点亮一个 pair，并读取 fluorescence monitor 信号，最终用 weighted quadratic fit 定位 TPA center。

该模块已经提供了一个重要范式：
**动态生成 pattern → display 到 SLM → 等待 settle → 读取 monitor → 保存 scan result → fit。**

这套 acquire–fit 流程可以直接扩展为后续 phase calibration / pair LUT calibration 的实验软件框架。

---

## 3. 新项目总目标

本项目的总目标是实现一个 two-pass phase-aware calibration and compensation pipeline，使 RbONN 从当前的 intensity-only encoding 扩展为 phase-aware complex-field encoding。

软件上，需要实现以下完整链条：

1. 扫描 two-pass 中两条 SLM line / region 的坐标与 wavelength map；
2. 分别建立 Region A / Region B 的 grayscale-to-intensity transfer curves；
3. 使用 reference-pair interference 标定每个 TPA pair 的 effective amplitude 和 effective phase；
4. 建立 pair-level compensation LUT；
5. 实现 phase-aware compiler，将目标 $w_k x_k$ 映射为 two-pass gray-level states；
6. 输出 corrected two-pass SLM pattern；
7. 用 held-out MAC 测试验证 corrected encoding 相比 raw encoding 的改善。

核心校准对象不是单个 channel 的绝对相位，而是每个 TPA pair 的有效复系数：

$$Q_k = H_k C_{k,+}C_{k,-} = a_k e^{i\Theta_k}.$$

只要 LUT 能为每个 pair 选择灰度状态，使

$$a_k \propto w_kx_k, \qquad \Theta_k \approx \Phi_0,$$

就可以恢复 coherent multiply–accumulate。这个 pair-level LUT 能自动吸收 SLM phase、PBS phase、relay phase、Rb response phase 和其他静态系统误差。

---

## 4. 推荐软件架构

建议在现有 `src/slm_module` 下新增一个独立子包：

```text
src/slm_module/two_pass/
    __init__.py
    geometry.py
    line_scan.py
    amplitude.py
    phase.py
    lut.py
    compiler.py
    pattern.py
    io.py

```

不要直接修改现有 `encoding.py` 的主路径。现有 encoder 继续服务于 single-pass / intensity-only encoding。Two-pass 逻辑作为更高层的 phase-aware compiler 存在。

---

## 5. 模块设计

### 5.1 `geometry.py`：two-pass 几何描述

目标：描述第一程和第二程在 SLM 上对应的两个 line / region。

建议数据结构：

```python
@dataclass
class TwoPassRegion:
    name: str              # "A" or "B"
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    pass_index: int        # 1 or 2
    port_type: str         # "sin" / "cos" / "crossed" / "parallel"

```

```python
@dataclass
class TwoPassGeometry:
    region_a: TwoPassRegion
    region_b: TwoPassRegion
    slm_width: int
    slm_height: int
    second_pass_mirrored_x: bool = False
    second_pass_x_offset_px: float = 0.0

```

功能目标：

1. 明确第一程和第二程分别打在 SLM 哪个区域；
2. 允许第二程发生 x mirror 或 offset；
3. 为后续 line coordinate scan 和 pattern rendering 提供统一几何接口。

---

### 5.2 `line_scan.py`：两条 line 的坐标扫描

目标：分别扫描 Region A 和 Region B 的 wavelength-to-pixel map。

现有 Step 2 的 `wavelength_calibration` 已经能通过 bright-window sweep 建立 coordinate→wavelength map。
新的 two-pass line scan 可以复用这套逻辑，但需要允许：

1. 只扫描 Region A 的 y 区域；
2. 只扫描 Region B 的 y 区域；
3. 分别保存两个 line 的 wavelength calibration；
4. 检查两条 line 的 slope / intercept / mirror relation。

建议结果结构：

```python
@dataclass
class LineMapResult:
    region_name: str
    coordinates: np.ndarray
    wavelengths_nm: np.ndarray
    fit_coefficients: np.ndarray
    valid_range_px: tuple[int, int]
    rms_residual_nm: float

```

推荐实现函数：

```python
def scan_two_pass_lines(
    osa,
    slm,
    geometry: TwoPassGeometry,
    base_calibration_settings,
    *,
    window_size: int,
    level_on: int,
    level_off: int,
    region_a_scan: tuple[int, int],
    region_b_scan: tuple[int, int],
) -> tuple[LineMapResult, LineMapResult]:
    ...

```

验收标准：

1. 两条 line 都能得到单调 wavelength map；
2. 两条 line 的 wavelength slope方向明确；
3. 若第二程出现 mirror，需要在 result 里显式标记；
4. pair 计算时能够把同一个目标 wavelength pair 映射到 Region A 和 Region B 的正确 x 坐标。

---

### 5.3 `amplitude.py`：两程 amplitude LUT

目标：分别建立 Region A 和 Region B 的 grayscale-to-output transfer curve。

现有 Step 3 已经定义了 per-coordinate grayscale transfer curve：一个 `window_size` 宽的窗口在每个 calibrated coordinate 上点亮，扫描 `level_range`，测量输出强度。
Two-pass 中需要对 Region A 和 Region B 分别做类似校准。

建议数据结构：

```python
@dataclass
class RegionAmplitudeLUT:
    region_name: str
    coordinates: np.ndarray
    wavelengths_nm: np.ndarray
    level_range: np.ndarray
    intensity_levels: np.ndarray
    raw_intensity_levels: np.ndarray | None
    off_level: np.ndarray
    on_level: np.ndarray

```

推荐函数：

```python
def calibrate_region_amplitude(
    detector,
    slm,
    line_map: LineMapResult,
    region: TwoPassRegion,
    *,
    level_range: np.ndarray,
    window_size: int,
    backend: Literal["osa", "daq"],
) -> RegionAmplitudeLUT:
    ...

```

```python
def level_for_region_coordinate(
    lut: RegionAmplitudeLUT,
    coordinate: float,
    target_intensity: float,
) -> int:
    ...

```

注意事项：

* 这里的 `target_intensity` 是功率强度，不是场幅度；
* 若最终要编码场幅度 $A$，则目标功率应使用 $I = A^2$；
* 若使用互补 two-pass 路径，单程强度可能需要设置为 $\sqrt{I_{\rm out}}$。

---

### 5.4 `phase.py`：reference-pair interference phase calibration

目标：直接标定每个 TPA pair 的 effective complex coefficient：

$$Q_{k,m}=a_{k,m}e^{i\Theta_{k,m}}.$$

这里不建议先尝试恢复每个单独 channel 的绝对相位。TPA 真正关心的是 pair-level phase：

$$\arg Q_k = \arg H_k + \arg C_{k,+} + \arg C_{k,-}.$$

为了让所有 two-photon pathways 同相，需要满足 $\arg Q_k = \Phi_0$。

建议数据结构：

```python
@dataclass
class TwoPassState:
    pair_index: int
    state_index: int
    x_levels: tuple[int, int]   # (pass1, pass2) for x side
    w_levels: tuple[int, int]   # (pass1, pass2) for w side
    nominal_amplitude: float
    metadata: dict[str, Any] = field(default_factory=dict)

```

```python
@dataclass
class PairComplexState:
    pair_index: int
    state_index: int
    amplitude: float
    phase_rad: float
    visibility: float
    signal_mean: float
    signal_std: float
    state: TwoPassState

```

```python
@dataclass
class PairPhaseLUT:
    reference_pair: int
    states_by_pair: dict[int, list[PairComplexState]]
    phase_zero_rad: float
    created_at: str

```

推荐函数：

```python
def measure_pair_interference(
    monitor,
    slm,
    layout,
    reference_state: TwoPassState,
    variable_states: list[TwoPassState],
    *,
    repeats: int,
    settle: float,
    subtract_background: bool = True,
) -> list[InterferenceMeasurement]:
    ...

```

```python
def fit_complex_states(
    measurements: list[InterferenceMeasurement],
    *,
    reference_pair: int,
) -> PairPhaseLUT:
    ...

```

拟合模型：

$$F = B + \eta \vert{}Q_r + Q_k\vert{}^2$$

展开为：

$$F = B+\eta\left(\vert{}Q_r\vert{}^2+\vert{}Q_k\vert{}^2 +2\vert{}Q_r\vert{}\vert{}Q_k\vert{}\cos(\Theta_k-\Theta_r)\right).$$

如果实验上能额外扫描一个 reference phase knob，则拟合形式可以写为：

$$F(\delta)=c_0+c_1\cos(\delta+\phi_0).$$

若暂时没有干净的 phase knob，可以先通过一组 candidate two-pass states 的 fluorescence contrast 做相对排序，得到 empirical phase labels，再通过 compiler 选择相位最接近的状态。

---

### 5.5 `lut.py`：量化补偿 LUT

目标：把每个 pair 的 candidate states 整理成可查询的 compensation LUT。

建议结构：

```python
@dataclass
class CompensationLUT:
    pair_lut: PairPhaseLUT
    target_phase_rad: float
    amplitude_scale: float
    max_phase_error_rad: float
    min_visibility: float

```

核心接口：

```python
def select_state_for_target(
    lut: CompensationLUT,
    pair_index: int,
    target_amplitude: float,
    *,
    lambda_amp: float = 1.0,
    lambda_phase: float = 10.0,
) -> PairComplexState:
    ...

```

目标函数：

$$m_k^\star = \arg\min_m \left[ \lambda_A(a_{k,m}-\alpha w_kx_k)^2 + \lambda_\Theta \operatorname{wrap}^2(\Theta_{k,m}-\Phi_0) \right].$$

这正是 hardware-aware compiler 的核心。

---

### 5.6 `compiler.py`：phase-aware MAC compiler

目标：输入 $x$、$w$，输出 two-pass gray-level pattern。

推荐接口：

```python
@dataclass
class CompileResult:
    pattern: np.ndarray
    selected_states: list[PairComplexState]
    target_dot: float
    predicted_complex_sum: complex
    predicted_signal: float
    diagnostics: dict[str, Any]

```

```python
def compile_two_pass_mac(
    x_vals: np.ndarray,
    w_vals: np.ndarray,
    compensation_lut: CompensationLUT,
    geometry: TwoPassGeometry,
    *,
    slm_width: int,
    slm_height: int,
    normalize: bool = True,
) -> CompileResult:
    ...

```

流程：

1. 计算每个 pair 的目标贡献：

$$d_k = w_k x_k$$


2. 把 $d_k$ 映射为目标 field amplitude；
3. 从 `CompensationLUT` 中选择 phase-aligned state；
4. 渲染 Region A 和 Region B 的 SLM pattern；
5. 合并成完整 two-pass mask；
6. 输出 diagnostics，包括：
* 每个 pair 的目标 amplitude；
* 选中 state 的 measured amplitude；
* 选中 state 的 phase error；
* predicted complex sum；
* predicted signal。



---

### 5.7 `pattern.py`：two-pass pattern renderer

目标：把第一程和第二程的 channel states 写到 SLM 的两个区域。

现有 `encode_to_pattern` 是 single-layout renderer，它默认一组 x/w channels 写到整个 SLM pattern。
Two-pass renderer 应该扩展为：

```python
def render_two_pass_pattern(
    selected_states: list[PairComplexState],
    geometry: TwoPassGeometry,
    line_map_a: LineMapResult,
    line_map_b: LineMapResult,
    *,
    slm_width: int,
    slm_height: int,
    background_level: int | np.ndarray,
) -> np.ndarray:
    ...

```

关键要求：

1. Region A 只写第一程灰度；
2. Region B 只写第二程灰度；
3. 其他区域保持 off level；
4. 若第二程 x 方向镜像，需要正确变换 coordinate；
5. 支持每个 pair 的 x side 和 w side 分别写入灰度；
6. 支持 debug overlay / preview。

---

### 5.8 `io.py`：保存和加载校准结果

目标：所有实验校准结果必须能保存、复现、回放。

推荐文件格式：

```text
data/
    calibrations/
        2026-07-xx_two_pass_line_map.json
        2026-07-xx_region_a_amplitude.npz
        2026-07-xx_region_b_amplitude.npz
        2026-07-xx_pair_phase_lut.json
        2026-07-xx_compensation_lut.json

```

推荐原则：

* 小型 metadata 用 JSON；
* 大型 ndarray 用 NPZ；
* 每个 file 都保存 Git commit hash、date、experiment notes、SLM config、OSA/DAQ settings；
* 每次 MAC validation 都保存所用 LUT 的文件名和 hash。

---

## 6. 实验软件工作流

### Step A：two-pass line coordinate scan

输入：

* SLM controller
* OSA controller
* Region A / Region B 几何
* bright-window scan 设置

输出：

* `LineMapResult(region="A")`
* `LineMapResult(region="B")`

验收：

* 两条线的 wavelength map 都单调；
* 778 nm center 可以定位；
* x/w pair 能在两条 line 上一致映射；
* 第二程是否 mirror 被自动记录。

---

### Step B：Region A / B amplitude LUT

输入：

* `LineMapResult`
* level_range
* window_size
* OSA 或 DAQ backend

输出：

* `RegionAmplitudeLUT(A)`
* `RegionAmplitudeLUT(B)`

验收：

* 每个选中 channel 至少有 5 个可用灰度点；
* transfer curve 重复性足够；
* low / mid / high / max 几档可稳定调用；
* off-level 不应明显漏光。

---

### Step C：reference pair 选择

输入：

* candidate pair list
* fluorescence monitor
* preliminary amplitude LUT

流程：

1. 对多个 pair 逐个打开；
2. 测 fluorescence signal；
3. 选择信号强、重复性好、phase response 稳定的 pair 作为 reference pair。

输出：

* `reference_pair_index`
* `reference_state`

---

### Step D：pair-level phase calibration

输入：

* reference pair
* variable pair candidates
* candidate two-pass states
* fluorescence monitor

流程：

1. 固定 reference pair；
2. 对 variable pair 扫 candidate states；
3. 记录 fluorescence；
4. 拟合或估计每个 state 的 effective amplitude / phase；
5. 生成 `PairPhaseLUT`。

输出：


$$\mathcal{C}_k = \{(a_{k,m},\Theta_{k,m},g_{1,k,m},g_{2,k,m})\}_{m=1}^M$$

验收：

* 每个 pair 至少保留 5 个有效 candidate states；
* phase error 可估计；
* visibility 太低的状态剔除；
* LUT 在重复测量下稳定。

---

### Step E：建立 compensation LUT

输入：

* `PairPhaseLUT`
* target phase $\Phi_0$
* amplitude normalization scale

输出：

* `CompensationLUT`

选择规则：


$$m_k^\star = \arg\min_m \left[ \lambda_A(a_{k,m}-\alpha w_kx_k)^2 + \lambda_\Theta \operatorname{wrap}^2(\Theta_{k,m}-\Phi_0) \right].$$

---

### Step F：corrected MAC validation

输入：

* held-out $(x,w)$ test set
* compensation LUT
* fluorescence monitor

流程：

1. 对每组 $(x,w)$，调用 `compile_two_pass_mac`；
2. display corrected pattern；
3. 读取 fluorescence ($F$)；
4. 测 background ($B$)；
5. 比较：

$$\sqrt{F-B} \quad \text{vs.} \quad D=\sum_k w_kx_k.$$


6. 同时测 raw / uncompensated encoding 作为对照。

输出：

* corrected $R^2$
* raw $R^2$
* corrected NRMSE
* raw NRMSE
* parity plot
* residual plot

最终论文级判据：


$$R^2_{\rm corr} \gg R^2_{\rm raw}, \qquad \mathrm{NRMSE}_{\rm corr} \ll \mathrm{NRMSE}_{\rm raw}.$$

---

## 7. 最小可行版本

为了在短时间内完成 proof-of-principle，不建议一开始做全 20 pair。最小版本建议为：

* 使用 $K=3$ 个 pair；
* 每个 pair 只保留 5–10 个 candidate states；
* 每个 state 重复测量 3–5 次；
* 先支持非负 $(x,w)$；
* 暂不支持 signed weight；
* 暂不做 full $256 \times 256$ LUT；
* 暂不恢复单 channel absolute phase；
* 优先实现 pair-level empirical LUT。

最小交付物：

```text
1. two-pass line coordinate scan
2. Region A/B amplitude LUT
3. reference-pair interference measurement
4. PairPhaseLUT
5. CompensationLUT
6. compile_two_pass_mac()
7. corrected-vs-raw MAC validation figure

```

---

## 8. 代码实现优先级

### Priority 0：不破坏现有路径

现有 single-pass intensity encoding、Step 3 calibration、TPA center scan 应保持兼容。

### Priority 1：新增数据结构

先实现：

```text
TwoPassRegion
TwoPassGeometry
LineMapResult
RegionAmplitudeLUT
TwoPassState
PairComplexState
PairPhaseLUT
CompensationLUT
CompileResult

```

### Priority 2：pattern renderer

实现 `render_two_pass_pattern()`，使你可以手动指定每个 pair 的 two-pass gray state，并确认 SLM 上两条 line 正确显示。

### Priority 3：line scan

复用现有 `wavelength_calibration` 思路，对 Region A 和 Region B 分别做 coordinate scan。

### Priority 4：amplitude LUT

复用 Step 3 transfer curve 思路，分别为 Region A / B 建立灰度→强度 LUT。

### Priority 5：phase LUT

实现 reference-pair interference acquisition 和 fitting。

### Priority 6：compiler

实现从目标 $(x,w)$ 到 selected states 再到 SLM pattern 的完整流程。

### Priority 7：validation

实现 corrected/raw 自动测量脚本和 plotting script。

---

## 9. 推荐测试设计

新增测试文件：

```text
tests/test_two_pass_geometry.py
tests/test_two_pass_pattern.py
tests/test_two_pass_lut.py
tests/test_two_pass_compiler.py
tests/test_phase_fit.py

```

测试内容：

1. Region A / B 坐标变换正确；
2. second-pass mirror 逻辑正确；
3. target wavelength 能正确映射到两条 line；
4. compensation LUT 能按 amplitude/phase cost 正确选择 state；
5. phase wrapping 正确处理 $(-\pi,\pi)$ 边界；
6. `compile_two_pass_mac()` 对简单 mock LUT 输出预期 pattern；
7. raw / corrected predicted complex sum 可复现。

---

## 10. 项目完成标准

本项目完成后，应满足以下条件：

1. 可以扫描并保存 two-pass 两条 line 的 wavelength map；
2. 可以分别建立 Region A / B 的 grayscale transfer curve；
3. 可以选择 reference pair 并测量 pair interference；
4. 可以生成 pair-level complex LUT；
5. 可以根据目标 $w_k x_k$ 选择 phase-compensated two-pass state；
6. 可以输出完整 two-pass SLM mask；
7. 可以完成 $K=3$ 的 corrected MAC validation；
8. corrected encoding 相比 raw encoding 显著提升 $\sqrt{F-B}$ 与 $\sum_k w_kx_k$ 的线性关系。

最终实验叙事为：

> Rb TPA coherent MAC 对 spectral phase 敏感，single-pass intensity-only encoding 受到 amplitude–phase coupling 限制。通过 two-pass SLM–PBS joint complex-field encoding，并使用 reference-pair interference 建立 empirical compensation LUT，可以让多个 wavelength pair 的 effective TPA phase 对齐，从而恢复 phase-compensated coherent multiply–accumulate。


