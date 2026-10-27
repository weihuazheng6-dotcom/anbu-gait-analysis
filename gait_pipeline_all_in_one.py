"""
╔══════════════════════════════════════════════════════════════════╗
║           步态分析系统 — 完整工作流（单文件整合版）               ║
║                                                                  ║
║  模块结构（按顺序）：                                             ║
║   1. 全局常量 & 依赖导入                                         ║
║   2. 时间戳 / 滤波 工具函数                                      ║
║   3. 步态事件检测（足跟着地 / 离地）                             ║
║   4. 6项步态参数提取（倒立摆模型 + 压力信号）                    ║
║   5. 步长校准（1330cm 已知距离）                                 ║
║   6. GaitNormModel：统计正常模型（均值±n×σ 判断）               ║
║   7. GaitDeepModel：深度学习模型（1D-CNN + BiLSTM + Attention）  ║
║   8. 受试者信息加载 & 文件名自动分类                             ║
║   9. train_stat()：统计模型训练流水线                            ║
║  10. train_deep()：深度学习训练流水线                            ║
║  11. 可视化：步态报告图 / 时序图                                 ║
║  12. HTML / JSON 报告生成                                        ║
║  13. analyze()：推理主流程（自动识别模型类型）                   ║
║  14. __main__：命令行入口（train / analyze 双模式）              ║
║                                                                  ║
║  运行示例：                                                       ║
║    # 训练（深度学习）                                             ║
║    python gait_pipeline_all_in_one.py train                      ║
║        --dataset ./dataset --output ./model --mode deep          ║
║                                                                  ║
║    # 推理 + 生成报告                                             ║
║    python gait_pipeline_all_in_one.py analyze                    ║
║        --input ./test/直线来回.csv                               ║
║        --model ./model/gait_deep_model.pkl                       ║
║        --height 170                                              ║
╚══════════════════════════════════════════════════════════════════╝

数据集文件夹结构：
  dataset/
    subject_001/
      subject_info.json          ← {"id":"001","height_cm":170,"weight_kg":65}
      1330cm_1.csv               ← 用于步长校准的已知距离行走文件
      直线来回_1.csv             ← is_normal = True
      双任务直线_1.csv           ← is_normal = False（认知双任务，视为异常）
      上楼梯_1.csv               ← is_normal = True
      ...
    subject_002/
      ...

传感器 CSV 列名（左右各一组）：
  timestamp,
  P_first_meta_R, P_Fifth_meta_R, P_heel_R,   ← 右脚 3 点压力
  acc_x_R, acc_y_R, acc_z_R,                   ← 右 IMU 加速度
  ave_x_R, ave_y_R, ave_z_R,                   ← 右 IMU 陀螺仪
  ang_x_R, ang_y_R, ang_z_R,                   ← 右 IMU 姿态角
  P_first_meta_L, P_Fifth_meta_L, P_heel_L,   ← 左脚 3 点压力
  acc_x_L, acc_y_L, acc_z_L,
  ave_x_L, ave_y_L, ave_z_L,
  ang_x_L, ang_y_L, ang_z_L,
  Label                                         ← 可选
"""

# ──────────────────────────────────────────────────────────────────
# 第 1 节：依赖导入 & 全局常量
# ──────────────────────────────────────────────────────────────────
import os
import sys
import json
import glob
import base64
import argparse
import warnings
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                   # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
from scipy.stats import norm as scipy_norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# 采样率（Hz）
FS = 100

# 滑动窗口参数（深度学习用）
WINDOW_SEC  = 3.0
WINDOW_SIZE = int(WINDOW_SEC * FS)     # 300 个采样点
STEP_SIZE   = int(WINDOW_SIZE * 0.5)   # 50% 重叠（150 点）

# ── 传感器列名 ──
PRESSURE_COLS_R = ['P_first_meta_R', 'P_Fifth_meta_R', 'P_heel_R']
PRESSURE_COLS_L = ['P_first_meta_L', 'P_Fifth_meta_L', 'P_heel_L']
ACC_COLS_R      = ['acc_x_R', 'acc_y_R', 'acc_z_R']
ACC_COLS_L      = ['acc_x_L', 'acc_y_L', 'acc_z_L']
GYRO_COLS_R     = ['ave_x_R', 'ave_y_R', 'ave_z_R']
GYRO_COLS_L     = ['ave_x_L', 'ave_y_L', 'ave_z_L']

# 深度模型输入的 24 通道顺序（左右各 12 通道）
SENSOR_COLS = (
    PRESSURE_COLS_R +
    ['acc_x_R', 'acc_y_R', 'acc_z_R',
     'ave_x_R', 'ave_y_R', 'ave_z_R',
     'ang_x_R', 'ang_y_R', 'ang_z_R'] +
    PRESSURE_COLS_L +
    ['acc_x_L', 'acc_y_L', 'acc_z_L',
     'ave_x_L', 'ave_y_L', 'ave_z_L',
     'ang_x_L', 'ang_y_L', 'ang_z_L']
)

# 6 项步态参数键名 & 中文标签
PARAM_KEYS = [
    'step_length_m',
    'step_speed_ms',
    'cadence_spm',
    'step_width_ratio',
    'stability_score',
    'symmetry_score',
]
PARAM_LABELS = {
    'step_length_m':    '步长 (m)',
    'step_speed_ms':    '步速 (m/s)',
    'cadence_spm':      '步频 (步/分)',
    'step_width_ratio': '负载不对称度',
    'stability_score':  '稳定性',
    'symmetry_score':   '对称性',
}

# ── 报告配色 ──
COLOR_NORMAL   = '#2ecc71'
COLOR_WARN     = '#f39c12'
COLOR_ABNORMAL = '#e74c3c'
COLOR_REF      = '#3498db'
COLOR_BG       = '#f8f9fa'


# ──────────────────────────────────────────────────────────────────
# 第 2 节：工具函数 — 时间戳解析 & Butterworth 滤波
# ──────────────────────────────────────────────────────────────────

def parse_timestamps(timestamps, fs=FS):
    """
    把 timestamp 列转成以秒为单位的浮点数组。
    若无法解析（非标准格式），则按采样率生成等间距时间轴。
    NaN 值用线性插值填充。
    """
    try:
        ts = pd.to_datetime(timestamps, errors='coerce')
        if ts.isna().all():
            raise ValueError("全部无法解析")
        t0 = ts.dropna().iloc[0]
        t_sec = (ts - t0).dt.total_seconds().values
        # 用线性插值填补 NaN（例如 BLE 丢包导致的空洞）
        nan_mask = np.isnan(t_sec)
        if nan_mask.any():
            idx = np.arange(len(t_sec))
            t_sec[nan_mask] = np.interp(
                idx[nan_mask], idx[~nan_mask], t_sec[~nan_mask])
        return t_sec
    except Exception:
        return np.arange(len(timestamps)) / fs


def _lowpass(data, cutoff=6.0, fs=FS, order=4):
    """
    Butterworth 低通滤波，用于平滑压力/加速度信号。
    cutoff: 截止频率 Hz
    数据长度过短时原样返回，避免 filtfilt 崩溃。
    """
    nyq = 0.5 * fs
    wn = min(cutoff / nyq, 0.99)
    b, a = butter(order, wn, btype='low')
    if len(data) < 3 * max(len(a), len(b)):
        return data.copy()
    return filtfilt(b, a, data)


# ──────────────────────────────────────────────────────────────────
# 第 3 节：步态事件检测 — 足跟着地 & 离地
# ──────────────────────────────────────────────────────────────────

def detect_heel_strikes(pressure_total, fs=FS,
                        min_step_sec=0.25,
                        max_step_sec=2.0):
    """
    从单脚压力总和信号中检测足跟着地时刻（Heel Strike）。
    采用自适应阈值：均值 + 0.5 × 标准差，适应不同信号强度。
    """
    signal = pressure_total.copy()
    if np.max(signal) < 10:
        return np.array([], dtype=int)

    # 自适应阈值
    threshold = np.mean(signal) + 0.5 * np.std(signal)

    above = signal > threshold
    rising_edges = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    if len(rising_edges) == 0:
        return np.array([], dtype=int)

    min_gap = int(min_step_sec * fs)
    filtered = [rising_edges[0]]
    for idx in rising_edges[1:]:
        if idx - filtered[-1] >= min_gap:
            filtered.append(idx)
    return np.array(filtered)


# ──────────────────────────────────────────────────────────────────
# 第 4 节：6 项步态参数提取（倒立摆模型 + 压力对称性）
# ──────────────────────────────────────────────────────────────────

def extract_parameters(df, leg_length_m, calib_factor=1.0, fs=FS):
    """
    从 DataFrame 提取 6 项步态参数。

    算法说明：
      步频：检测左右足跟着地时刻，取步间隔中位数换算为步/分钟。
      步长：对垂直加速度做双重积分得垂直位移，用倒立摆公式
            step_length = 2 × sqrt(2Lh - h²)
            其中 L 为腿长，h 为单步垂直振幅中位数。
      步速：步长 / 平均步时。
      不对称度：左右脚压力积分之差，归一化到 [0, 1]。
      稳定性：水平加速度的变异系数，映射到 [0, 1]。
      对称性：左右步时中位数的比值。

    返回：
      dict，含上述 6 项及辅助信息；失败时含 'error' 键。
    """
    # ── 检查必须列 ──
    required = PRESSURE_COLS_R + PRESSURE_COLS_L + ACC_COLS_R + ACC_COLS_L
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {'error': f'缺少列: {missing[:3]}...'}

    n = len(df)
    if n < fs * 2:                          # 至少需要 2 秒数据
        return {'error': f'数据太短（{n} 点 < {fs * 2}）'}

    # ── 步骤 1：压力总和（三点合并为单脚总压力） ──
    p_r = df[PRESSURE_COLS_R].sum(axis=1).values.astype(np.float64)
    p_l = df[PRESSURE_COLS_L].sum(axis=1).values.astype(np.float64)

    # ── 步骤 2：足跟着地检测 ──
    hs_r = detect_heel_strikes(p_r, fs)  # 把默认的 0.08 改成 0.25
    hs_l = detect_heel_strikes(p_l, fs)

    if len(hs_r) < 2 and len(hs_l) < 2:
        return {'error': '无法检测到足够的步态周期'}

    # ── 步骤 3：步频计算 ──
    # step_times = []
    # if len(hs_r) >= 2:
    #     step_times.extend(np.diff(hs_r) / fs)
    # if len(hs_l) >= 2:
    #     step_times.extend(np.diff(hs_l) / fs)
    # step_times = np.array(step_times)
    # step_times = step_times[(step_times > 0.25) & (step_times < 2.0)]
    #
    # if len(step_times) < 1:
    #     return {'error': '步态周期检测失败（步时全部异常）'}
    #
    # mean_step_time = float(np.median(step_times))
    # cadence_spm = 60.0 / mean_step_time
    # ── 步骤 3：步频计算（直接使用总步数和总时间） ──
    all_hs = np.sort(np.concatenate([hs_r, hs_l]))
    n_steps_total = len(all_hs)
    duration = n / fs
    if duration <= 0:
        return {'error': '时长为零'}
    cadence_spm = n_steps_total / duration * 60.0  # 步/分

    # 仍计算 mean_step_time（用于其他辅助信息，但不再用于步频）
    step_times = []
    if len(hs_r) >= 2:
        step_times.extend(np.diff(hs_r) / fs)
    if len(hs_l) >= 2:
        step_times.extend(np.diff(hs_l) / fs)
    step_times = np.array(step_times)
    step_times = step_times[(step_times > 0.25) & (step_times < 2.0)]
    mean_step_time = float(np.median(step_times)) if len(step_times) > 0 else duration / n_steps_total

    # ── 步骤 4：步长估计（倒立摆模型） ──
    # 4a. 取左右垂直加速度均值，去重力基线
    acc_z = (df['acc_z_R'].values + df['acc_z_L'].values) / 2.0
    acc_z_detrend = acc_z - np.mean(acc_z)
    acc_z_filt = _lowpass(acc_z_detrend, cutoff=6.0, fs=fs)

    # 4b. 双重积分：加速度 → 速度 → 位移，每步用高通去漂移
    vel = np.cumsum(acc_z_filt) / fs
    vel = vel - _lowpass(vel, cutoff=0.5, fs=fs)
    disp = np.cumsum(vel) / fs
    disp = disp - _lowpass(disp, cutoff=0.5, fs=fs)

    # 4c. 每一步内的垂直振幅 h
    all_hs = np.sort(np.concatenate([hs_r, hs_l]))
    h_list = []
    for i in range(len(all_hs) - 1):
        seg = disp[all_hs[i]:all_hs[i + 1]]
        if len(seg) > 5:
            h_list.append(np.max(seg) - np.min(seg))

    if len(h_list) > 0:
        h_median = float(np.median(h_list))
        h_median = np.clip(h_median, 0.001, leg_length_m * 0.5)
    else:
        h_median = 0.02      # 无法估计时取经验值

    # 4d. 倒立摆公式
    val = 2 * leg_length_m * h_median - h_median ** 2
    step_length = 2.0 * np.sqrt(val) if val > 0 else leg_length_m * 0.7

    # 4e. 乘以校准系数并限幅
    step_length = float(np.clip(step_length * calib_factor, 0.1, 2.5))

    # ── 步骤 5：步速 ──
    step_speed = float(np.clip(step_length * cadence_spm / 60.0, 0.05, 4.0))

    # ── 步骤 6：左右负载不对称度 ──
    # 0 = 完全对称，1 = 全部压在一侧
    total_r = float(np.sum(p_r))
    total_l = float(np.sum(p_l))
    total_all = total_r + total_l
    step_width_ratio = abs(total_r / total_all - 0.5) * 2 if total_all > 0 else 0.0

    # ── 步骤 7：稳定性评分（基于水平加速度变异系数） ──
    acc_horiz = np.sqrt(
        ((df['acc_x_R'].values + df['acc_x_L'].values) / 2) ** 2 +
        ((df['acc_y_R'].values + df['acc_y_L'].values) / 2) ** 2
    )
    cv_acc = float(np.std(acc_horiz) / (np.mean(acc_horiz) + 1e-6))
    stability_score = float(np.clip(1.0 - cv_acc * 0.5, 0.0, 1.0))

    # ── 步骤 8：对称性评分（左右步时差异） ──f
    if len(hs_r) >= 2 and len(hs_l) >= 2:
        st_r = np.median(np.diff(hs_r) / fs)
        st_l = np.median(np.diff(hs_l) / fs)
        symmetry_score = float(
            np.clip(min(st_r, st_l) / (max(st_r, st_l) + 1e-6), 0.0, 1.0))
    else:
        symmetry_score = 0.5   # 单侧数据不足时给中性值

    return {
        'step_length_m':    step_length,
        'step_speed_ms':    step_speed,
        'cadence_spm':      float(cadence_spm),
        'step_width_ratio': float(step_width_ratio),
        'stability_score':  float(stability_score),
        'symmetry_score':   float(symmetry_score),
        'mean_step_time_s': float(mean_step_time),
        'n_steps_R':        int(len(hs_r)),
        'n_steps_L':        int(len(hs_l)),
        'n_steps_total':    int(len(all_hs)),
        'duration_sec':     float(n / fs),
    }


def extract_parameters_from_file(csv_path, leg_length_m,
                                  calib_factor=1.0, fs=FS):
    """
    从 CSV 文件读取数据并调用 extract_parameters()。
    捕获读取异常，返回含 'error' 键的字典。
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        return {'error': f'文件读取失败: {e}'}

    result = extract_parameters(df, leg_length_m, calib_factor, fs)
    result['file'] = os.path.basename(csv_path)
    return result


# ──────────────────────────────────────────────────────────────────
# 第 5 节：步长校准 — 使用 1330cm 已知距离行走数据
# ──────────────────────────────────────────────────────────────────

def calibrate_step_length_factor(csv_path, true_distance_m,
                                  leg_length_m, fs=FS):
    """
    利用已知距离的行走数据（通常为 13.30 m 往返）计算步长校准系数 K。

    原理：
      K = 真实距离 / （原始估计步长 × 总步数）
      后续所有步长估计乘以 K，消除个体体型差异和传感器误差。

    K 被限制在 [0.3, 3.0] 防止极端值。
    """
    params = extract_parameters_from_file(
        csv_path, leg_length_m, calib_factor=1.0, fs=fs)

    if 'error' in params:
        print(f"  [校准警告] {params['error']}，使用 K=1.0")
        return 1.0

    raw_step_length = params['step_length_m']
    n_steps = params['n_steps_total']

    if n_steps < 2 or raw_step_length < 0.01:
        print(f"  [校准警告] 步数={n_steps} 步长={raw_step_length:.3f}，使用 K=1.0")
        return 1.0

    estimated_distance = raw_step_length * n_steps
    factor = float(np.clip(true_distance_m / estimated_distance, 0.3, 3.0))
    print(f"  [校准] 估计={estimated_distance:.2f}m  真实={true_distance_m}m  K={factor:.4f}")
    return factor


# ──────────────────────────────────────────────────────────────────
# 第 6 节：GaitNormModel — 统计正常模型
# ──────────────────────────────────────────────────────────────────

class GaitNormModel:
    """
    统计正常模型：用健康受试者的步态参数建立均值/标准差参考范围。

    判断逻辑：
      Z 分数 = (测量值 - 均值) / 标准差
      |Z| > n_std（默认 2.0）→ 该参数异常
      ≥2 项参数异常 → 总体异常

    与 GaitDeepModel 保持相同的 fit / score / save / load 接口，
    可在 inference.py 中无缝互换。
    """

    PARAM_KEYS   = PARAM_KEYS
    PARAM_LABELS = PARAM_LABELS

    def __init__(self, n_std=2.0):
        self.n_std = n_std
        self.stats = {}           # 每个参数的统计量字典
        self.is_fitted = False

    # ── 训练：计算各参数统计量 ──
    def fit(self, records):
        """
        records: list of dict，每个 dict 来自 extract_parameters()。
        计算每个参数的均值、标准差、分位数等并存入 self.stats。
        """
        df = pd.DataFrame(records)
        for key in self.PARAM_KEYS:
            if key not in df.columns:
                continue
            vals = df[key].dropna().values.astype(float)
            if len(vals) < 2:
                continue
            self.stats[key] = {
                'mean':   float(np.mean(vals)),
                'std':    float(np.std(vals, ddof=1)),
                'median': float(np.median(vals)),
                'q05':    float(np.percentile(vals, 5)),
                'q25':    float(np.percentile(vals, 25)),
                'q75':    float(np.percentile(vals, 75)),
                'q95':    float(np.percentile(vals, 95)),
                'min':    float(np.min(vals)),
                'max':    float(np.max(vals)),
                'n':      int(len(vals)),
            }
        self.is_fitted = True
        print(f"  [统计模型] 拟合完成，{len(self.stats)} 个参数")

    # ── 推理：计算 Z 分数并判断异常 ──
    def score(self, record):
        """
        record: dict，含步态参数键值对。

        返回：
          params            各参数评分详情（value / z_score / is_abnormal 等）
          n_abnormal_params 异常参数个数
          overall_abnormal  是否总体异常（≥2 项偏离）
          abnormality_level 四级文字描述
        """
        if not self.is_fitted:
            raise RuntimeError("模型未拟合，请先调用 fit()")

        result = {}
        for key in self.PARAM_KEYS:
            val = record.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                result[key] = {'value': None, 'z_score': None, 'is_abnormal': None}
                continue

            val = float(val)
            if key in self.stats:
                s = self.stats[key]
                z   = (val - s['mean']) / (s['std'] + 1e-8)
                dev = (val - s['mean']) / (abs(s['mean']) + 1e-8) * 100
                result[key] = {
                    'value':         val,
                    'z_score':       float(z),
                    'is_abnormal':   bool(abs(z) > self.n_std),
                    'deviation_pct': float(dev),
                    'normal_mean':   s['mean'],
                    'normal_std':    s['std'],
                    'normal_q25':    s['q25'],
                    'normal_q75':    s['q75'],
                    'normal_q05':    s['q05'],
                    'normal_q95':    s['q95'],
                }
            else:
                result[key] = {'value': val, 'z_score': None, 'is_abnormal': None}

        n_abn = sum(1 for v in result.values() if v.get('is_abnormal') is True)
        return {
            'params':            result,
            'n_abnormal_params': n_abn,
            'overall_abnormal':  n_abn >= 2,
            'abnormality_level': self._level(n_abn),
        }

    def _level(self, n_abn):
        if n_abn == 0:  return '正常'
        if n_abn == 1:  return '轻度偏差'
        if n_abn <= 3:  return '中度异常'
        return '明显异常'

    # ── 持久化 ──
    def save(self, path):
        state = {'stats': self.stats, 'n_std': self.n_std, 'model_type': 'stat'}
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"  [统计模型] 已保存: {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.stats     = state['stats']
        self.n_std     = state.get('n_std', 2.0)
        self.is_fitted = True
        print(f"  [统计模型] 已加载: {path}")


# ──────────────────────────────────────────────────────────────────
# 第 7 节：GaitDeepModel — 深度学习模型
# ──────────────────────────────────────────────────────────────────

# ── 7a. 数据集类（PyTorch Dataset） ──

class GaitDataset(Dataset):
    """
    封装滑动窗口数组、分类标签、参数标签，供 DataLoader 使用。
    params 可选，仅在多任务训练时使用。
    """
    def __init__(self, windows, labels, params=None):
        self.windows = torch.FloatTensor(windows)
        self.labels  = torch.LongTensor(labels)
        self.params  = torch.FloatTensor(params) if params is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x, y = self.windows[idx], self.labels[idx]
        if self.params is not None:
            return x, y, self.params[idx]
        return x, y


def extract_windows(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    从 DataFrame 中提取滑动窗口，形状为 (N, window_size, 24)。
    若数据长度不足或列缺失，返回空数组。
    """
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        return np.empty((0, window_size, len(SENSOR_COLS)), dtype=np.float32)

    data = df[SENSOR_COLS].values.astype(np.float32)
    if len(data) < window_size:
        return np.empty((0, window_size, len(SENSOR_COLS)), dtype=np.float32)

    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    return np.array(windows, dtype=np.float32)


# ── 7b. 卷积模块 ──

class ConvBlock(nn.Module):
    """
    单个 1D 卷积块：Conv1d → BatchNorm → ReLU → Dropout。
    使用 padding=kernel_size//2 保持时序长度不变。
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dropout=0.2):
        super().__init__()
        self.conv    = nn.Conv1d(in_ch, out_ch, kernel_size,
                                 padding=kernel_size // 2)
        self.bn      = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))


# ── 7c. 主网络架构：1D-CNN + BiLSTM + Attention + 多任务输出 ──

class GaitNet(nn.Module):
    """
    网络结构概述：
      输入  (B, T=300, C=24)
        → 转置 (B, C, T)
        → 3 层 ConvBlock + MaxPool  → 特征图 (B, 256, T/8)
        → 转置 (B, T/8, 256)
        → 2 层双向 LSTM            → 时序特征 (B, T/8, 256)
        → Attention 加权求和        → 上下文向量 (B, 256)
        → 分类头：FC → 正常/异常 logits (B, 2)
        → 回归头：FC → 6 项参数预测 (B, 6)

    多任务学习：同时优化分类损失（CrossEntropy）和参数回归损失（MSE），
    总损失 = cls_loss + 0.5 × reg_loss。
    """
    def __init__(self, n_channels=24, n_classes=2, n_params=6,
                 cnn_channels=(64, 128, 256), lstm_hidden=128,
                 lstm_layers=2, dropout=0.3):
        super().__init__()
        self.n_channels = n_channels

        # CNN 特征提取：逐层扩大感受野
        layers = []
        in_ch = n_channels
        for out_ch in cnn_channels:
            layers.append(ConvBlock(in_ch, out_ch, dropout=dropout))
            layers.append(nn.MaxPool1d(2))   # 每层时序长度减半
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # BiLSTM 时序建模：双向捕捉前后文步态节律
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        out_size = lstm_hidden * 2    # 双向拼接

        # Attention：对不同时刻的重要性加权，聚合为单一向量
        self.attention = nn.Sequential(
            nn.Linear(out_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 分类头：判断正常 / 异常
        self.classifier = nn.Sequential(
            nn.Linear(out_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        # 回归头：预测 6 项步态参数（在标准化空间中）
        self.regressor = nn.Sequential(
            nn.Linear(out_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_params),
        )

    def forward(self, x):
        # x: (B, T, C) → CNN 需要 (B, C, T)
        x = x.transpose(1, 2)
        x = self.cnn(x)                             # (B, 256, T')
        x = x.transpose(1, 2)                       # (B, T', 256)

        lstm_out, _ = self.lstm(x)                  # (B, T', 256)

        # Attention 加权聚合
        attn = F.softmax(self.attention(lstm_out), dim=1)
        ctx  = torch.sum(lstm_out * attn, dim=1)    # (B, 256)

        return self.classifier(ctx), self.regressor(ctx)


# ── 7d. GaitDeepModel 封装类 ──

class GaitDeepModel:
    """
    深度学习模型封装，接口与 GaitNormModel 完全一致。

    fit()   : 从 records 列表（含 file_path / is_normal / 参数）训练网络。
    score() : 输入 DataFrame / CSV路径 / dict，返回与统计模型相同格式的评分。
    save() / load() : 序列化整个模型（网络权重 + 两个 Scaler + 统计参数）。
    """

    PARAM_KEYS   = PARAM_KEYS
    PARAM_LABELS = PARAM_LABELS

    def __init__(self, n_std=2.0, device=None):
        self.n_std        = n_std
        self.device       = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = None
        self.scaler       = StandardScaler()    # 对原始传感器窗口做归一化
        self.param_scaler = StandardScaler()    # 对步态参数做归一化（供回归头）
        self.stats        = {}                  # 正常样本参数统计（用于 score() 中 Z 分数）
        self.is_fitted    = False

    # ── 数据准备：读取 CSV → 提取滑窗 → 组装标签 ──
    def _prepare_data(self, records, for_training=True):
        all_w, all_y, all_p, all_g = [], [], [], []
        for rec in records:
            # 优先从内存 DataFrame 读取，否则从 file_path 读 CSV
            if 'raw_df' in rec:
                df = rec['raw_df']
            elif 'file_path' in rec and os.path.exists(rec['file_path']):
                try:
                    df = pd.read_csv(rec['file_path'], encoding='utf-8-sig')
                except Exception as e:
                    print(f"  [跳过] 读取失败: {rec['file_path']}: {e}")
                    continue
            else:
                continue

            wins = extract_windows(df)
            if len(wins) == 0:
                continue
            all_w.append(wins)
            # 【新增】记录每个窗口对应的 subject_id，供按受试者切分
            all_g.extend([rec.get('subject_id', 'unknown')] * len(wins))

            if for_training:
                label = 0 if rec.get('is_normal', True) else 1
                all_y.extend([label] * len(wins))
                all_p.extend(
                    [[rec.get(k, np.nan) for k in PARAM_KEYS]] * len(wins))

        if not all_w:
            return None, None, None, None

        X = np.concatenate(all_w, axis=0)
        y = np.array(all_y) if all_y else None
        p = np.array(all_p, dtype=np.float32) if all_p else None
        g = np.array(all_g)
        return X, y, p, g

    def _compute_stats(self, normal_records):
        """从正常样本计算参数统计量，供 score() 中计算 Z 分数。"""
        df = pd.DataFrame(normal_records)
        for key in PARAM_KEYS:
            if key not in df.columns:
                continue
            vals = df[key].dropna().values
            if len(vals) < 2:
                continue
            self.stats[key] = {
                'mean':   float(vals.mean()),
                'std':    float(vals.std()),
                'median': float(np.median(vals)),
                'q05':    float(np.percentile(vals, 5)),
                'q25':    float(np.percentile(vals, 25)),
                'q75':    float(np.percentile(vals, 75)),
                'q95':    float(np.percentile(vals, 95)),
                'min':    float(vals.min()),
                'max':    float(vals.max()),
                'n':      len(vals),
            }

    # ── 训练 ──
    def fit(self, records, epochs=50, batch_size=32, lr=1e-3,
            val_split=0.15, early_stop_patience=10, verbose=True):
        print("[深度模型] 准备训练数据...")
        # 1. 获取数据及受试者分组
        X, y, params, groups = self._prepare_data(records, for_training=True)

        if X is None or len(X) < 10:
            raise ValueError(f"有效训练样本不足（{0 if X is None else len(X)}）")

        # 计算正常样本统计量（用于推理时的 Z 分数）
        normal_records = [r for r in records if r.get('is_normal', True)]
        self._compute_stats(normal_records)

        n_normal = int((y == 0).sum())
        n_abn    = int((y == 1).sum())
        if n_abn == 0 or n_normal == 0:
            raise ValueError(
                f"训练需要正常和异常两类样本，当前 正常={n_normal}, 异常={n_abn}")

        # 2.【核心修改】使用 GroupShuffleSplit 按受试者划分，
        #    保证同一个人的窗口不会同时出现在训练集和验证集
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        tr_idx, val_idx = next(gss.split(X, y, groups))

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        p_tr, p_val = params[tr_idx], params[val_idx]

        # 3.【核心修改】Scaler 仅在训练集上 fit，防止数据泄漏
        n_s, n_t, n_c = X_tr.shape
        self.scaler.fit(X_tr.reshape(-1, n_c))
        X_tr = self.scaler.transform(X_tr.reshape(-1, n_c)).reshape(n_s, n_t, n_c)
        X_val = self.scaler.transform(X_val.reshape(-1, n_c)).reshape(-1, n_t, n_c)

        p_tr = np.nan_to_num(p_tr, nan=0.0)
        p_val = np.nan_to_num(p_val, nan=0.0)
        self.param_scaler.fit(p_tr)
        p_tr = self.param_scaler.transform(p_tr)
        p_val = self.param_scaler.transform(p_val)

        # 4. 组装 DataLoader
        train_loader = DataLoader(
            GaitDataset(X_tr, y_tr, p_tr), batch_size=batch_size,
            shuffle=True, drop_last=False)
        val_loader = DataLoader(
            GaitDataset(X_val, y_val, p_val), batch_size=batch_size)

        # 类权重：异常样本少时自动上调权重（防止模型只预测正常）
        weight = torch.FloatTensor(
            [1.0, max(n_normal / max(n_abn, 1), 1.0)]).to(self.device)

        self.model = GaitNet(n_channels=n_c).to(self.device)
        optimizer  = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        cls_loss_fn = nn.CrossEntropyLoss(weight=weight)
        reg_loss_fn = nn.MSELoss()

        print(f"[深度模型] 设备: {self.device}")
        print(f"[深度模型] 训练={len(X_tr)}  验证={len(X_val)}  "
              f"正常={n_normal}  异常={n_abn}")

        best_val, best_state, patience_cnt = float('inf'), None, 0

        for ep in range(epochs):
            # ── 训练阶段 ──
            self.model.train()
            tr_loss = 0.0
            for x, yt, pt in train_loader:
                x, yt, pt = x.to(self.device), yt.to(self.device), pt.to(self.device)
                optimizer.zero_grad()
                logits, p_pred = self.model(x)
                loss = cls_loss_fn(logits, yt) + 0.5 * reg_loss_fn(p_pred, pt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= max(len(train_loader), 1)

            # ── 验证阶段 ──
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for x, yt, pt in val_loader:
                    x, yt, pt = x.to(self.device), yt.to(self.device), pt.to(self.device)
                    logits, p_pred = self.model(x)
                    val_loss += (cls_loss_fn(logits, yt) +
                                 0.5 * reg_loss_fn(p_pred, pt)).item()
                    correct += (logits.argmax(1) == yt).sum().item()
                    total   += yt.size(0)
            val_loss /= max(len(val_loader), 1)
            val_acc   = correct / max(total, 1)
            scheduler.step(val_loss)

            if verbose and (ep + 1) % 5 == 0:
                print(f"  Epoch {ep+1:3d}: train={tr_loss:.4f}  "
                      f"val={val_loss:.4f}  acc={val_acc:.3f}")

            # 早停：保存最佳权重
            if val_loss < best_val:
                best_val = val_loss
                patience_cnt = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= early_stop_patience:
                    print(f"  早停于 epoch {ep + 1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_fitted = True
        print(f"[深度模型] 训练完成  best_val_loss={best_val:.4f}")

    # ── 推理 ──
    def score(self, input_data):
        """
        input_data 支持三种形式：
          - pd.DataFrame : 直接使用
          - str          : CSV 文件路径
          - dict         : 含 'raw_df' / 'file_path'，或纯参数字典（退化为统计判断）
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")

        if isinstance(input_data, pd.DataFrame):
            df = input_data
        elif isinstance(input_data, str):
            df = pd.read_csv(input_data, encoding='utf-8-sig')
        elif isinstance(input_data, dict):
            if 'raw_df' in input_data:
                df = input_data['raw_df']
            elif 'file_path' in input_data:
                df = pd.read_csv(input_data['file_path'], encoding='utf-8-sig')
            else:
                # 退化：只有参数字典，用统计 Z 分数判断
                return self._score_from_params(input_data)
        else:
            raise ValueError("score 输入需为 DataFrame / 字典 / 文件路径")

        windows = extract_windows(df)
        if len(windows) == 0:
            return self._empty_result()

        n, t, c = windows.shape
        X   = self.scaler.transform(windows.reshape(-1, c)).reshape(n, t, c)
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            logits, p_pred = self.model(X_t)
            probs = F.softmax(logits, dim=1)

        # 聚合所有窗口：异常概率取均值，参数预测取均值后反归一化
        abn_prob = float(probs[:, 1].mean().item())
        p_real   = self.param_scaler.inverse_transform(
            p_pred.mean(dim=0).cpu().numpy().reshape(1, -1))[0]

        result = {}
        for i, key in enumerate(PARAM_KEYS):
            val = float(p_real[i])
            if key in self.stats:
                s   = self.stats[key]
                z   = (val - s['mean']) / (s['std'] + 1e-6)
                dev = (val - s['mean']) / (abs(s['mean']) + 1e-6) * 100
                result[key] = {
                    'value': val, 'z_score': float(z),
                    'is_abnormal': bool(abs(z) > self.n_std),
                    'deviation_pct': float(dev),
                    'normal_mean': s['mean'], 'normal_std': s['std'],
                    'normal_q25': s['q25'],   'normal_q75': s['q75'],
                    'normal_q05': s['q05'],   'normal_q95': s['q95'],
                }
            else:
                result[key] = {'value': val, 'z_score': None, 'is_abnormal': None}

        n_abn = sum(1 for v in result.values() if v.get('is_abnormal'))
        return {
            'params':            result,
            'n_abnormal_params': n_abn,
            'overall_abnormal':  bool(abn_prob > 0.5 or n_abn >= 2),
            'abnormality_level': self._abnormality_level(abn_prob, n_abn),
            'abnormal_prob':     abn_prob,
        }

    def _score_from_params(self, params_dict):
        """退化路径：仅有参数字典时，用正常样本统计做 Z 分数判断。"""
        result = {}
        for key in PARAM_KEYS:
            val = params_dict.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                result[key] = {'value': None, 'z_score': None, 'is_abnormal': None}
                continue
            if key in self.stats:
                s   = self.stats[key]
                z   = (val - s['mean']) / (s['std'] + 1e-6)
                dev = (val - s['mean']) / (abs(s['mean']) + 1e-6) * 100
                result[key] = {
                    'value': float(val), 'z_score': float(z),
                    'is_abnormal': bool(abs(z) > self.n_std),
                    'deviation_pct': float(dev),
                    'normal_mean': s['mean'], 'normal_std': s['std'],
                    'normal_q25': s['q25'],   'normal_q75': s['q75'],
                    'normal_q05': s['q05'],   'normal_q95': s['q95'],
                }
            else:
                result[key] = {'value': float(val), 'z_score': None, 'is_abnormal': None}

        n_abn = sum(1 for v in result.values() if v.get('is_abnormal'))
        return {
            'params':            result,
            'n_abnormal_params': n_abn,
            'overall_abnormal':  n_abn >= 2,
            'abnormality_level': self._abnormality_level(
                0.5 if n_abn >= 2 else 0.2, n_abn),
        }

    def _empty_result(self):
        return {
            'params':            {k: {'value': None, 'z_score': None,
                                      'is_abnormal': None} for k in PARAM_KEYS},
            'n_abnormal_params': 0,
            'overall_abnormal':  False,
            'abnormality_level': '数据不足',
            'abnormal_prob':     0.0,
        }

    @staticmethod
    def _abnormality_level(prob, n_abn):
        if prob < 0.3 and n_abn == 0: return '正常'
        if prob < 0.5 and n_abn <= 1: return '轻度偏差'
        if prob < 0.7 or n_abn <= 2:  return '中度异常'
        return '明显异常'

    # ── 持久化 ──
    def save(self, path):
        state = {
            'model_state':  self.model.state_dict() if self.model else None,
            'scaler':       self.scaler,
            'param_scaler': self.param_scaler,
            'stats':        self.stats,
            'n_std':        self.n_std,
            'model_type':   'deep',
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[深度模型] 已保存: {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        if state.get('model_type') != 'deep':
            raise ValueError("这不是深度学习模型文件")
        self.scaler       = state['scaler']
        self.param_scaler = state['param_scaler']
        self.stats        = state['stats']
        self.n_std        = state['n_std']
        self.model        = GaitNet().to(self.device)
        self.model.load_state_dict(state['model_state'])
        self.model.eval()
        self.is_fitted = True
        print(f"[深度模型] 已加载: {path}")


def load_gait_model(path):
    """
    根据 pkl 文件内容自动判断并加载统计模型或深度学习模型。
    返回 (model, model_type)，model_type 为 'stat' 或 'deep'。
    """
    with open(path, 'rb') as f:
        state = pickle.load(f)
    if isinstance(state, dict) and state.get('model_type') == 'deep':
        m = GaitDeepModel()
        m.load(path)
        return m, 'deep'
    else:
        m = GaitNormModel()
        m.load(path)
        return m, 'stat'


# ──────────────────────────────────────────────────────────────────
# 第 8 节：受试者信息加载 & 文件名自动分类
# ──────────────────────────────────────────────────────────────────

def _parse_height_weight(text):
    """
    从文本里解析身高(cm)/体重(kg)。
    兼容：JSON 键值、身高=170 体重=65、纯数字序列。
    返回 (height, weight, id) 或 None。
    """
    sid = None
    m = re.search(r'["\']?id["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-\.]+)', text)
    if m:
        sid = m.group(1)
    height = weight = None
    m = re.search(r'(?:身高|height)[^\d]*?(\d{2,3}(?:\.\d+)?)', text)
    if m:
        height = float(m.group(1))
    m = re.search(r'(?:体重|weight)[^\d]*?(\d{2,3}(?:\.\d+)?)', text)
    if m:
        weight = float(m.group(1))
    if height is None:
        all_nums = [float(x) for x in
                    re.findall(r'\d+(?:\.\d+)?', text)]
        h_cands = [n for n in all_nums if 100 <= n <= 220]
        w_cands = [n for n in all_nums if 30 <= n <= 150]
        if h_cands:
            height = h_cands[0]
            for w in w_cands:
                if w != height:
                    weight = w
                    break
    if height is None:
        return None
    return height, weight, sid


def load_subject_info(subject_folder):
    """
    读取受试者信息，按优先级：
      1. subject_info.json
      2. 新建文本文档.txt → subject_info.txt → info.txt
      3. 任意 txt 兜底
      4. 默认 170cm / 65kg
    自动计算腿长：腿长 = 身高 × 0.53。
    """
    default_id = os.path.basename(subject_folder)
    info = None

    json_path = os.path.join(subject_folder, 'subject_info.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

    if info is None:
        for name in ['新建文本文档.txt', 'subject_info.txt', 'info.txt']:
            p = os.path.join(subject_folder, name)
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8-sig') as f:
                    info = _parse_height_weight(f.read())
                if info:
                    break
        if not info:
            for p in glob.glob(os.path.join(subject_folder, '*.txt')):
                with open(p, 'r', encoding='utf-8-sig') as f:
                    info = _parse_height_weight(f.read())
                if info:
                    break

    if info is None:
        info = {'id': default_id, 'height_cm': 170.0, 'weight_kg': 65.0}
        print("  [警告] 未找到受试者信息，使用默认 170cm/65kg")
    elif isinstance(info, dict):
        info.setdefault('id', default_id)
    else:  # _parse_height_weight 返回的元组
        height, weight, sid = info
        info = {'id': sid or default_id,
                'height_cm': height, 'weight_kg': weight or 65.0}

    h = info.get('height_cm') or info.get('height') or 170
    w = info.get('weight_kg') or info.get('weight') or 65
    info['height_cm'] = float(h)
    info['weight_kg'] = float(w)
    info['leg_length_m'] = info['height_cm'] * 0.53 / 100.0
    return info


def match_action_type(filename):
    """
    根据文件名关键词推断动作类型和正常/异常标签。

    规则（优先级从上到下，先匹配先返回）：
      含「双任务」→ 认知双任务，视为异常（is_normal=False）
      含「1330」  → 校准文件（is_normal=True，训练时单独处理）
      含「下楼」/「上楼」/「楼梯」→ 楼梯行走，正常
      含「停顿」/「转弯」/「直线」→ 各类正常行走

    返回：(action_type: str, is_normal: bool)
    """
    fname = filename.lower()
    mapping = [
        ('1330',       '1330cm',       True),
        ('双任务下楼', '双任务下楼梯', False),
        ('双任务上楼', '双任务上楼梯', False),
        ('双任务楼梯', '双任务上楼梯', False),
        ('双任务',     '双任务直线',   False),
        ('下楼',       '下楼梯',       True),
        ('上楼',       '上楼梯',       True),
        ('楼梯',       '上楼梯',       True),
        ('停顿',       '直线停顿',     True),
        ('停下',       '直线停顿',     True),
        ('停止',       '起步停止',     True),
        ('起步',       '起步停止',     True),
        ('转弯',       '直角转弯',     True),
        ('直线',       '直线来回',     True),
        ('来回',       '直线来回',     True),
        ('坐站走',     '坐站走',       True),
        ('坐站',       '坐站走',       True),
    ]
    for keyword, action, is_normal in mapping:
        if keyword in fname:
            return action, is_normal
    return '未知', False


def _iter_subject_folders(root):
    """
    扫描受试者文件夹，支持两层目录结构：
      root/受试者/*_filtered.csv        → 一层
      root/组/受试者/*_filtered.csv     → 两层
    返回受试者文件夹路径列表。
    """
    folders = []
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d):
            continue
        if glob.glob(os.path.join(d, '*_filtered.csv')):
            folders.append(d)
            continue
        folders.extend(
            s for s in sorted(glob.glob(os.path.join(d, '*')))
            if os.path.isdir(s) and glob.glob(os.path.join(s, '*_filtered.csv'))
        )
    return folders


# ──────────────────────────────────────────────────────────────────
# 第 9 节：train_stat() — 统计模型训练流水线
# ──────────────────────────────────────────────────────────────────

def _process_subject_for_training(subject_folder, dataset_root=None):
    """
    处理单个受试者文件夹：
      1. 读取受试者信息（身高/腿长）
      2. 找到 1330cm 校准文件，计算步长校准系数
      3. 遍历所有 CSV，提取步态参数，按正常/异常分类
    返回包含所有记录的字典，失败返回 None。
    """
    subject_info = load_subject_info(subject_folder)
    if dataset_root:
        rel = os.path.relpath(subject_folder, dataset_root)
        unique_id = rel.replace(os.sep, '_').replace('/', '_')
    else:
        unique_id = os.path.basename(subject_folder)
    # 【修改】强制使用「组别_编号」作为全局唯一 ID，
    # 否则 老年男\1 和 模拟男\1 会撞成同一个 id=1
    sid = unique_id
    subject_info['id'] = sid
    leg_length  = subject_info['leg_length_m']

    print(f"\n{'='*55}")
    print(f"  受试者: {sid}  身高: {subject_info.get('height_cm')}cm  "
          f"腿长: {leg_length:.3f}m")
    print(f"{'='*55}")

    csv_files = sorted(glob.glob(os.path.join(subject_folder, '*_filtered.csv')))
    if not csv_files:
        print(f"  [跳过] 未找到预处理文件，请先运行第一步预处理")
        return None

    # 步骤 1：找校准文件
    calib_files = [f for f in csv_files
                   if '1330' in os.path.basename(f).lower()]
    calibration_factor = 1.0
    if calib_files:
        print(f"  [校准] 使用: {os.path.basename(calib_files[0])}")
        calibration_factor = calibrate_step_length_factor(
            calib_files[0], true_distance_m=13.30,
            leg_length_m=leg_length, fs=FS)
    else:
        print("  [警告] 未找到 1330cm 校准文件，使用 K=1.0")

    normal_records, abnormal_records, all_records = [], [], []

    # 步骤 2：逐文件提取参数
    for csv_path in csv_files:
        fname         = os.path.basename(csv_path)
        action_type, is_normal = match_action_type(fname)
        tag = '✓正常' if is_normal else '✗异常'
        print(f"  ├── {fname}  [{action_type}] {tag}")

        params = extract_parameters_from_file(
            csv_path, leg_length, calibration_factor, FS)

        if 'error' in params:
            print(f"      ⚠ {params['error']}")
            continue

        params.update({
            'subject_id':   sid,
            'action_type':  action_type,
            'is_normal':    is_normal,
            'calib_factor': calibration_factor,
            'file_path':    csv_path,
        })
        all_records.append(params)

        if is_normal and action_type not in ('1330cm', '未知'):
            normal_records.append(params)
            print(f"      → SL={params['step_length_m']:.3f}m  "
                  f"Speed={params['step_speed_ms']:.2f}m/s  "
                  f"Cad={params['cadence_spm']:.0f}spm")
        elif not is_normal:
            abnormal_records.append(params)

    return {
        'subject_info':       subject_info,
        'calibration_factor': calibration_factor,
        'normal_records':     normal_records,
        'abnormal_records':   abnormal_records,
        'all_records':        all_records,
    }


def _loso_validation(subject_results, n_std=2.0):
    """
    留一法（LOSO）验证统计模型：
      每次留出一名受试者作为测试集，其余人训练统计模型，
      统计测试集中正常样本被误判为异常的比率（假阳率）。
    需要至少 3 名受试者。
    """
    if len(subject_results) < 3:
        print("[LOSO] 受试者不足 3 人，跳过")
        return {}

    overall_fp = []
    for i, held in enumerate(subject_results):
        train_records = [
            rec for j, other in enumerate(subject_results)
            if i != j for rec in other['normal_records']
        ]
        if not train_records or not held['normal_records']:
            continue
        m = GaitNormModel(n_std=n_std)
        m.fit(train_records)
        for rec in held['normal_records']:
            overall_fp.append(int(m.score(rec)['overall_abnormal']))

    fpr = float(np.mean(overall_fp)) if overall_fp else None
    if fpr is not None:
        print(f"\n  LOSO 假阳率: {fpr * 100:.2f}%  (n={len(overall_fp)})")
    return {'false_positive_rate': fpr, 'n': len(overall_fp)}


def train_stat(dataset_dir, output_dir='./gait_model_output',
               n_std=2.0, run_loso=True):
    """
    统计模型训练主流程：
      1. 遍历所有受试者文件夹，提取步态参数
      2. 用全体正常记录拟合 GaitNormModel
      3. 可选：LOSO 验证假阳率
      4. 保存模型、校准参数、训练记录
    """
    os.makedirs(output_dir, exist_ok=True)
    subject_folders = _iter_subject_folders(dataset_dir)
    if not subject_folders:
        raise ValueError(f"未找到受试者文件夹: {dataset_dir}")

    print(f"\n{'#'*55}")
    print(f"  [统计模型] 数据集: {dataset_dir}")
    print(f"  受试者: {len(subject_folders)}  阈值: ±{n_std}σ")
    print(f"{'#'*55}")

    all_normal, all_records, subject_calib, subject_results = [], [], {}, []

    for folder in subject_folders:
        result = _process_subject_for_training(folder, dataset_root=dataset_dir)
        if result is None:
            continue
        all_normal.extend(result['normal_records'])
        all_records.extend(result['all_records'])
        sid = result['subject_info']['id']
        subject_calib[sid] = {
            'calibration_factor': result['calibration_factor'],
            'subject_info':       result['subject_info'],
        }
        subject_results.append(result)

    if not all_normal:
        raise ValueError("无正常步态记录，请检查数据集")

    print(f"\n  正常记录: {len(all_normal)}  总记录: {len(all_records)}")

    model = GaitNormModel(n_std=n_std)
    model.fit(all_normal)

    # 打印参数统计摘要
    print("\n  参数统计：")
    for key, label in GaitNormModel.PARAM_LABELS.items():
        if key in model.stats:
            s = model.stats[key]
            print(f"    {label:<16} {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")

    if run_loso:
        _loso_validation(subject_results, n_std)

    # 保存
    model.save(os.path.join(output_dir, 'gait_norm_model.pkl'))
    with open(os.path.join(output_dir, 'subject_calibrations.json'),
              'w', encoding='utf-8') as f:
        json.dump(subject_calib, f, ensure_ascii=False, indent=2)
    pd.DataFrame(all_records).to_csv(
        os.path.join(output_dir, 'all_training_records.csv'),
        index=False, encoding='utf-8-sig')

    print(f"\n  ✅ 统计模型训练完成: {output_dir}")
    return model


# ──────────────────────────────────────────────────────────────────
# 第 10 节：train_deep() — 深度学习训练流水线
# ──────────────────────────────────────────────────────────────────

def train_deep(dataset_dir, output_dir='./gait_model_output',
               n_std=2.0, epochs=50, batch_size=32, lr=1e-3):
    """
    深度学习训练主流程：
      1. 遍历受试者文件夹，收集所有有效 CSV 的路径和标签
      2. 要求同时有正常和异常两类数据（否则抛出提示）
      3. 训练 GaitDeepModel，保存模型和校准记录

    注意：文件名含「双任务」的记录被标记为异常（is_normal=False）。
    如需更改标注规则，修改 match_action_type() 中的映射表。
    """
    os.makedirs(output_dir, exist_ok=True)
    subject_folders = _iter_subject_folders(dataset_dir)
    if not subject_folders:
        raise ValueError(f"未找到受试者文件夹: {dataset_dir}")

    print(f"\n{'#'*55}")
    print(f"  [深度学习] 数据集: {dataset_dir}")
    print(f"  受试者: {len(subject_folders)}")
    print(f"  参数: epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"{'#'*55}")

    all_records, n_normal_files, n_abnormal_files = [], 0, 0
    subject_calib = {}

    for folder in subject_folders:
        result = _process_subject_for_training(folder, dataset_root=dataset_dir)
        if result is None:
            continue

        sid = result['subject_info']['id']
        subject_calib[sid] = {
            'calibration_factor': result['calibration_factor'],
            'subject_info':       result['subject_info'],
        }

        for rec in result['all_records']:
            # 【新增】过滤掉校准文件和未识别动作
            if rec.get('action_type') in ('1330cm', '未知'):
                continue
            if not os.path.exists(rec.get('file_path', '')):
                continue
            all_records.append(rec)
            if rec.get('is_normal', True):
                n_normal_files += 1
            else:
                n_abnormal_files += 1

    print(f"\n  有效文件: {len(all_records)}")
    print(f"  正常: {n_normal_files}  异常: {n_abnormal_files}")

    if n_normal_files == 0 or n_abnormal_files == 0:
        raise ValueError(
            f"深度学习训练需要正常和异常两类数据！\n"
            f"  正常文件: {n_normal_files}\n"
            f"  异常文件: {n_abnormal_files}\n"
            f"提示：文件名包含「双任务」的被视为异常样本。\n"
            f"      也可以修改 match_action_type() 中的映射规则。")

    model = GaitDeepModel(n_std=n_std)
    model.fit(all_records, epochs=epochs, batch_size=batch_size, lr=lr)

    model.save(os.path.join(output_dir, 'gait_deep_model.pkl'))
    with open(os.path.join(output_dir, 'subject_calibrations.json'),
              'w', encoding='utf-8') as f:
        json.dump(subject_calib, f, ensure_ascii=False, indent=2)
    pd.DataFrame(all_records).to_csv(
        os.path.join(output_dir, 'all_training_records.csv'),
        index=False, encoding='utf-8-sig')

    print(f"\n  ✅ 深度学习模型训练完成: {output_dir}")
    return model


def train(dataset_dir, output_dir='./gait_model_output', n_std=2.0,
          mode='deep', run_loso=True, epochs=50, batch_size=32, lr=1e-3):
    """
    统一训练入口，根据 mode 分发到 train_stat / train_deep。
    mode: 'deep'（默认）或 'stat'
    """
    if mode == 'stat':
        return train_stat(dataset_dir, output_dir, n_std, run_loso)
    elif mode == 'deep':
        return train_deep(dataset_dir, output_dir, n_std, epochs, batch_size, lr)
    else:
        raise ValueError(f"未知模式: {mode}，请使用 'stat' 或 'deep'")


# ──────────────────────────────────────────────────────────────────
# 第 11 节：可视化 — 步态报告图 & 传感器时序图
# ──────────────────────────────────────────────────────────────────

def _setup_font():
    """
    自动选择中文字体，优先 SimHei / Microsoft YaHei / PingFang SC 等。
    在 Linux 服务器上通常使用 WenQuanYi 或 Noto Sans CJK。
    """
    candidates = ['SimHei', 'Microsoft YaHei', 'PingFang SC',
                  'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Arial Unicode MS']
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for f in candidates:
        if f in available:
            plt.rcParams['font.sans-serif'] = [f]
            break
    plt.rcParams['axes.unicode_minus'] = False

_setup_font()


def plot_gait_report(params, score_result, stats, n_std=2.0,
                     output_path='gait_report.png'):
    """
    绘制 3×3 步态分析报告图。
    前 6 格：每项参数的正态分布曲线 + 当前值标注 + 异常标记。
    最后 1 格：综合评估面板（等级 + 异常概率 + 建议）。
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    fig.suptitle('步态分析报告', fontsize=20, fontweight='bold',
                 y=0.98, color='#2c3e50')

    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.07, right=0.97, top=0.92, bottom=0.05)

    # 每个参数的配置：(键名, 显示标签, 单位, x轴显示范围)
    param_configs = [
        ('step_length_m',    '步长',         'm',     (0.3, 1.2)),
        ('step_speed_ms',    '步速',         'm/s',   (0.3, 2.0)),
        ('cadence_spm',      '步频',         '步/分', (60,  160)),
        ('step_width_ratio', '负载不对称度', '',      (0.0, 0.5)),
        ('stability_score',  '稳定性',       '',      (0.0, 1.0)),
        ('symmetry_score',   '对称性',       '',      (0.0, 1.0)),
    ]

    scored = score_result['params']

    for idx, (key, label, unit, xlim) in enumerate(param_configs):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        if key not in stats or key not in scored:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(label)
            continue

        s = stats[key]
        d = scored[key]
        val = d.get('value')
        if val is None:
            ax.text(0.5, 0.5, '本次未计算', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(label)
            continue

        mean, std = s['mean'], s['std']
        is_abn    = d.get('is_abnormal', False)
        z         = d.get('z_score', 0)

        # 绘制参考正态分布曲线
        x = np.linspace(xlim[0], xlim[1], 300)
        y = scipy_norm.pdf(x, mean, max(std, 1e-3))
        mask_normal = (x >= mean - n_std * std) & (x <= mean + n_std * std)
        ax.fill_between(x, y, where=mask_normal,
                        color=COLOR_REF, alpha=0.2, label='正常范围')
        ax.plot(x, y, color=COLOR_REF, linewidth=1.5, alpha=0.7)

        # 当前值竖线
        val_color = COLOR_ABNORMAL if is_abn else COLOR_NORMAL
        ax.axvline(val, color=val_color, linewidth=3, zorder=5,
                   label=f'当前: {val:.3f}{unit}')
        ax.axvline(mean, color='gray', linewidth=1, linestyle='--',
                   alpha=0.6, label=f'均值: {mean:.3f}')

        ax.set_xlim(xlim)
        ax.set_title(label, fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_xlabel(unit if unit else '指数值', fontsize=9)
        ax.set_ylabel('概率密度', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_facecolor(COLOR_BG)
        ax.grid(True, alpha=0.3)

        # 状态标签
        status_text  = '⚠ 异常' if is_abn else '✓ 正常'
        status_color = COLOR_ABNORMAL if is_abn else COLOR_NORMAL
        ax.text(0.97, 0.93, status_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=10, fontweight='bold',
                color=status_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=status_color, alpha=0.8))
        if z is not None:
            ax.text(0.97, 0.78, f'Z={z:.2f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, color='#555')

    # 第 9 格：综合评估面板
    ax_s = fig.add_subplot(gs[2, 2])
    _draw_summary_panel(ax_s, score_result)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [图表] 已保存: {output_path}")


def _draw_summary_panel(ax, score_result):
    """绘制综合评估面板（步态报告图右下角）。"""
    ax.axis('off')
    ax.set_facecolor(COLOR_BG)

    level = score_result['abnormality_level']
    n_abn = score_result['n_abnormal_params']
    level_color = {
        '正常':     COLOR_NORMAL,
        '轻度偏差': COLOR_WARN,
        '中度异常': COLOR_ABNORMAL,
        '明显异常': '#8e44ad',
        '数据不足': 'gray',
    }.get(level, 'gray')

    # 背景圆圈
    circle = plt.Circle((0.5, 0.68), 0.22, color=level_color, alpha=0.15,
                         transform=ax.transAxes, zorder=0)
    ax.add_patch(circle)

    ax.text(0.5, 0.92, '综合评估', ha='center', va='top',
            transform=ax.transAxes, fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(0.5, 0.70, level, ha='center', va='center',
            transform=ax.transAxes, fontsize=20, fontweight='bold',
            color=level_color)
    ax.text(0.5, 0.52, f'{n_abn} 项参数偏离正常范围',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10, color='#555')

    # 深度模型额外显示异常概率
    abn_prob = score_result.get('abnormal_prob')
    if abn_prob is not None:
        ax.text(0.5, 0.42, f'异常概率: {abn_prob:.1%}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color=level_color, fontweight='bold')

    # 偏异项目列表
    abn_keys = [k for k, v in score_result['params'].items() if v.get('is_abnormal')]
    if abn_keys:
        labels = [PARAM_LABELS.get(k, k) for k in abn_keys]
        ax.text(0.5, 0.32, '偏异项目：', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#888')
        ax.text(0.5, 0.22, '  '.join(labels), ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color=COLOR_ABNORMAL,
                fontweight='bold')

    ax.text(0.5, 0.08, _generate_suggestion(score_result),
            ha='center', va='center', transform=ax.transAxes,
            fontsize=8, color='#666', style='italic',
            bbox=dict(boxstyle='round', facecolor='#eef', alpha=0.5))


def _generate_suggestion(score_result):
    """根据异常参数组合生成文字建议。"""
    level    = score_result['abnormality_level']
    abn_keys = [k for k, v in score_result['params'].items() if v.get('is_abnormal')]

    if level == '正常':                                             return '步态状态良好，请继续保持。'
    if level == '数据不足':                                        return '数据量不足，建议增加采集时长。'
    if 'stability_score' in abn_keys and 'symmetry_score' in abn_keys:
        return '稳定性和对称性均偏低，\n建议咨询专业医师评估。'
    if 'step_speed_ms' in abn_keys:                               return '步速偏离正常范围，\n建议关注步行节奏。'
    if 'stability_score' in abn_keys:                             return '步态稳定性偏低，\n建议加强核心训练。'
    return '部分参数存在偏差，\n建议进行更多次测试或咨询医师。'


def plot_time_series(raw_df, output_path='timeseries.png'):
    """
    绘制 4 行时序图：
      行 1：右/左足跟压力
      行 2：右脚加速度 XYZ
      行 3：右脚陀螺仪 XYZ
      行 4：右脚姿态角 XYZ
    用于直观检查数据质量和步态节律。
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, facecolor='white')
    fig.suptitle('传感器时序数据', fontsize=14, fontweight='bold')

    t = parse_timestamps(raw_df.get('timestamp', pd.Series(range(len(raw_df)))), FS)

    # 压力
    ax = axes[0]
    for col, c, lbl in [('P_heel_R', '#e74c3c', '跟部_R'),
                        ('P_heel_L', '#3498db', '跟部_L')]:
        if col in raw_df.columns:
            ax.plot(t, raw_df[col], color=c, alpha=0.7, linewidth=0.8, label=lbl)
    ax.set_ylabel('压力 (ADC)', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, alpha=0.3)

    # 加速度 / 陀螺仪 / 姿态角（右脚）
    groups = [
        (['acc_x_R', 'acc_y_R', 'acc_z_R'], '加速度_R (g)',   axes[1]),
        (['ave_x_R', 'ave_y_R', 'ave_z_R'], '陀螺仪_R (°/s)', axes[2]),
        (['ang_x_R', 'ang_y_R', 'ang_z_R'], '姿态角_R (°)',   axes[3]),
    ]
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for cols, ylabel, ax in groups:
        for col, c in zip(cols, colors):
            if col in raw_df.columns:
                ax.plot(t, raw_df[col], color=c, alpha=0.7, linewidth=0.8,
                        label=col.split('_')[1])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_facecolor(COLOR_BG)
        ax.grid(True, alpha=0.3)

    axes[3].set_xlabel('时间 (s)', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [图表] 时序图已保存: {output_path}")


# ──────────────────────────────────────────────────────────────────
# 第 12 节：HTML / JSON 报告生成
# ──────────────────────────────────────────────────────────────────

def _img_to_base64(path):
    """将图片文件编码为 base64 data URI，嵌入 HTML 报告（无需外部图片文件）。"""
    if not path or not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()


def generate_html_report(params, score_result, subject_info=None,
                         model_type='unknown', report_image=None,
                         output_path='report.html'):
    """
    生成自包含 HTML 报告（图表以 base64 内嵌），包含：
      - 受试者基本信息卡片
      - 综合评估等级
      - 6 项参数详情表（含偏差百分比、正常参考、状态图标）
      - 建议文字
      - 参数分布图（嵌入式 PNG）
    """
    level     = score_result['abnormality_level']
    n_abn     = score_result['n_abnormal_params']
    abn_prob  = score_result.get('abnormal_prob')
    n_std_val = score_result.get('n_std', 2.0)

    level_color = {
        '正常': '#27ae60', '轻度偏差': '#f39c12',
        '中度异常': '#e74c3c', '明显异常': '#8e44ad', '数据不足': '#7f8c8d',
    }.get(level, '#7f8c8d')

    # 参数详情表行
    rows = ''
    for key in PARAM_KEYS:
        d = score_result['params'].get(key, {})
        if d.get('value') is None:
            continue
        val  = d['value']
        abn  = d.get('is_abnormal', False)
        z    = d.get('z_score')
        dev  = d.get('deviation_pct', 0)
        mean = d.get('normal_mean', 0)
        std  = d.get('normal_std', 0)
        q25  = d.get('normal_q25', 0)
        q75  = d.get('normal_q75', 0)
        sc   = '#e74c3c' if abn else '#27ae60'
        icon = '⚠️' if abn else '✅'
        dev_str = f"+{dev:.1f}%" if dev >= 0 else f"{dev:.1f}%"
        rows += f"""
        <tr>
            <td><strong>{PARAM_LABELS.get(key, key)}</strong></td>
            <td style="color:{sc};font-weight:bold">{val:.3f}</td>
            <td>{mean:.3f} ± {std:.3f}</td>
            <td>[{q25:.3f} ~ {q75:.3f}]</td>
            <td style="color:{sc}">{dev_str}</td>
            <td style="font-size:1.2em;text-align:center">{icon}</td>
        </tr>"""

    img_b64 = _img_to_base64(report_image)
    img_tag = (f'<img src="{img_b64}" style="width:100%;border-radius:8px;margin-top:20px;">'
               if img_b64 else '')

    sname   = (subject_info or {}).get('id', '未知')
    h_cm    = (subject_info or {}).get('height_cm', '-')
    w_kg    = (subject_info or {}).get('weight_kg', '-')
    model_badge = '🧠 深度学习模型' if model_type == 'deep' else '📊 统计模型'
    prob_html   = (f'<p style="color:{level_color};font-weight:bold">'
                   f'异常概率: {abn_prob:.1%}</p>' if abn_prob is not None else '')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>步态分析报告</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"PingFang SC","Microsoft YaHei",sans-serif;
        background:#f0f4f8;color:#2c3e50;line-height:1.6}}
  .container{{max-width:860px;margin:30px auto;padding:0 16px 40px}}
  .card{{background:white;border-radius:12px;
         box-shadow:0 2px 12px rgba(0,0,0,.08);padding:24px;margin-bottom:20px}}
  h1{{font-size:1.8em;margin-bottom:6px}}
  h2{{font-size:1.2em;color:#555;margin-bottom:16px;
      border-bottom:2px solid #eee;padding-bottom:8px}}
  .status-badge{{display:inline-block;padding:10px 28px;border-radius:50px;
        font-size:1.4em;font-weight:bold;color:white;
        background:{level_color};margin:12px 0}}
  .model-badge{{display:inline-block;padding:4px 12px;border-radius:20px;
        font-size:.8em;background:#eef;color:#555;margin-left:12px}}
  .meta{{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:8px}}
  .meta-item{{background:#f8f9fa;border-radius:8px;padding:8px 16px;font-size:.9em}}
  table{{width:100%;border-collapse:collapse;font-size:.92em}}
  th{{background:#f0f4f8;padding:10px;text-align:left;color:#555}}
  td{{padding:10px;border-bottom:1px solid #f0f0f0}}
  tr:hover{{background:#fafafa}}
  .suggestion{{background:#eef7ff;border-left:4px solid #3498db;
               padding:14px 18px;border-radius:0 8px 8px 0;
               font-size:.95em;color:#2980b9}}
  .footer{{text-align:center;color:#aaa;font-size:.8em;margin-top:20px}}
</style></head>
<body><div class="container">
  <div class="card">
    <h1>🦶 步态分析报告 <span class="model-badge">{model_badge}</span></h1>
    <div class="meta">
      <div class="meta-item">👤 受试者: <strong>{sname}</strong></div>
      <div class="meta-item">📏 身高: <strong>{h_cm} cm</strong></div>
      <div class="meta-item">⚖️ 体重: <strong>{w_kg} kg</strong></div>
    </div>
    <div>综合评估：</div>
    <div class="status-badge">{level}</div>
    {prob_html}
    <p style="color:#666;margin-top:8px">
      {n_abn} 项参数偏离正常范围（阈值：±{n_std_val:.1f} 标准差）
    </p>
  </div>
  <div class="card">
    <h2>📊 步态参数详情</h2>
    <table>
      <thead><tr>
        <th>参数</th><th>测量值</th><th>正常参考</th>
        <th>四分位区间</th><th>偏差</th><th>状态</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p style="color:#aaa;font-size:.8em;margin-top:10px">
      * 正常参考值基于训练集健康受试者统计。偏差为相对均值百分比。
    </p>
  </div>
  <div class="card">
    <h2>💡 建议</h2>
    <div class="suggestion">{_generate_suggestion(score_result)}</div>
  </div>
  {'<div class="card"><h2>📈 参数分布图</h2>' + img_tag + '</div>' if img_tag else ''}
  <div class="footer">本报告由步态分析系统自动生成，仅供参考，不构成医疗诊断。</div>
</div></body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  [报告] HTML 已保存: {output_path}")


# ──────────────────────────────────────────────────────────────────
# 第 13 节：analyze() — 推理主流程（自动识别模型类型）
# ──────────────────────────────────────────────────────────────────

def analyze(input_csv, model_path, height_cm, weight_kg=None,
            calib_factor=None, calib_csv=None,
            output_dir='./report_output', subject_id='test'):
    """
    对单个 CSV 文件执行步态分析并生成完整报告。

    流程：
      1. 自动加载模型（统计 / 深度，由 pkl 文件内容决定）
      2. 计算步长校准系数（优先用传入的 calib_factor，其次用 calib_csv，否则 K=1.0）
      3. 根据模型类型选择推理路径：
           深度模型：直接输入原始时序 → GaitDeepModel.score()
           统计模型：先提取 6 项参数 → GaitNormModel.score()
      4. 绘制参数分布图和时序图
      5. 生成 JSON 和 HTML 报告

    输出文件（保存在 output_dir）：
      {subject_id}_report.png    — 参数分布图
      {subject_id}_timeseries.png — 时序图
      {subject_id}_report.json   — 结构化结果
      {subject_id}_report.html   — 完整 HTML 报告
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  步态分析推理")
    print(f"  输入: {os.path.basename(input_csv)}")
    print(f"{'='*55}")

    # 步骤 1：加载模型
    model, model_type = load_gait_model(model_path)
    print(f"  模型类型: {'深度学习' if model_type == 'deep' else '统计模型'}")

    leg_length_m = height_cm * 0.53 / 100.0
    subject_info = {
        'id':           subject_id,
        'height_cm':    height_cm,
        'weight_kg':    weight_kg if weight_kg is not None else '-',
        'leg_length_m': leg_length_m,
    }

    # 步骤 2：确定校准系数
    if calib_factor is not None:
        factor = calib_factor
        print(f"  校准系数: K={factor:.4f}")
    elif calib_csv and os.path.exists(calib_csv):
        print("  自动校准...")
        factor = calibrate_step_length_factor(calib_csv, 13.30, leg_length_m, FS)
    else:
        factor = 1.0
        print("  无校准信息，使用 K=1.0")

    # 步骤 3：读取原始数据并推理
    raw_df = pd.read_csv(input_csv, encoding='utf-8-sig')

    if model_type == 'deep':
        # 深度模型：直接从时序推理
        print("  [深度模型] 推理中...")
        score_result = model.score(raw_df)
        score_result['n_std'] = model.n_std
        # 同时用传统方法提取参数做日志参考（不影响评分）
        params = extract_parameters_from_file(input_csv, leg_length_m, factor, FS)
        if 'error' in params:
            print(f"  [传统参数提取] ⚠ {params['error']}（不影响深度模型结果）")
            params = {}
    else:
        # 统计模型：先提取参数再评分
        print("  提取步态参数...")
        params = extract_parameters_from_file(input_csv, leg_length_m, factor, FS)
        if 'error' in params:
            print(f"  ❌ 错误: {params['error']}")
            return None, None
        score_result = model.score(params)
        score_result['n_std'] = model.n_std

    # 步骤 4：打印结果摘要
    print(f"\n  ── 评估结果 ──")
    print(f"     综合判断: {score_result['abnormality_level']}")
    print(f"     异常参数: {score_result['n_abnormal_params']} 项")
    if score_result.get('abnormal_prob') is not None:
        print(f"     异常概率: {score_result['abnormal_prob']:.1%}")
    print(f"\n  ── 步态参数 ──")
    for key in PARAM_KEYS:
        d = score_result['params'].get(key, {})
        val = d.get('value')
        if val is not None:
            mark = ' ⚠' if d.get('is_abnormal') else ''
            print(f"     {PARAM_LABELS.get(key, key):<16} {val:.4f}{mark}")

    # 步骤 5：生成图表
    report_img = os.path.join(output_dir, f'{subject_id}_report.png')
    ts_img     = os.path.join(output_dir, f'{subject_id}_timeseries.png')
    plot_gait_report(params, score_result, model.stats, model.n_std, report_img)
    plot_time_series(raw_df, ts_img)

    # 步骤 6：保存 JSON
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return str(obj)

    json_path = os.path.join(output_dir, f'{subject_id}_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_type':         model_type,
            'subject_info':       subject_info,
            'calibration_factor': factor,
            'parameters':         params if params else {},
            'score_result':       score_result,
        }, f, ensure_ascii=False, indent=2, default=_serialize)
    print(f"  [报告] JSON 已保存: {json_path}")

    # 步骤 7：保存 HTML
    html_path = os.path.join(output_dir, f'{subject_id}_report.html')
    generate_html_report(params, score_result, subject_info,
                         model_type=model_type,
                         report_image=report_img,
                         output_path=html_path)

    print(f"\n  ✅ 分析完成！报告目录: {output_dir}")
    return params, score_result


# ──────────────────────────────────────────────────────────────────
# 第 14 节：命令行入口（train / analyze 双子命令）
# ──────────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='步态分析系统 — 训练 & 推理一体化',
#         formatter_class=argparse.RawTextHelpFormatter)
#
#     subparsers = parser.add_subparsers(dest='command', help='子命令')
#
#     # ── train 子命令 ──
#     p_train = subparsers.add_parser('train', help='训练模型')
#     p_train.add_argument('--dataset',    required=True,
#                          help='数据集根目录（含多个受试者子文件夹）')
#     p_train.add_argument('--output',     default='./gait_model_output',
#                          help='模型输出目录（默认 ./gait_model_output）')
#     p_train.add_argument('--mode',       default='deep',
#                          choices=['deep', 'stat'],
#                          help='训练模式：deep=深度学习  stat=统计模型（默认 deep）')
#     p_train.add_argument('--epochs',     type=int,   default=50)
#     p_train.add_argument('--batch_size', type=int,   default=32)
#     p_train.add_argument('--lr',         type=float, default=1e-3)
#     p_train.add_argument('--n_std',      type=float, default=2.0,
#                          help='异常判断阈值（标准差倍数，默认 2.0）')
#     p_train.add_argument('--no_loso',    action='store_true',
#                          help='跳过 LOSO 验证（仅统计模型）')
#
#     # ── analyze 子命令 ──
#     p_analyze = subparsers.add_parser('analyze', help='推理 + 生成报告')
#     p_analyze.add_argument('--input',        required=True,
#                            help='待分析的 CSV 文件路径')
#     p_analyze.add_argument('--model',        required=True,
#                            help='模型文件路径（.pkl）')
#     p_analyze.add_argument('--height',       required=True, type=float,
#                            help='受试者身高（cm）')
#     p_analyze.add_argument('--weight',       type=float, default=None,
#                            help='受试者体重（kg，可选）')
#     p_analyze.add_argument('--calib_factor', type=float, default=None,
#                            help='预设步长校准系数（可选）')
#     p_analyze.add_argument('--calib_csv',    default=None,
#                            help='1330cm 校准 CSV 文件路径（可选）')
#     p_analyze.add_argument('--output',       default='./report_output',
#                            help='报告输出目录（默认 ./report_output）')
#     p_analyze.add_argument('--subject_id',   default='test',
#                            help='受试者 ID（用于文件命名，默认 test）')
#
#     args = parser.parse_args()
#
#     if args.command == 'train':
#         train(
#             dataset_dir=args.dataset,
#             output_dir=args.output,
#             mode=args.mode,
#             n_std=args.n_std,
#             run_loso=not args.no_loso,
#             epochs=args.epochs,
#             batch_size=args.batch_size,
#             lr=args.lr,
#         )
#
#     elif args.command == 'analyze':
#         analyze(
#             input_csv=args.input,
#             model_path=args.model,
#             height_cm=args.height,
#             weight_kg=args.weight,
#             calib_factor=args.calib_factor,
#             calib_csv=args.calib_csv,
#             output_dir=args.output,
#             subject_id=args.subject_id,
#         )
#
#     else:
#         # 未传子命令时显示帮助（也可直接在此修改路径快速运行）
#         print("用法示例：")
#         print("  训练：python gait_pipeline_all_in_one.py train "
#               "--dataset ./dataset --output ./model --mode deep")
#         print("  推理：python gait_pipeline_all_in_one.py analyze "
#               "--input ./test/直线来回.csv --model ./model/gait_deep_model.pkl "
#               "--height 170")
#         print()
#         parser.print_help()
# 直接硬编码训练（放在文件最末尾）
# if __name__ == '__main__':
#      # ========== 注释掉原来的训练代码（在每一行前加 #） ==========
#      train(
#          dataset_dir='F:/实验/处理后的数据',
#          output_dir='./gait_model_stat',
#          mode='stat',
#          n_std=2.0,
#          run_loso=False
#      )

    # ========== 取消注释下面的分析代码（复制粘贴，修改路径） ==========
if __name__ == '__main__':
    train(
        dataset_dir=r"F:\实验\处理后的数据\训练集",  # 绝对不要传根目录，否则测试集会混入训练
        output_dir=r"./gait_model_deep",
        mode='deep',
        epochs=50,
        batch_size=32,
        lr=1e-3,
    )