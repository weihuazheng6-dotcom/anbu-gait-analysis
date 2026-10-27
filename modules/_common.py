# -*- coding: utf-8 -*-
"""公共工具：路径/扫描/静默调用。模块只依赖本文件 + 原始 gait_pipeline_all_in_one.py。"""
import contextlib, glob, io, os, pickle, sys
from pathlib import Path
import pandas as pd

# 定位 gait_pipeline_all_in_one.py：从本文件向上逐级搜索，找到就加入 sys.path
for _p in Path(__file__).resolve().parents:
    if (_p / 'gait_pipeline_all_in_one.py').exists():
        sys.path.insert(0, str(_p))
        break
else:
    raise FileNotFoundError('未找到 gait_pipeline_all_in_one.py，请把它放在 gait_project 的上级目录中')
from gait_pipeline_all_in_one import (  # noqa: E402
    SENSOR_COLS, PRESSURE_COLS_R, PRESSURE_COLS_L,
    FS, WINDOW_SIZE, STEP_SIZE, PARAM_KEYS, PARAM_LABELS,
    match_action_type, load_subject_info,
    _process_subject_for_training, extract_windows, GaitDeepModel, GaitNet)

PRESSURE_COLS = PRESSURE_COLS_R + PRESSURE_COLS_L


def _compatible_deep_load(self, path):
    """Load saved GaitNet weights while inferring the input-channel count.

    The upstream save file does not store n_channels.  The all-in-one model
    appends six action channels, so rebuilding every checkpoint with the
    default 24 channels makes otherwise valid 30-channel checkpoints unusable.
    """
    with open(path, 'rb') as f:
        state = pickle.load(f)
    if state.get('model_type') != 'deep':
        raise ValueError('这不是深度学习模型文件')
    conv_key = 'cnn.0.conv.weight'
    if conv_key not in state.get('model_state', {}):
        raise ValueError('模型权重缺少首层卷积参数，无法推断输入通道数')
    n_channels = int(state['model_state'][conv_key].shape[1])
    self.scaler = state['scaler']
    self.param_scaler = state['param_scaler']
    self.stats = state['stats']
    self.n_std = state['n_std']
    self.model = GaitNet(n_channels=n_channels).to(self.device)
    self.model.load_state_dict(state['model_state'])
    self.model.eval()
    self.is_fitted = True


GaitDeepModel.load = _compatible_deep_load

def map_group(g):
    """组目录名 → 老年 / 年轻模拟"""
    if g.startswith('老年'): return '老年'
    if g.startswith('模拟'): return '年轻模拟'
    return g

def iter_subjects(dataset):
    """扫描所有含 *_filtered.csv 的受试者文件夹。yield (part, group, subject_id, folder)。"""
    for d, _dirs, files in os.walk(dataset):
        if not any(f.endswith('_filtered.csv') for f in files):
            continue
        rel = os.path.relpath(d, dataset).replace(os.sep, '/').lower()
        part = 'test' if ('测试' in rel or '/test' in rel) else 'train'
        group = os.path.basename(os.path.dirname(d))
        yield part, group, f'{group}_{os.path.basename(d)}', d

def iter_files(dataset):
    """yield (csv_path, part, group, subject_id)"""
    for part, group, sid, folder in iter_subjects(dataset):
        for f in sorted(glob.glob(os.path.join(folder, '*_filtered.csv'))):
            yield f, part, group, sid

def read_df(path):
    return pd.read_csv(path, encoding='utf-8-sig')

def silent(fn, *a, **k):
    """静默执行（吞掉打印），返回 (结果, stdout)。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()
