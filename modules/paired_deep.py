# -*- coding: utf-8 -*-
"""Paired dual-branch deep model using raw windows and gait parameters."""
import argparse
import copy
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from _common import PARAM_KEYS, extract_windows, read_df
from label_dtc import _pair


PARAMETERS = PARAM_KEYS
FAMILIES = {
    '直线': ('直线来回', '双任务直线'),
    '上楼梯': ('上楼梯', '双任务上楼梯'),
    '下楼梯': ('下楼梯', '双任务下楼梯'),
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (
            torch.FloatTensor(row['base_windows']),
            torch.FloatTensor(row['current_windows']),
            torch.FloatTensor(row['features']),
            torch.FloatTensor([row['label']]),
        )


def _collate(batch):
    base, current, features, labels = [], [], [], []
    base_index, current_index = [], []
    for index, (base_w, current_w, feature, label) in enumerate(batch):
        base.append(base_w)
        current.append(current_w)
        features.append(feature)
        labels.append(label)
        base_index.extend([index] * len(base_w))
        current_index.extend([index] * len(current_w))
    return (
        torch.cat(base),
        torch.cat(current),
        torch.LongTensor(base_index),
        torch.LongTensor(current_index),
        len(batch),
        torch.stack(features),
        torch.stack(labels),
    )


class PairEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(24, 32, 5, padding=2), nn.BatchNorm1d(32),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.attention = nn.Sequential(
            nn.Linear(96, 32), nn.Tanh(), nn.Linear(32, 1))

    def pool(self, windows, index, n_bags):
        embedding = self.layers(windows.transpose(1, 2))
        score = torch.exp(self.attention(embedding))
        denominator = torch.zeros(n_bags, 1, device=embedding.device)
        denominator.index_add_(0, index, score)
        output = torch.zeros(n_bags, 96, device=embedding.device)
        output.index_add_(0, index, embedding * score / denominator[index])
        return output


class PairNet(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.encoder = PairEncoder()
        self.head = nn.Sequential(
            nn.Linear(96 * 3 + feature_size, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, base, current, base_index, current_index,
                n_pairs, features):
        base_embedding = self.encoder.pool(base, base_index, n_pairs)
        current_embedding = self.encoder.pool(current, current_index, n_pairs)
        combined = torch.cat([
            base_embedding,
            current_embedding,
            current_embedding - base_embedding,
            features,
        ], dim=1)
        return self.head(combined).squeeze(1)


def _family(action):
    for name, actions in FAMILIES.items():
        if action in actions:
            return name
    return None


def _features(base, current, family):
    output = []
    for prefix, values in (('base', base), ('current', current)):
        output.extend(float(values.get(key, np.nan)) for key in PARAMETERS)
    for key in PARAMETERS:
        b = float(base.get(key, np.nan))
        c = float(current.get(key, np.nan))
        output.append(c - b)
        output.append((c - b) / (abs(b) + 1e-6))
    output.extend([int(family == name) for name in FAMILIES])
    return np.nan_to_num(np.asarray(output, dtype=np.float32))


def _item(row, base_path, current_path, base_params, current_params,
          family, label, window_size, step_size):
    try:
        base_windows = extract_windows(
            read_df(base_path), window_size, step_size)
        current_windows = extract_windows(
            read_df(current_path), window_size, step_size)
    except Exception:
        return None
    if len(base_windows) == 0 or len(current_windows) == 0:
        return None
    return {
        'subject_id': row['subject_id'],
        'split': row['split'],
        'family': family,
        'label': label,
        'base_windows': base_windows,
        'current_windows': current_windows,
        'features': _features(base_params, current_params, family),
    }


def _build_pairs(record_path, window_size=300, step_size=150):
    frame = pd.read_csv(record_path, encoding='utf-8-sig')
    frame['family'] = frame['action'].map(_family)
    frame = frame[frame['family'].notna()]
    samples = []
    for (subject_id, split, family), subset in frame.groupby(
            ['subject_id', 'split', 'family']):
        singles = subset[~subset.action.str.startswith('双任务')].copy()
        duals = subset[subset.action.str.startswith('双任务')].copy()
        singles = singles.sort_values('repeat')
        duals = duals.sort_values('repeat')
        single_items = [{
            'rep': int(row['repeat']), 'file': row['file'],
            'p': {key: row[key] for key in PARAMETERS},
        } for _, row in singles.iterrows()]
        dual_items = [{
            'rep': int(row['repeat']), 'file': row['file'],
            'p': {key: row[key] for key in PARAMETERS},
        } for _, row in duals.iterrows()]
        base_row = {'subject_id': subject_id, 'split': split}
        for dual, single, _method in _pair(single_items, dual_items)[0]:
            item = _item(base_row, single['file'], dual['file'],
                         single['p'], dual['p'], family, 1,
                         window_size, step_size)
            if item:
                samples.append(item)

        if len(single_items) >= 2:
            first, second = single_items[0], single_items[1]
            item = _item(base_row, first['file'], second['file'],
                         first['p'], second['p'], family, 0,
                         window_size, step_size)
            if item:
                samples.append(item)
        elif family == '直线' and single_items:
            # Long straight files provide a within-file normal pair.
            single = single_items[0]
            data = read_df(single['file'])
            midpoint = len(data) // 2
            first = extract_windows(
                data.iloc[:midpoint], window_size, step_size)
            second = extract_windows(
                data.iloc[midpoint:], window_size, step_size)
            if len(first) and len(second):
                samples.append({
                    'subject_id': subject_id,
                    'split': split,
                    'family': family,
                    'label': 0,
                    'base_windows': first,
                    'current_windows': second,
                    'features': _features(
                        single['p'], single['p'], family),
                })
    return samples


def _threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.round(np.arange(0.05, 0.991, 0.01), 2):
        pred = (p > threshold).astype(int)
        score = f1_score(y, pred)
        if score > best_f1:
            best_t, best_f1 = float(threshold), score
    return best_t


def _predict(model, loader):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for base, current, bi, ci, n, features, target in loader:
            logits = model(
                base.to(DEVICE), current.to(DEVICE), bi.to(DEVICE),
                ci.to(DEVICE), n, features.to(DEVICE))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(target.numpy().ravel())
    return np.asarray(labels), np.asarray(probabilities)


def _metrics(y, p, threshold):
    pred = (p > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'n': len(y), 'positive': int(y.sum()),
        'accuracy': accuracy_score(y, pred),
        'sensitivity': recall_score(y, pred),
        'specificity': tn / max(tn + fp, 1),
        'precision': precision_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred),
        'auc': roc_auc_score(y, p),
        'threshold': threshold,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def run(record_path, output_dir, window_size=300, step_size=150):
    rows = _build_pairs(record_path, window_size, step_size)
    if not rows:
        raise RuntimeError('没有生成配对深度样本')
    train = [row for row in rows if row['split'] == 'train']
    val = [row for row in rows if row['split'] == 'val']
    test = [row for row in rows if row['split'] == 'test']
    all_windows = np.concatenate(
        [row['base_windows'] for row in train]
        + [row['current_windows'] for row in train]
    )
    window_scaler = StandardScaler().fit(
        all_windows.reshape(-1, all_windows.shape[-1]))
    feature_scaler = StandardScaler().fit(
        np.asarray([row['features'] for row in train]))
    for row in train + val + test:
        for key in ('base_windows', 'current_windows'):
            shape = row[key].shape
            row[key] = window_scaler.transform(
                row[key].reshape(-1, shape[-1])).reshape(shape)
        row['features'] = feature_scaler.transform([row['features']])[0]

    loaders = [
        DataLoader(PairDataset(train), batch_size=4, shuffle=True,
                   collate_fn=_collate),
        DataLoader(PairDataset(val), batch_size=4, shuffle=False,
                   collate_fn=_collate),
        DataLoader(PairDataset(test), batch_size=4, shuffle=False,
                   collate_fn=_collate),
    ]
    model = PairNet(train[0]['features'].shape[0]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, patience = -1.0, None, 0
    for epoch in range(80):
        model.train()
        total = 0.0
        for base, current, bi, ci, n, features, labels in loaders[0]:
            optimizer.zero_grad()
            output = model(
                base.to(DEVICE), current.to(DEVICE), bi.to(DEVICE),
                ci.to(DEVICE), n, features.to(DEVICE))
            loss = loss_fn(output, labels.to(DEVICE).squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        y_val, p_val = _predict(model, loaders[1])
        auc = roc_auc_score(y_val, p_val)
        if (epoch + 1) % 10 == 0:
            print(f'  epoch={epoch + 1} loss={total / len(loaders[0]):.4f} '
                  f'val_auc={auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(best_state)
    y_val, p_val = _predict(model, loaders[1])
    threshold = _threshold(y_val, p_val)
    y_test, p_test = _predict(model, loaders[2])
    overall = _metrics(y_test, p_test, threshold)
    by_action = []
    test_meta = [row['family'] for row in test]
    for family in FAMILIES:
        mask = np.asarray([name == family for name in test_meta])
        if mask.sum() and len(np.unique(y_test[mask])) > 1:
            by_action.append({
                'action': family,
                **_metrics(y_test[mask], p_test[mask], threshold),
            })
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'paired_deep.pkl'), 'wb') as stream:
        pickle.dump({
            'state': model.state_dict(),
            'window_scaler': window_scaler,
            'feature_scaler': feature_scaler,
            'threshold': threshold,
        }, stream)
    summary = {
        'n_samples': len(rows),
        'window_size': window_size,
        'step_size': step_size,
        'overlap': 1 - step_size / window_size,
        'train': len(train), 'val': len(val), 'test': len(test),
        'best_val_auc': best_auc,
        'test': overall,
        'by_action': by_action,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w',
              encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(pd.DataFrame([overall]).round(4).to_string(index=False))
    print(pd.DataFrame(by_action).round(4).to_string(index=False))
    print(f'[paired_deep] -> {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--record_path',
        default=r'./gait_report_condition_action/paired/parameter_records.csv')
    parser.add_argument('--output_dir', default='./gait_model_paired_deep')
    parser.add_argument('--window_size', type=int, default=300)
    parser.add_argument('--step_size', type=int, default=150)
    args = parser.parse_args()
    run(args.record_path, args.output_dir,
        args.window_size, args.step_size)
