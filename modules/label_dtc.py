# -*- coding: utf-8 -*-
"""DTC 风险标签生成 + 配对统计：受试者内 单任务 vs 双任务 配对。

配对规则（用户确认）：
    上楼梯  ↔ 双任务上楼梯
    下楼梯  ↔ 双任务下楼梯
    直线来回 ↔ 双任务直线
1330（13.3 米直线行走）、直线停顿、直角转弯 = 单任务低风险；"未知" 排除。
DTC = (单任务 − 双任务) / 单任务 × 100%，正值 = 退化。
高风险规则（输出敏感性分析）：
    R1: 步速 DTC > 10%（文献常用）
    R2: 综合 DTC（步速/步长/步频均值）> 组内 75 分位（仅双任务文件内）
    risk = R1 或 R2
输出 gait_report/label_dtc/：labels_dtc.csv、labels_all.csv、paired_stats.csv、
    summary.json、assets/dtc_*.png
"""
import argparse, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from _common import (iter_subjects, silent, PARAM_KEYS, map_group,
                     _process_subject_for_training)

PAIRS = [('上楼梯', '双任务上楼梯'), ('下楼梯', '双任务下楼梯'), ('直线来回', '双任务直线')]
REP = re.compile(r'[-_](\d+)_calibrated')
DTC_PARAMS = ('step_speed_ms', 'step_length_m', 'cadence_spm')  # symmetry ICC 不可靠，不进规则
SPEED_THR = 10.0
COMP_Q = 0.75

def _collect(dataset):
    """{受试者: {动作: [{'rep','file','g','p':{参数:值}}, ...]}}，跳过'未知'。"""
    subs = {}
    for part, group, sid, folder in iter_subjects(dataset):
        sub, _ = silent(_process_subject_for_training, folder, dataset)
        if not sub:
            continue
        act = {}
        for r in sub['all_records']:
            if r['action_type'] == '未知':
                continue
            m = REP.search(os.path.basename(r['file_path']))
            act.setdefault(r['action_type'], []).append(
                {'rep': int(m.group(1)) if m else -1, 'file': r['file_path'],
                 'g': map_group(group), 'p': {k: r.get(k) for k in PARAM_KEYS}})
        subs[sid] = act
    return subs

def _pair(singles, duals):
    """按重复号配对，缺号时按顺序兜底。返回 (配对列表, 未配对的双任务)。"""
    out, used = [], set()
    ss = sorted(singles, key=lambda x: x['rep'])
    ds = sorted(duals, key=lambda x: x['rep'])
    for d in ds:
        for si, s in enumerate(ss):
            if si not in used and s['rep'] == d['rep']:
                used.add(si); out.append((d, s, 'repeat')); break
    rest_d = [d for d in ds if all(o[0] is not d for o in out)]
    rest_s = [s for si, s in enumerate(ss) if si not in used]
    for d, s in zip(rest_d, rest_s):
        out.append((d, s, 'index'))
    left = rest_d[len(rest_s):] if len(rest_d) > len(rest_s) else []
    return out, left

def _dtc(pd_, ps):
    """逐参数 DTC，分母为 0 或缺失 → NaN。"""
    out = {}
    for k in PARAM_KEYS:
        vd, vs = pd_.get(k), ps.get(k)
        bad = (vd is None or vs is None or vs == 0 or pd.isna(vd) or pd.isna(vs))
        out[f'dtc_{k}'] = np.nan if bad else (vs - vd) / vs * 100.0
    return out

def run(dataset, report_out):
    subs = _collect(dataset)
    label_rows, first_pairs, no_base = [], [], []
    for sid, act in subs.items():
        for base, dual in PAIRS:
            pairs, left = _pair(act.get(base, []), act.get(dual, []))
            if not pairs:
                no_base += [d['file'] for d in act.get(dual, [])]
                continue
            no_base += [d['file'] for d in left]
            first_pairs.append((sid, dual, pairs[0]))
            for d, s, mt in pairs:
                label_rows.append({'file': d['file'], 'subject_id': sid, 'group': d['g'],
                                   'action': dual, 'base_action': base, 'repeat': d['rep'],
                                   'match': mt, **_dtc(d['p'], s['p'])})
    df = pd.DataFrame(label_rows)
    if df.empty:
        raise RuntimeError('未找到任何单/双任务配对，请检查数据集路径')

    # 高风险规则。综合 DTC 分位阈值只允许从训练集估计，
    # 避免测试集标签通过阈值计算反向污染评估。
    df['dtc_composite'] = df[[f'dtc_{k}' for k in DTC_PARAMS]].mean(axis=1, skipna=True)
    df['rule_speed10'] = (df.dtc_step_speed_ms > SPEED_THR).astype(int)
    split_path = os.path.join(report_out, 'split', 'summary.json')
    train_ids = set()
    threshold_scope = 'all_subjects(fallback)'
    if os.path.exists(split_path):
        with open(split_path, encoding='utf-8') as f:
            split_summary = json.load(f)
        train_ids = set(split_summary.get('subjects', {}).get('train', []))
        threshold_scope = 'train_subjects'
    train_rows = df[df.subject_id.isin(train_ids)]
    if train_rows.empty:
        train_rows = df
        threshold_scope = 'all_subjects(fallback)'
    group_thresholds = train_rows.groupby('group')['dtc_composite'].quantile(COMP_Q)
    global_threshold = float(train_rows['dtc_composite'].quantile(COMP_Q))
    thresholds = df['group'].map(group_thresholds).fillna(global_threshold)
    df['rule_composite75'] = (df.dtc_composite > thresholds).astype(int)
    df['risk'] = ((df.rule_speed10 + df.rule_composite75) > 0).astype(int)
    df['class3'] = np.where(df.risk == 1, '双任务退化', '双任务未退化')

    # 配对统计（每受试者每动作取 1 对，避免重复次数膨胀）
    fl = []
    for sid, dual, (d, s, mt) in first_pairs:
        for k in PARAM_KEYS:
            fl.append({'sid': sid, 'action': dual, 'group': d['g'], 'param': k,
                       'single': s['p'].get(k), 'dual': d['p'].get(k)})
    fp = pd.DataFrame(fl)
    stat_rows = []
    for (act, pm), dd in fp.groupby(['action', 'param']):
        for g, gg in [('合并', dd)] + [(g, x) for g, x in dd.groupby('group')]:
            gg = gg.dropna(subset=['single', 'dual'])
            if len(gg) < 3:
                continue
            try:
                w = stats.wilcoxon(gg['single'], gg['dual'])
            except Exception:
                w = None
            t = stats.ttest_rel(gg['single'], gg['dual'])
            diff = gg['dual'] - gg['single']
            dtc_m = (-diff / gg['single'] * 100).mean()
            stat_rows.append({'action': act, 'param': pm, 'group': g, 'n_pairs': len(gg),
                              'mean_single': round(gg['single'].mean(), 4),
                              'mean_dual': round(gg['dual'].mean(), 4),
                              'dtc_mean%': round(dtc_m, 2),
                              'wilcoxon_p': round(float(w.pvalue), 4) if w else None,
                              'ttest_p': round(float(t.pvalue), 4),
                              'cohen_d': round(diff.mean() / max(diff.std(), 1e-12), 3)})

    # 全量标签表。condition 是情况一的主标签：
    # 单任务=0（正常），双任务=1（双任务模拟异常）。
    # risk/risk15/risk20 仅保留作严重程度敏感性分析。
    all_rows = []
    for sid, act in subs.items():
        for a, lst in act.items():
            for rec in lst:
                if a.startswith('双任务'):
                    r = df[df.file == rec['file']]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    spd = r.dtc_step_speed_ms
                    all_rows.append({'file': rec['file'], 'subject_id': sid, 'group': rec['g'],
                                     'action': a, 'condition': 1,
                                     'risk': int(r.risk), 'class3': r.class3,
                                     'risk15': int(spd > 15) if pd.notna(spd) else 0,
                                     'risk20': int(spd > 20) if pd.notna(spd) else 0})
                else:
                    all_rows.append({'file': rec['file'], 'subject_id': sid, 'group': rec['g'],
                                     'action': a, 'condition': 0,
                                     'risk': 0, 'class3': '单任务',
                                     'risk15': 0, 'risk20': 0})
    al = pd.DataFrame(all_rows)

    # 输出
    out = os.path.join(report_out, 'label_dtc'); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'labels_dtc.csv'), index=False, encoding='utf-8-sig')
    al.to_csv(os.path.join(out, 'labels_all.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(stat_rows).to_csv(os.path.join(out, 'paired_stats.csv'),
                                   index=False, encoding='utf-8-sig')
    sens = {f'speed_dtc>{t}': int((df.dtc_step_speed_ms > t).sum()) for t in (5, 15, 20)}
    summary = {'n_dual_files': len(df), 'n_risk': int(df.risk.sum()),
               'n_condition_normal': int((al.condition == 0).sum()),
               'n_condition_dual': int((al.condition == 1).sum()),
               'n_risk15': int((df.dtc_step_speed_ms > 15).sum()),
               'n_risk20': int((df.dtc_step_speed_ms > 20).sum()),
               'n_no_baseline': len(no_base),
               'speed_thr': SPEED_THR, 'composite_quantile': COMP_Q,
               'composite_threshold_scope': threshold_scope,
               'rule_counts': {'speed10': int(df.rule_speed10.sum()),
                               'composite75': int(df.rule_composite75.sum()),
                               'both': int(((df.rule_speed10 == 1) & (df.rule_composite75 == 1)).sum())},
               'sensitivity_extra': sens,
               'n_labels_all': len(al),
               'risk_by_action': df.groupby('action').risk.agg(['sum', 'count'])
                                   .reset_index().rename(columns={'sum': 'n_risk'}).to_dict('records'),
               'risk_by_group': df.groupby('group').risk.agg(['sum', 'count'])
                                  .reset_index().rename(columns={'sum': 'n_risk'}).to_dict('records'),
               'no_baseline_files': no_base[:50]}
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _boxplots(df, report_out)
    print(f"[label_dtc] 双任务 {len(df)} 个 → 高风险 {int(df.risk.sum())} 个，"
          f"条件标签 正常={int((al.condition == 0).sum())} 双任务={int((al.condition == 1).sum())}，"
          f"无基线跳过 {len(no_base)} 个 → {out}")

def _boxplots(df, report_out):
    adir = os.path.join(report_out, 'assets'); os.makedirs(adir, exist_ok=True)
    for base, dual in PAIRS:
        dd = df[df.action == dual]
        if dd.empty:
            continue
        fig, axs = plt.subplots(2, 3, figsize=(14, 7)); axs = axs.ravel()
        for i, k in enumerate(PARAM_KEYS):
            dd.boxplot(column=f'dtc_{k}', by='group', ax=axs[i])
            axs[i].set_title(f'{k} DTC（>0 = 退化）'); axs[i].set_xlabel('')
            axs[i].axhline(0, color='gray', lw=0.5)
        fig.suptitle(f'{dual}（基线 {base}）— 双任务成本分布')
        fig.savefig(os.path.join(adir, f'dtc_{dual}.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--report_out', default='./gait_report')
    a = p.parse_args(); run(a.dataset, a.report_out)
