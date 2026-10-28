# -*- coding: utf-8 -*-
"""Age-friendly web interface for the straight paired gait model."""
from __future__ import annotations

import io
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT.parents[1]))
from project_runtime import configure as _configure_project_runtime

_configure_project_runtime()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import streamlit as st
import torch

from gait_pipeline_all_in_one import (
    FS, PARAM_KEYS, PARAM_LABELS, SENSOR_COLS, extract_parameters,
)
from modules._common import extract_windows
from paired_deep import FAMILIES, PairNet, _features


MODEL_ROOT = ROOT / "gait_cv_straight"
WINDOW_SIZE = 300
STEP_SIZE = 150
CONTACT_EMAIL = "1297221477@qq.com"
PARAM_NAMES = {
    "step_length_m": "步长（米）",
    "step_speed_ms": "步速（米/秒）",
    "cadence_spm": "步频（步/分）",
    "step_width_ratio": "左右负载不对称度",
    "stability_score": "稳定性评分",
    "symmetry_score": "步态对称性",
}


def _age(birthday: date) -> int:
    today = date.today()
    return today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))


def _read_csv(uploaded) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()), encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"{uploaded.name} 无法读取，请确认它是有效的 CSV 文件。") from exc
    missing = [column for column in SENSOR_COLS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{uploaded.name} 缺少模型需要的传感器列（例如 {', '.join(missing[:4])}）。"
            "请上传项目处理后的 *_filtered.csv 文件。"
        )
    if len(frame) < WINDOW_SIZE:
        raise ValueError(f"{uploaded.name} 只有 {len(frame)} 行；至少需要 {WINDOW_SIZE} 行（约 3 秒）。")
    values = frame[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"{uploaded.name} 的传感器数据含空值或非数字内容，请先检查 CSV。")
    frame.loc[:, SENSOR_COLS] = values.astype(np.float32)
    return frame


def _extract(frame: pd.DataFrame, height_cm: float, calibration_factor: float):
    leg_length_m = height_cm * 0.53 / 100.0
    params = extract_parameters(frame, leg_length_m, calibration_factor, FS)
    if "error" in params:
        raise ValueError(f"无法从这份 CSV 提取完整步态参数：{params['error']}")
    windows = extract_windows(frame, WINDOW_SIZE, STEP_SIZE)
    if not len(windows):
        raise ValueError("CSV 中没有可用于模型分析的完整 3 秒步态窗口。")
    return params, windows


@st.cache_resource(show_spinner="正在载入直线配对模型…")
def _load_models():
    models = []
    for fold in range(5):
        model_dir = MODEL_ROOT / f"fold_{fold}" / "model"
        model_path = model_dir / "paired_deep.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"缺少第 {fold + 1} 折模型：{model_path}")
        with model_path.open("rb") as stream:
            package = __import__("pickle").load(stream)
        model = PairNet(feature_size=len(PARAM_KEYS) * 4 + len(FAMILIES))
        model.load_state_dict(package["state"])
        model.eval()
        models.append({
            "model": model,
            "window_scaler": package["window_scaler"],
            "feature_scaler": package["feature_scaler"],
            "threshold": float(package["threshold"]),
        })
    with (MODEL_ROOT / "summary.json").open(encoding="utf-8") as stream:
        validation = __import__("json").load(stream)
    return models, validation


def _predict_pair(base_windows, current_windows, base_params, current_params):
    models, validation = _load_models()
    features = _features(base_params, current_params, "直线")
    probabilities, votes = [], []
    for item in models:
        scaler = item["window_scaler"]
        base_shape, current_shape = base_windows.shape, current_windows.shape
        base = scaler.transform(base_windows.reshape(-1, base_shape[-1])).reshape(base_shape)
        current = scaler.transform(current_windows.reshape(-1, current_shape[-1])).reshape(current_shape)
        feature = item["feature_scaler"].transform(features.reshape(1, -1))
        base_tensor = torch.as_tensor(base, dtype=torch.float32)
        current_tensor = torch.as_tensor(current, dtype=torch.float32)
        base_index = torch.zeros(len(base_tensor), dtype=torch.long)
        current_index = torch.zeros(len(current_tensor), dtype=torch.long)
        with torch.inference_mode():
            logits = item["model"](
                base_tensor, current_tensor, base_index, current_index, 1,
                torch.as_tensor(feature, dtype=torch.float32),
            )
            probability = float(torch.sigmoid(logits)[0].item())
        probabilities.append(probability)
        votes.append(probability > item["threshold"])
    return {
        "score": float(np.mean(probabilities)),
        "positive_votes": int(np.sum(votes)),
        "thresholds": [item["threshold"] for item in models],
        "validation": validation,
    }


def _calibration(frame, height_cm):
    leg_length_m = height_cm * 0.53 / 100.0
    raw = extract_parameters(frame, leg_length_m, 1.0, FS)
    if "error" in raw or raw.get("n_steps_total", 0) < 2:
        raise ValueError("校准文件无法提取足够的步数；请上传完整的 13.3 米校准步行 CSV。")
    estimated_distance = raw["step_length_m"] * raw["n_steps_total"]
    return float(np.clip(13.30 / max(estimated_distance, 1e-6), 0.3, 3.0))


def _result_text(result):
    if result["positive_votes"] >= 3:
        return "两次测量的变化超过模型参考阈值", "建议在相同条件下复测，并留意日常活动情况；若变化持续或伴有不适，请咨询专业人员。"
    return "两次测量的变化暂未超过模型参考阈值", "模型结果不能替代专业评估；如有不适或跌倒风险，请及时咨询专业人员。"


def _font():
    for candidate in (r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\msyh.ttc"):
        if os.path.exists(candidate):
            return FontProperties(fname=candidate)
    return FontProperties(family="DejaVu Sans")


def _make_pdf(profile, result, base_params, dual_params, base_file, dual_file,
              base_rows, dual_rows, calibration_factor):
    font = _font()
    output = io.BytesIO()
    title_font = FontProperties(fname=font.get_file(), size=22) if font.get_file() else FontProperties(size=22)
    body_font = FontProperties(fname=font.get_file(), size=11) if font.get_file() else FontProperties(size=11)
    headline, followup = _result_text(result)
    score = result["score"] * 100
    bmi = profile["weight"] / ((profile["height"] / 100) ** 2)
    validation = result["validation"].get("aggregate", {})

    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        ax = fig.add_axes([0.10, 0.08, 0.80, 0.84])
        ax.axis("off")
        ax.text(0.0, 0.98, "步态分析报告", fontproperties=title_font,
                color="#1f6b66", va="top")
        ax.text(0.0, 0.92, f"受测者：{profile['name']}    性别：{profile['gender']}    年龄：{_age(profile['birthday'])} 岁",
                fontproperties=body_font, color="#334155", va="top")
        ax.text(0.0, 0.885,
                f"出生日期：{profile['birthday'].isoformat()}    身高：{profile['height']:.0f} cm    体重：{profile['weight']:.1f} kg    BMI：{bmi:.1f}",
                fontproperties=body_font, color="#334155", va="top")
        ax.text(0.0, 0.855, "身高用于估算腿长；性别与体重仅作报告信息，不作为当前模型输入。",
                fontproperties=body_font, color="#64748b", va="top")
        ax.text(0.0, 0.82, "两次普通直线步行的变化评估", fontproperties=body_font,
                color="#64748b", va="top")
        ax.text(0.0, 0.75, f"{score:.1f}%", fontproperties=FontProperties(fname=font.get_file(), size=34) if font.get_file() else FontProperties(size=34),
                color="#1f6b66", va="top")
        ax.text(0.0, 0.685, headline, fontproperties=FontProperties(fname=font.get_file(), size=15) if font.get_file() else FontProperties(size=15),
                color="#1f2937", va="top")
        ax.text(0.0, 0.635, followup, fontproperties=body_font, color="#475569",
                va="top", wrap=True)
        ax.text(0.0, 0.55, "测量文件", fontproperties=body_font, color="#1f6b66", va="top")
        ax.text(0.0, 0.515, f"较早时间点：{base_file}", fontproperties=body_font, color="#334155", va="top", wrap=True)
        ax.text(0.0, 0.48, f"较晚时间点：{dual_file}", fontproperties=body_font, color="#334155", va="top", wrap=True)
        ax.text(0.0, 0.445, f"13.3 米步长校准系数：{calibration_factor:.3f}", fontproperties=body_font, color="#334155", va="top")
        ax.text(0.0, 0.37, "模型分值用于衡量变化与训练参考模式的接近程度，不是患病概率；阈值是模型判别阈值。",
                fontproperties=body_font, color="#475569", va="top", wrap=True)
        auc = validation.get("aggregate", {}).get("auc", {}).get("mean")
        if auc is not None:
            ax.text(0.0, 0.32, f"训练参考：受试者级五折平均 AUC {auc:.3f}。该指标评估普通步态与双任务诱发变化的区分能力，不等同于真实长期随访准确率。",
                    fontproperties=body_font, color="#64748b", va="top", wrap=True)
        ax.text(0.0, 0.23,
                "重要说明：双任务变化仅作为训练参照。当前模型尚未用真实长期随访验证，提示结果用于关注和复测，不作疾病诊断。请尽量保持鞋靴、传感器、路线和采集设置一致。",
                fontproperties=body_font, color="#8a4b08", va="top", wrap=True)
        ax.text(0.0, 0.13, f"报告联系邮箱：{CONTACT_EMAIL}", fontproperties=body_font,
                color="#1f6b66", va="top")
        ax.text(0.0, 0.04, "步态分析平台 · 步态变化报告", fontproperties=body_font,
                color="#94a3b8", va="bottom")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
        ax.axis("off")
        ax.text(0.0, 0.98, "步态参数对比", fontproperties=title_font,
                color="#1f6b66", va="top")
        ax.text(0.0, 0.92, "同一受试者两个时间点的参数摘要",
                fontproperties=body_font, color="#475569", va="top")
        table_rows = []
        for key in PARAM_KEYS:
            base_value = float(base_params.get(key, float("nan")))
            dual_value = float(dual_params.get(key, float("nan")))
            change = dual_value - base_value
            pct = "—" if abs(base_value) < 1e-8 else f"{change / abs(base_value) * 100:+.1f}%"
            table_rows.append([
                PARAM_NAMES.get(key, PARAM_LABELS.get(key, key)),
                f"{base_value:.3f}", f"{dual_value:.3f}", f"{change:+.3f}", pct,
            ])
        table = ax.table(
            cellText=table_rows,
            colLabels=["参数", "较早测量", "较晚测量", "差值", "变化"],
            cellLoc="center", colLoc="center", bbox=[0.0, 0.60, 1.0, 0.25],
            colWidths=[0.30, 0.18, 0.18, 0.17, 0.17],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d8e6e3")
            cell.set_facecolor("#eaf5f2" if row == 0 else ("#f7fbfa" if row % 2 else "white"))
            cell.get_text().set_fontproperties(body_font)
            cell.get_text().set_color("#1f6b66" if row == 0 else "#334155")
        ax.text(0.0, 0.53, "数据概况", fontproperties=body_font,
                color="#1f6b66", va="top")
        ax.text(0.0, 0.49,
                f"较早测量：{len(base_rows)} 个采样点，约 {len(base_rows) / FS:.1f} 秒，{len(extract_windows_from_frame(base_rows))} 个完整窗口",
                fontproperties=body_font, color="#334155", va="top")
        ax.text(0.0, 0.45,
                f"较晚测量：{len(dual_rows)} 个采样点，约 {len(dual_rows) / FS:.1f} 秒，{len(extract_windows_from_frame(dual_rows))} 个完整窗口",
                fontproperties=body_font, color="#334155", va="top")
        ax.text(0.0, 0.36,
                "参数变化百分比以较早测量为参照。不同设备、鞋垫安装方式、行走环境与步态周期会影响参数，建议尽量保持测量条件一致。",
                fontproperties=body_font, color="#64748b", va="top", wrap=True)
        ax.text(0.0, 0.25, f"如需了解报告内容，请联系：{CONTACT_EMAIL}",
                fontproperties=body_font, color="#1f6b66", va="top")
        ax.text(0.0, 0.04, "步态分析平台 · 步态变化报告", fontproperties=body_font,
                color="#94a3b8", va="bottom")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return output.getvalue()


def extract_windows_from_frame(frame):
    return extract_windows(frame, WINDOW_SIZE, STEP_SIZE)


def _analyse(profile, base_upload, dual_upload, calibration_upload=None):
    base_frame = _read_csv(base_upload)
    dual_frame = _read_csv(dual_upload)
    factor = 1.0
    if calibration_upload is not None:
        calibration_frame = _read_csv(calibration_upload)
        factor = _calibration(calibration_frame, profile["height"])
    base_params, base_windows = _extract(base_frame, profile["height"], factor)
    dual_params, dual_windows = _extract(dual_frame, profile["height"], factor)
    result = _predict_pair(base_windows, dual_windows, base_params, dual_params)
    report = _make_pdf(
        profile, result, base_params, dual_params, base_upload.name,
        dual_upload.name, base_frame, dual_frame, factor,
    )
    return result, base_params, dual_params, base_frame, dual_frame, report, factor


def _style():
    st.markdown("""
    <style>
      :root { --ink:#193b3a; --muted:#607876; --teal:#2e837c; --mint:#eaf5f2; --cream:#fbfaf5; }
      html, body, [class*="css"] { font-family: 'Nunito','Microsoft YaHei',sans-serif; }
      .stApp { background: #f7faf8; color:var(--ink); }
      [data-testid="stAppViewContainer"] > .main { background: radial-gradient(circle at 92% 4%, #e2f1eb 0, #f7faf8 28rem); }
      .block-container { max-width:1120px; padding-top:4rem; padding-bottom:3rem; }
      h1,h2,h3,p,label { color:var(--ink); }
      h1 { letter-spacing:-.03em; }
      [data-testid="stMarkdownContainer"] p { font-size:1.08rem; line-height:1.72; }
      [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input { min-height:3.25rem; border-radius:14px; font-size:1.12rem; }
      [data-testid="stSelectbox"] div[data-baseweb="select"] { min-height:3.1rem; border-radius:14px; font-size:1.08rem; }
      [data-testid="stDateInput"] input { min-height:3.15rem; border-radius:14px; font-size:1.08rem; }
      [data-testid="stFileUploader"] section { border:2px dashed #91bdb4; border-radius:18px; background:#fcfefd; padding:1rem; }
      .stButton > button,.stFormSubmitButton > button,.stDownloadButton > button { min-height:3.25rem; border-radius:14px; font-weight:800; font-size:1.05rem; padding:.55rem 1.2rem; }
      .stButton > button[kind="primary"],.stFormSubmitButton > button[kind="primary"] { background:#2e837c; border-color:#2e837c; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #e0ebe7; border-radius:16px; padding:1rem; }
      .hero { border-radius:28px; padding:2.2rem; background:linear-gradient(120deg,#e5f2ed 0%,#f7fbf8 58%,#fffaf0 100%); border:1px solid #dcece5; }
      .pill { display:inline-block; background:#d7eee7; color:#246b63; padding:.42rem .8rem; border-radius:999px; font-size:.95rem; font-weight:800; }
      .soft-card { border:1px solid #e0ebe7; border-radius:20px; background:#fff; padding:1.1rem 1.3rem; }
      .step-number { display:inline-grid; place-items:center; width:2rem; height:2rem; border-radius:50%; color:#fff; background:#398b82; font-weight:800; margin-right:.45rem; }
      a:focus,button:focus,input:focus { outline:3px solid #f1c66b !important; outline-offset:2px !important; }
      @media(max-width:700px) { .hero {padding:1.35rem;} .block-container {padding-left:1rem;padding-right:1rem;} }
    </style>
    """, unsafe_allow_html=True)


def _header(step=None):
    left, right = st.columns([3, 1])
    with left:
        st.markdown("### 🌿 安步 · 步态分析平台")
    if step:
        with right:
            st.markdown(f"<div style='text-align:right;padding-top:1rem;color:#607876'>第 {step} 步 / 3</div>", unsafe_allow_html=True)


def _welcome():
    _header()
    st.markdown("""
    <div class="hero">
      <span class="pill">温和 · 清晰 · 便于阅读</span>
      <h1 style="font-size:2.6rem;line-height:1.2;margin:.8rem 0">了解每一步，<br>从一次安心测量开始</h1>
      <p style="font-size:1.2rem;max-width:680px;color:#496461">上传同一位受试者两个不同时间点的普通直线步行数据，查看步态变化是否超过模型参考阈值，并生成可下载的 PDF 报告。</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.subheader("使用流程")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='soft-card'><span class='step-number'>1</span><b>填写基本信息</b><p>姓名、性别、身高、体重和出生日期。</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='soft-card'><span class='step-number'>2</span><b>上传两份 CSV</b><p>同一位受试者在两个不同时间点的普通直线步行数据。</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='soft-card'><span class='step-number'>3</span><b>查看分析报告</b><p>查看步态变化，并下载 PDF 留存。</p></div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='soft-card'><h3>隐私与使用说明</h3><p><b>隐私提醒：</b>本机启动时，资料由本机处理；在线使用时，您填写的个人信息和上传的步态 CSV 会通过网络发送至托管服务器分析。应用代码不会把测量资料写入长期数据库或文件，但托管平台可能按其规则处理运行数据。请仅在取得本人同意后使用，并避免上传不必要的身份信息。</p><p>本平台不会给出疾病诊断。若您近期跌倒、疼痛或步态明显改变，请联系医生或康复专业人员。</p><p><b>需要帮助？</b> 1297221477@qq.com</p></div>", unsafe_allow_html=True)
    st.write("")
    if st.button("开始使用 →", type="primary", use_container_width=True):
        st.session_state.page = "profile"
        st.rerun()


def _profile():
    _header(1)
    st.title("填写基本信息")
    st.write("这些信息用于生成报告和估算步长。")
    saved_profile = st.session_state.get("profile", {})
    gender_options = ["女", "男", "其他 / 不便透露"]
    saved_gender = saved_profile.get("gender", gender_options[0])
    gender_index = gender_options.index(saved_gender) if saved_gender in gender_options else 0
    with st.form("profile_form"):
        name = st.text_input("姓名", value=saved_profile.get("name", ""), placeholder="请输入姓名")
        col_a, col_b = st.columns(2)
        with col_a:
            gender = st.selectbox("性别", gender_options, index=gender_index)
            height = st.number_input("身高（厘米）", min_value=100, max_value=220, value=int(saved_profile.get("height", 160)), step=1)
        with col_b:
            weight = st.number_input("体重（公斤）", min_value=25.0, max_value=250.0, value=float(saved_profile.get("weight", 60.0)), step=0.5)
            birthday = st.date_input("出生年月日", value=saved_profile.get("birthday", date(1960, 1, 1)), min_value=date(1900, 1, 1), max_value=date.today(), format="YYYY-MM-DD")
        consent = st.checkbox("我了解本页面仅作步态数据观察，不用于疾病诊断。")
        back, forward = st.columns(2)
        with back:
            back_clicked = st.form_submit_button("← 返回欢迎页", use_container_width=True)
        with forward:
            submitted = st.form_submit_button("保存信息，继续上传 CSV →", type="primary", use_container_width=True)
    if back_clicked:
        st.session_state.page = "welcome"
        st.rerun()
    if submitted:
        if not name.strip():
            st.error("请先填写姓名。")
        elif not consent:
            st.error("请确认您已了解报告用途。")
        else:
            st.session_state.profile = {
                "name": name.strip(), "gender": gender, "height": float(height),
                "weight": float(weight), "birthday": birthday,
            }
            st.session_state.pop("analysis", None)
            st.session_state.page = "measure"
            st.rerun()


def _measure():
    profile = st.session_state.profile
    _header(2)
    st.title(f"您好，{profile['name']}。准备开始测量分析了吗？")
    st.write("请按时间顺序上传同一位受试者两次**普通直线步行**测量。")
    st.markdown("<div class='soft-card'><b>两次测量请尽量保持一致</b><br>使用同一受试者、同一鞋靴与传感器设置、相同的普通直线步行任务和相近的采集时长。先上传较早的一次，再上传较晚的一次。</div>", unsafe_allow_html=True)
    st.caption("身高用于估算腿长；性别和体重用于报告展示，当前直线配对模型不把它们作为输入特征。")
    st.write("")
    left, right = st.columns(2)
    with left:
        base_upload = st.file_uploader("① 较早时间点的普通直线步行 CSV", type=["csv"], key="base_csv", help="选择同一受试者较早时间采集的普通直线步行数据。")
    with right:
        dual_upload = st.file_uploader("② 较晚时间点的普通直线步行 CSV", type=["csv"], key="dual_csv", help="选择同一受试者较晚时间采集的普通直线步行数据。")
    calibration_upload = st.file_uploader(
        "可选：13.3 米校准步行 CSV", type=["csv"], key="cal_csv",
        help="如果有该受试者已知 13.3 米往返步行的校准文件，可用于两次测量的步长校准；没有也可以继续。",
    )
    with st.expander("上传前请确认", expanded=False):
        st.markdown("- 请上传项目处理后的 `*_filtered.csv` 文件。\n- 每份数据至少需要 300 行，并包含左右脚鞋垫的 24 个传感器通道。\n- 两份测量应来自同一人，且都为普通直线来回步行。\n- 可选校准文件对应已知 13.3 米往返步行，校准系数会用于两次测量。")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("← 返回修改基本信息", use_container_width=True):
            st.session_state.page = "profile"
            st.rerun()
    with col2:
        run_analysis = st.button("分析步态并生成 PDF 报告", type="primary", use_container_width=True, disabled=not (base_upload and dual_upload))

    if run_analysis:
        st.session_state.pop("analysis", None)
        try:
            with st.spinner("正在检查数据并运行五折直线配对模型，请稍候…"):
                result, base_params, dual_params, base_frame, dual_frame, pdf, factor = _analyse(
                    profile, base_upload, dual_upload, calibration_upload,
                )
            st.session_state.analysis = {
                "result": result, "base_params": base_params,
                "dual_params": dual_params, "base_name": base_upload.name,
                "dual_name": dual_upload.name, "pdf": pdf, "factor": factor,
                "base_duration": len(base_frame) / FS,
                "dual_duration": len(dual_frame) / FS,
                "base_windows": len(extract_windows_from_frame(base_frame)),
                "dual_windows": len(extract_windows_from_frame(dual_frame)),
            }
            st.session_state.page = "report"
            st.rerun()
        except Exception as exc:
            st.error(f"分析没有完成：{exc}")
            st.info("请检查 CSV 格式、动作类型与文件配对。原始测量文件不会被更改。")



def _report():
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.session_state.page = "measure"
        st.rerun()
        return
    _header(3)
    st.title("步态分析报告")
    st.caption(f"受试者：{st.session_state.profile['name']} · 本次报告已生成")
    result = analysis["result"]
    headline, followup = _result_text(result)
    st.markdown(f"### {headline}")
    st.write(f"五个模型中有 **{result['positive_votes']} 个**认为两次测量的变化超过模型判别阈值。模型平均提示分值为 **{result['score'] * 100:.1f}%**；它不是疾病概率或临床诊断结果。")
    st.caption(followup)
    met1, met2, met3 = st.columns(3)
    met1.metric("较早测量时长", f"{analysis['base_duration']:.1f} 秒")
    met2.metric("较晚测量时长", f"{analysis['dual_duration']:.1f} 秒")
    met3.metric("步长校准系数", f"{analysis['factor']:.3f}")
    st.subheader("步态参数对比")
    table_rows = []
    for key in PARAM_KEYS:
        base = float(analysis["base_params"].get(key, np.nan))
        dual = float(analysis["dual_params"].get(key, np.nan))
        table_rows.append({
            "参数": PARAM_NAMES.get(key, PARAM_LABELS.get(key, key)),
            "较早测量": round(base, 3), "较晚测量": round(dual, 3),
            "变化": round(dual - base, 3),
            "变化比例": "—" if abs(base) < 1e-8 else f"{(dual - base) / abs(base) * 100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", st.session_state.profile["name"])
    st.download_button(
        "下载步态分析报告（PDF）", data=analysis["pdf"],
        file_name=f"{safe_name}_步态分析报告.pdf", mime="application/pdf",
        type="primary", use_container_width=True,
    )
    st.caption(f"报告联系邮箱：{CONTACT_EMAIL}。关闭或刷新网页后，本次结果可能不会保留。")
    with st.expander("模型验证背景"):
        validation = result["validation"].get("aggregate", {})
        auc = validation.get("auc", {}).get("mean")
        sensitivity = validation.get("sensitivity", {}).get("mean")
        specificity = validation.get("specificity", {}).get("mean")
        st.write(f"受试者级五折验证：AUC {auc:.3f}，敏感度 {sensitivity:.1%}，特异度 {specificity:.1%}。")
        st.caption("验证测试集每折约 18–20 份配对样本。该验证针对单任务与双任务配对；不同时间点的实际随访变化尚未独立验证。")
    back, restart = st.columns(2)
    with back:
        if st.button("← 返回 CSV 上传页", use_container_width=True):
            st.session_state.page = "measure"
            st.rerun()
    with restart:
        if st.button("开始新的分析", use_container_width=True):
            for key in ("profile", "analysis"):
                st.session_state.pop(key, None)
            st.session_state.page = "welcome"
            st.rerun()


def main():
    st.set_page_config(page_title="安步 · 步态分析平台", page_icon="🌿", layout="wide")
    _style()
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    if st.session_state.page == "welcome":
        _welcome()
    elif st.session_state.page == "profile":
        _profile()
    elif st.session_state.page == "measure" and "profile" in st.session_state:
        _measure()
    else:
        _report()


if __name__ == "__main__":
    main()
