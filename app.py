"""Giao diện web theo đúng luồng của đề (Hình 1):
Data Source -> EDA -> Preprocessing -> Feature -> Model -> Evaluation -> Error Analysis -> Conclusion

Chạy:  streamlit run app.py     (cần có kết quả: python -m spam_filter run)
"""
import html
import re
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # joblib cần import được spam_filter khi nạp mô hình
from spam_filter.config import OUT_FIGURES, OUT_MODELS, OUT_RESULTS, DataConfig  # noqa: E402
from spam_filter.data import load_dataset, split_data  # noqa: E402
from spam_filter.eda import _top_words  # noqa: E402
from spam_filter.error_analysis import error_examples, top_spam_words  # noqa: E402
from spam_filter.experiments import get_experiments  # noqa: E402
from spam_filter.preprocessing import TextCleaner  # noqa: E402

HAM, SPAM = "#4C72B0", "#C44E52"
st.set_page_config(page_title="Spam Filter", page_icon="✉️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* =====================================================================
   HỆ THỐNG GIAO DIỆN
   - đen trắng thuần + các sắc xám; đỏ/xanh chỉ dùng cho dữ liệu (spam/ham, lỗi, trạng thái)
   - 2 font: Source Sans 3 (chữ thường) · Be Vietnam Pro (tiêu đề, con số)
   - Độ đậm: 700 tiêu đề trang · 600 tiêu đề mục, nhãn ô nhập, nhãn thẻ, số liệu · 400 còn lại
   - Khung xám có viền = chỗ thao tác được · khung xám rất nhạt = giải thích · vạch trái = ghi chú
   ===================================================================== */
:root{
  --ink:#1A1A1A; --text:#2B2B2B; --mute:#6B6B6B; --line:#E2E2E2; --line-2:#EEEEEE;
  --accent:#2B2B2B; --accent-dk:#1A1A1A; --a50:#F2F3F5; --a100:#E4E7EB; --a200:#D3D7DC;
  --neutral:#FAFAFA; --side:#F5F5F5; --field:#BDBDBD;
  --f-body:'Source Sans 3','Segoe UI',sans-serif;
  --f-head:'Be Vietnam Pro','Segoe UI',sans-serif;
  --f-mono:'JetBrains Mono',Consolas,monospace;
}

/* ---------- 1. chữ ---------- */
.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp td,.stApp th,
.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stCaptionContainer"],.stApp [data-baseweb="select"] div
  {font-family:var(--f-body)!important}
.stApp h1,.stApp h2,.stApp h3,.side-brand .bt,.ph-badge,[data-testid="stMetricValue"],[data-testid="stMetricValue"] div,
.flow .fs b,.res-prob b,.res-tag,.side-sum .sc-f b,.side-sum .sc-pr b,.errbox .num,.result .rid,.result .rstats b,
.strip b,.cmp b{font-family:var(--f-head)!important}
.stApp [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] *{font-family:var(--f-head)!important}
.stApp code,.stApp pre,.flow.xform .tx{font-family:var(--f-mono)!important}
.stApp{font-size:16px;line-height:1.6;color:var(--text);-webkit-font-smoothing:antialiased}
.stApp b,.stApp strong{font-weight:600}
[data-testid="stMetricValue"],.side-sum b,.strip b,.result b,.cmp,.stApp td{font-variant-numeric:tabular-nums}
a{color:var(--accent)}

/* ---------- 2. khung trang ---------- */
.stApp,[data-testid="stMain"]{background:#fff}
[data-testid="stHeader"]{background:transparent}
[data-testid="stAppDeployButton"],[data-testid="stMainMenu"],#MainMenu,footer{display:none!important}
.block-container{max-width:1100px;padding-top:2rem;padding-bottom:4rem}
[data-testid="stVerticalBlock"]{gap:1rem}
[data-testid="stHorizontalBlock"]{align-items:stretch}
[data-testid="stColumn"]>[data-testid="stVerticalBlock"]{height:100%}
[data-testid="stColumn"] [data-testid="stElementContainer"]:has(>[data-testid="stMetric"]){height:100%}
hr{border-color:var(--line);margin:.6rem 0}
/* Streamlit đặt lề dưới âm (-16px) cho khối markdown → các khối tự vẽ bị dính / tràn khung: trả về 0 */
[data-testid="stMarkdownContainer"]:has(> .ph),[data-testid="stMarkdownContainer"]:has(> .note),
[data-testid="stMarkdownContainer"]:has(> .legend),[data-testid="stMarkdownContainer"]:has(> .msg),
[data-testid="stMarkdownContainer"]:has(> .strip),[data-testid="stMarkdownContainer"]:has(> .flow),
[data-testid="stMarkdownContainer"]:has(> .errgrid),[data-testid="stMarkdownContainer"]:has(> .result),
[data-testid="stMarkdownContainer"]:has(> .res),[data-testid="stMarkdownContainer"]:has(> .cards){margin-bottom:0!important}

/* ---------- 3. thanh bên: nền bạc nhạt, ít đường kẻ, các mục thụt vào hai bên ----------
   mục đang mở = khối xám than chữ trắng, bo nhẹ 8px, không chạm mép (bớt cứng); không bóng, không viền */
[data-testid="stSidebar"],[data-testid="stSidebar"]>div,[data-testid="stSidebarContent"]{background:#EEF0F3!important}
[data-testid="stSidebar"]{border-right:1px solid #DFE2E6!important}
[data-testid="stSidebarUserContent"]{padding:0 0 1.5rem 0!important}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button,[data-testid="stSidebar"] [data-testid="stSidebarHeader"] span{
  color:#6B6F75!important}
/* logo + tên ứng dụng ở góc trên-trái, cùng hàng với nút thu gọn « (kéo lên vùng đầu trống của thanh bên).
   pointer-events:none để khối này không che mất nút « khi nằm chồng lên vùng đầu. */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.side-brand){margin-top:-63px!important;
  margin-bottom:22px!important;pointer-events:none}
.side-brand{display:flex;align-items:center;gap:10px;margin:0;padding:0;color:#1A1A1A}
/* logo: ô vuông xám than (cùng màu mục đang mở), biểu tượng phong thư có dấu tích = thư đã được lọc */
.side-brand .logo{flex:none;width:32px;height:32px;border-radius:8px;background:#2B2B2B;color:#FFFFFF;display:flex;
  align-items:center;justify-content:center}
.side-brand .logo svg{width:18px;height:18px;stroke:#FFFFFF}
.side-brand .bt{white-space:nowrap;font-size:15.5px;font-weight:600;letter-spacing:-.01em;line-height:1.2;color:#1A1A1A}
.side-brand .bt small{display:block;font-family:var(--f-body);font-size:12px;font-weight:400;color:#6B6F75;
  margin-top:2px;letter-spacing:.01em}
/* danh sách trang: khung chứa tràn mép (bù lề ~30px của Streamlit), từng mục thụt 14px hai bên */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]){
  margin:0 -30px!important;width:calc(100% + 60px)!important;max-width:none!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"],
[data-testid="stSidebar"] [data-testid="stButtonGroup"] div:not(button div){width:100%!important;max-width:100%!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"]>div{display:flex!important;flex-direction:column!important;
  flex-wrap:nowrap!important;gap:2px!important;align-items:stretch!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button{width:calc(100% - 28px)!important;margin:0 14px!important;
  min-height:40px!important;justify-content:flex-start!important;padding:0 14px 0 16px!important;border:none!important;
  border-radius:8px!important;background:transparent!important;box-shadow:none!important;transition:background .12s}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button>div{justify-content:flex-start!important;width:100%;gap:12px}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button p{font-size:16px!important;font-weight:400!important;
  color:#2B2B2B!important;text-align:left!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button [data-testid="stIconMaterial"]{font-size:20px!important;
  color:#5F6368!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button:hover{background:rgba(0,0,0,.05)!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-checked="true"]{background:#2B2B2B!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-checked="true"] p{color:#FFFFFF!important;font-weight:600!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-checked="true"] [data-testid="stIconMaterial"]{color:#FFFFFF!important}
/* "Prediction Demo" tách khỏi 8 bước: khoảng trống + một vạch ngắn thụt vào (không chạy hết chiều ngang) */
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button:first-child{margin-bottom:15px!important;position:relative;
  overflow:visible!important}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button:first-child::after{content:"";position:absolute;left:16px;right:16px;
  bottom:-9px;height:1px;background:#DCDFE3}
/* kết quả: khối trắng bo nhẹ, không kẻ dòng; nhãn trái - số phải */
.side-sum{margin:22px -16px 0;padding:12px 16px 8px;background:#FFFFFF;border-radius:8px;font-size:14.5px;color:#4A4F55}
.side-sum .t{font-size:13px;color:#55595F;margin-bottom:4px}
.side-sum .sc,.side-sum .sc-pr{display:block;background:none;border:none;border-radius:0}
.side-sum .sc-m,.side-sum .sc-f,.side-sum .sc-pr div{display:flex;justify-content:space-between;align-items:baseline;
  padding:6px 0;border:none}
.side-sum .sc-f span,.side-sum .sc-pr span{display:inline;font-size:14.5px}
.side-sum b{color:#1A1A1A;font-weight:600;font-size:15.5px;background:none;padding:0;border-radius:0;
  font-family:var(--f-head)!important;font-variant-numeric:tabular-nums}
.side-sum .sc-f b{font-size:21px;color:#1A1A1A;letter-spacing:-.01em;line-height:1.1}

/* ---------- 4. tiêu đề ---------- */
/* 4a. thanh tiêu đề trang (gọn): hàng 1 = ô số nhỏ + tên trang + tiến độ; hàng 2 = câu dẫn; kẻ mảnh bên dưới */
[data-testid="stElementContainer"]:has(.ph){margin:0 0 4px!important}
.ph{display:grid;grid-template-columns:auto minmax(0,1fr) auto;column-gap:16px;row-gap:2px;align-items:center;
  padding:0 0 16px;border-bottom:1px solid var(--line)}
.ph-main{display:contents}
/* ô số: nền bạc, viền mảnh, số xám than — nhẹ nhàng, cân với khối 2 dòng bên cạnh */
.ph-badge{grid-column:1;grid-row:1/3;align-self:center;width:40px;height:40px;border-radius:10px;background:var(--a100);
  border:1px solid var(--a200);color:var(--accent);display:flex;align-items:center;justify-content:center;
  font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}
.ph-badge svg{width:19px;height:19px}
/* tên trang: phần tiếng Anh đậm vừa, phần tiếng Việt sau dấu gạch nhạt và thường */
.stApp h1.ph-title{grid-column:2;grid-row:1;font-size:21px!important;font-weight:600!important;color:var(--ink)!important;
  margin:0!important;padding:0!important;line-height:1.3!important;letter-spacing:-.01em}
.stApp h1.ph-title .vi{font-weight:400;color:#5F6368}
.ph [data-testid="stHeaderActionElements"]{display:none}
.ph .lead{grid-column:2/4;grid-row:2;color:var(--mute)!important;font-size:15.5px!important;line-height:1.5!important;
  margin:0!important}
.ph-step{grid-column:3;grid-row:1;display:flex;align-items:center;gap:10px}
.ph-step small{font-size:13px;color:var(--mute);font-variant-numeric:tabular-nums;white-space:nowrap}
.ph-dots{display:flex;gap:3px}
.ph-dots i{width:12px;height:4px;border-radius:2px;background:#D6D6D6}
.ph-dots i.on{background:var(--accent)}
/* 4b. tiêu đề mục (st.subheader): chữ navy, gạch chân mảnh chạy hết chiều ngang */
.stApp h2,.stApp h3{font-size:19px!important;font-weight:600!important;line-height:1.35!important;color:var(--accent-dk)!important;
  letter-spacing:-.01em;padding:0!important}
[data-testid="stHeading"]:has(h3){margin:1.9rem 0 .2rem!important;padding:0 0 9px!important;border-bottom:1px solid var(--a200)}
[data-testid="stHeading"]:has(h3) [data-testid="stMarkdownContainer"]{margin-bottom:0!important}
[data-testid="stHeading"] [data-testid="stHeaderActionElements"]{display:none}

/* ---------- 5. đoạn văn, chú thích, thông báo ---------- */
[data-testid="stMarkdownContainer"] p{margin:0 0 .6rem}
[data-testid="stMarkdownContainer"] p:last-child{margin-bottom:0}
.stApp [data-testid="stMarkdownContainer"] code{font-size:.85em;padding:1px 5px;border-radius:4px;background:#F0F0F0;
  color:#2B2B2B;white-space:nowrap}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{font-size:15px!important;color:var(--mute)!important}
[data-testid="stAlertContainer"]{background:#F3F3F3!important;border:1px solid var(--line)!important;border-radius:8px!important}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){background:#FFF7E6!important;border-color:#F0D9A8!important}
[data-testid="stAlertContainer"] p{color:var(--text)!important;font-size:15.5px!important}
[data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"]{margin-bottom:0!important}
/* khung chú thích khi bấm/rê vào dấu "?": bỏ lề âm để chữ không bị cắt nửa dưới */
[data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"]{margin-bottom:0!important}
[data-testid="stTooltipContent"]{padding:8px 12px!important;max-width:340px}
[data-testid="stTooltipContent"] p{font-size:14.5px!important;line-height:1.5!important;color:var(--text)!important;margin:0!important}

/* ---------- 6. ô nhập: nhãn đậm, viền rõ, nền trắng — nổi trên mọi khung ---------- */
[data-testid="stWidgetLabel"]{margin-bottom:5px}
[data-testid="stWidgetLabel"] p{font-size:15px!important;font-weight:600!important;color:var(--ink)!important}
[data-testid="stTextAreaRootElement"],[data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] div:has(> input),[data-testid="stMultiSelect"] div:has(> input){
  background:#fff!important;border:1px solid var(--field)!important;border-radius:8px!important;
  transition:border-color .15s,box-shadow .15s}
[data-testid="stTextAreaRootElement"]:hover,[data-testid="stTextInputRootElement"]:hover,
[data-testid="stSelectbox"] div:has(> input):hover{border-color:#8A8A8A!important}
[data-testid="stTextAreaRootElement"]:focus-within,[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stSelectbox"] div:has(> input):focus-within{border-color:var(--accent)!important;
  box-shadow:0 0 0 3px rgba(0,0,0,.12)!important}
[data-testid="stTextAreaRootElement"] *,[data-testid="stTextInputRootElement"] *{background:transparent!important}
/* chỉ áp cho ô nhập của Streamlit — KHÔNG áp cho ô nổi xem chi tiết khi bấm vào ô của bảng cuộn (cũng là textarea) */
[data-testid="stTextAreaRootElement"] textarea{font-size:16px!important;line-height:1.6!important;padding:12px 14px!important;
  color:var(--ink)!important}
[data-testid="stTextAreaRootElement"] textarea::placeholder,[data-testid="stTextInputRootElement"] input::placeholder{
  color:#9A9A9A!important;opacity:1}
[data-testid="stSelectbox"] div:has(> input){min-height:42px}
[data-testid="stSelectbox"] div:has(> input) *{font-size:16px!important;color:var(--ink)}
/* lựa chọn trong ô tick / nút radio: chữ thường, không đậm */
[data-testid="stCheckbox"] p,[data-testid="stRadio"] [role="radiogroup"] p{font-size:15.5px!important;font-weight:400!important;
  color:var(--ink)!important}
[data-testid="stSliderThumbValue"]{font-size:14px!important;font-weight:600!important;color:var(--accent)!important}
[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"]{font-size:13px!important;color:var(--mute)!important}

/* ---------- 7. khung ---------- */
/* 7a. khung thao tác (có ô nhập / lựa chọn): nền xanh rất nhạt, viền xanh */
.st-key-demo_in,.st-key-pre_tool,.st-key-eval_tool,.st-key-err_tool,.st-key-ds_tool,.st-key-eda_tool,.st-key-batch_tool{
  background:var(--a50)!important;border:1px solid var(--a200)!important;border-radius:10px!important;
  padding:16px 20px 18px!important}
:is(.st-key-demo_in,.st-key-pre_tool,.st-key-eval_tool,.st-key-err_tool,.st-key-ds_tool,.st-key-eda_tool,.st-key-batch_tool,
    .st-key-feat_intro,.st-key-eval_intro,.st-key-concl_lim) [data-testid="stMarkdownContainer"]{margin-bottom:0!important}
.st-key-eval_tool{gap:10px!important}
.st-key-eval_tool .strip{border:none;border-top:1px solid var(--a200);padding:10px 0 0;margin-top:2px}
.st-key-pre_tool .flow.xform,.st-key-ds_tool [data-testid="stDataFrame"],.st-key-err_tool [data-testid="stTable"]{background:#fff}
/* 7b. khung giải thích (chỉ để đọc): nền xám rất nhạt, các ý ngăn bằng vạch dọc */
.st-key-feat_intro,.st-key-eval_intro,.st-key-concl_lim{background:var(--neutral);border:1px solid var(--line);
  border-radius:10px;padding:14px 20px!important}
:is(.st-key-feat_intro,.st-key-eval_intro,.st-key-concl_lim) .cards{column-gap:20px}
:is(.st-key-feat_intro,.st-key-eval_intro,.st-key-concl_lim) .card{border-bottom:none}
:is(.st-key-feat_intro,.st-key-eval_intro,.st-key-concl_lim) .card+.card{border-left:1px solid var(--line);padding-left:20px}
.st-key-concl_lim{margin-bottom:12px}

/* ---------- 8. thẻ chữ, ghi chú, dải số ---------- */
.cards{display:grid;grid-template-columns:repeat(var(--cols,2),minmax(0,1fr));column-gap:28px}
.card{padding:11px 0;border-bottom:1px solid var(--line-2)}
.card.wide{grid-column:1/-1}
.card .lbl{font-size:14px;font-weight:600;color:var(--accent);margin-bottom:3px}
.card .body{font-size:15.5px;line-height:1.6;color:var(--text)}
.note{border-left:3px solid var(--accent);padding:3px 0 3px 14px;color:var(--text)}
.note .lbl{font-weight:600;color:var(--accent-dk);display:inline}
.note .lbl::after{content:": "}
.note .body{display:inline;font-size:15.5px;line-height:1.6}
.cap-lbl{font-size:14px;font-weight:600;color:var(--ink);margin:.3rem 0 -.6rem}
.strip{display:flex;flex-wrap:wrap;gap:6px 28px;align-items:baseline;padding:10px 0;border-top:1px solid var(--line-2);
  border-bottom:1px solid var(--line-2)}
.strip .lead-cell{color:var(--mute)}
.strip small{color:var(--mute);margin-right:7px;font-size:14.5px}
.strip b{font-size:19px;font-weight:600;color:var(--accent-dk)}
.legend{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:14.5px;color:var(--text);margin-top:-4px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.legend .hint{color:var(--mute)}

/* ---------- 9. số liệu, biểu đồ, ảnh ---------- */
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:10px;padding:13px 18px;height:100%}
[data-testid="stMetricLabel"] p{font-size:14px!important;font-weight:400!important;color:var(--mute)!important}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] div{font-size:28px!important;font-weight:600!important;
  color:var(--accent-dk)!important;line-height:1.2}
[data-testid="stPlotlyChart"]{border:1px solid var(--line);border-radius:10px;padding:8px 0 2px;overflow:hidden;background:#fff}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:10px!important}
[data-testid="stExpander"] summary p{font-weight:500!important;color:var(--ink)!important}
[data-testid="stImage"] img{border:1px solid var(--line);border-radius:8px}
[data-testid="stCode"] pre{border:1px solid var(--line);border-radius:8px}

/* ---------- 10. bảng tĩnh (st.table): dòng tiêu đề bạc, dòng kẻ xen kẽ ---------- */
[data-testid="stTable"]{border:1px solid var(--a200);border-radius:10px;overflow:hidden;background:#fff}
[data-testid="stTable"] table{width:100%;border-collapse:collapse;font-size:15px}
[data-testid="stTable"] th{background:var(--a100)!important;color:var(--ink)!important;border-bottom:1px solid var(--a200)!important;font-weight:600;font-size:14px;
  text-align:left;padding:10px 12px;border:none}
[data-testid="stTable"] th *{color:var(--ink)!important}
[data-testid="stTable"] td{padding:9px 12px;border-bottom:1px solid var(--line-2);vertical-align:top;color:var(--text)}
[data-testid="stTable"] tbody tr:nth-child(even) td{background:#FAFAFA}
[data-testid="stTable"] tbody tr:hover td{background:#F0F0F0}
[data-testid="stTable"] tr:last-child td{border-bottom:none}
[data-testid="stTable"] td,[data-testid="stTable"] th{white-space:nowrap}
[data-testid="stTable"] td div,[data-testid="stTable"] td p,[data-testid="stTable"] th p,
[data-testid="stTable"] th div{white-space:inherit!important;margin:0}
/* cột chữ dài được xuống dòng; cột số căn phải */
.st-key-tbl_model td:nth-child(3),.st-key-tbl_model td:nth-child(4){white-space:normal}
.st-key-tbl_model td:nth-last-child(-n+2),.st-key-tbl_model th:nth-last-child(-n+2){text-align:right}
.st-key-tbl_abl td:nth-child(2),.st-key-tbl_eval td:nth-child(2){white-space:normal}
.st-key-tbl_abl td:nth-child(n+3),.st-key-tbl_abl th:nth-child(n+3),
.st-key-tbl_eval td:nth-child(n+3),.st-key-tbl_eval th:nth-child(n+3){text-align:right}
.st-key-tbl_feat th{white-space:normal!important;vertical-align:bottom}
.st-key-tbl_feat td:nth-child(n+4),.st-key-tbl_feat th:nth-child(n+4){text-align:right}
.st-key-tbl_chk td:nth-child(1),.st-key-tbl_chk td:nth-child(2){white-space:normal}
/* bảng tin lỗi: hiện đủ cả câu, cuộn dọc nếu dài, dòng tiêu đề dính trên cùng */
.st-key-err_tool [data-testid="stTable"]{max-height:460px;overflow-y:auto}
.st-key-err_tool [data-testid="stTable"] th{position:sticky;top:0;z-index:1}
.st-key-err_tool td:first-child{white-space:normal;line-height:1.5}
.st-key-err_tool td:last-child,.st-key-err_tool th:last-child{text-align:right;width:70px}

/* ---------- 11. thành phần riêng từng trang ---------- */
/* Demo: tin nhắn tô màu từng từ */
.msg{line-height:2;font-size:16px;padding:44px 20px 16px;border:1px solid var(--line);border-radius:10px;position:relative;
  background:#fff}
.msg::before{content:"Tin nhắn";position:absolute;left:20px;top:12px;font-size:14px;font-weight:600;color:var(--ink)}
.msg::after{content:"";position:absolute;left:0;right:0;top:36px;border-top:1px solid var(--line-2)}
.msg mark{padding:2px 4px;border-radius:3px;color:inherit}
/* Demo: khung kết quả */
.res{border:1px solid var(--line);border-radius:10px;padding:20px 22px 18px;height:100%;background:#fff}
.res-tag{display:inline-block;padding:4px 12px;border-radius:4px;color:#fff;font-size:14px;font-weight:700;letter-spacing:.08em}
.res-prob{margin:18px 0 14px}
.res-prob small{display:block;font-size:14px;color:var(--mute)}
.res-prob b{font-size:46px;font-weight:600;line-height:1.1;color:var(--ink);font-variant-numeric:tabular-nums}
.res-bar{position:relative;height:8px;background:#EDEFF1;border-radius:4px}
.res-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px}
.res-bar u{position:absolute;top:-5px;bottom:-5px;width:2px;margin-left:-1px;background:var(--ink)}
.res-scale{position:relative;display:flex;justify-content:space-between;margin-top:8px;padding-bottom:20px;
  font-size:13px;color:var(--mute)}
.res-scale .thr{position:absolute;top:19px;color:var(--ink);white-space:nowrap}
/* Data Source / Preprocessing: dòng chảy dữ liệu */
.flow{display:flex;align-items:stretch;border:1px solid var(--line);border-radius:10px;margin-bottom:16px;overflow:hidden;
  background:#fff}
.flow .fs{flex:1;padding:14px 18px}
.flow .fs small{display:block;font-size:14px;color:var(--mute)}
.flow .fs b{font-size:28px;font-weight:600;line-height:1.25;color:var(--accent-dk);font-variant-numeric:tabular-nums}
.flow .fs.split{flex:1.6;display:flex;padding:0}
.flow .fs.split div{flex:1;padding:14px 18px}
.flow .fs.split div+div{border-left:1px solid var(--line-2)}
.flow .fa{flex:none;width:118px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;
  font-size:13px;color:var(--mute);border-left:1px solid var(--line-2);border-right:1px solid var(--line-2);background:var(--a50)}
.flow .fa::after{content:"→";font-size:20px;line-height:1;color:var(--accent)}
.flow.xform{margin-bottom:0}
.flow.xform .fa{width:56px}
.flow.xform .tx{margin-top:6px;font-size:14px;line-height:1.75;color:var(--ink);word-break:break-word}
.flow.xform mark{background:#E6E6E6;color:var(--accent-dk);padding:1px 4px;border-radius:3px}
/* Error Analysis: số lượng + nhận xét trong cùng một ô */
.errgrid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.errbox{border:1px solid var(--line);border-radius:10px;padding:13px 18px;background:#fff}
.errbox .lbl{color:var(--mute);font-size:14px}
.errbox .num{font-size:28px;font-weight:600;line-height:1.25;color:var(--accent-dk);font-variant-numeric:tabular-nums}
.errbox.fp .num{color:#B3261E}
.errbox .why{margin-top:10px;padding-top:9px;border-top:1px solid var(--line-2);font-size:15.5px;line-height:1.6;color:var(--text)}
/* Conclusion: khối kết quả + 3 ô so sánh */
.result{background:var(--a50);border:1px solid var(--a200);border-radius:10px;padding:18px 22px}
.result .rhead{margin-bottom:12px}
.result .rid{font-size:24px;font-weight:700;color:var(--accent);margin-right:8px}
.result .rdesc{font-size:16px;color:var(--text)}
.result .rstats{display:flex;flex-wrap:wrap;gap:6px 32px}
.result .rstats small{color:var(--mute);margin-right:7px;font-size:14.5px}
.result .rstats b{font-size:21px;font-weight:600;color:var(--accent-dk)}
.st-key-concl_cmp .cap-lbl{margin:.6rem 0 .5rem}
.st-key-concl_cmp .cards{column-gap:16px}
.st-key-concl_cmp .card{border:1px solid var(--line);border-radius:10px;padding:10px 16px;background:#fff}
.st-key-concl_cmp .card .lbl{color:var(--mute);font-weight:400}
.cmp{font-size:18px;color:var(--text)}
.cmp .arr{color:var(--mute);margin:0 7px}
.cmp b{font-weight:600;color:var(--accent-dk)}
.cmp .delta{color:var(--mute);margin-left:7px;font-size:16px}

/* ---------- 12. nút ---------- */
.stButton button{border-radius:8px;min-height:42px}
.stButton button p{font-size:15.5px!important;font-weight:600!important}
.stButton button[kind="secondary"]{border:1px solid var(--field);background:#fff;color:var(--ink)}
.stButton button[kind="secondary"]:hover{background:var(--a50);border-color:var(--accent);color:var(--accent)}
.stButton button[kind="primary"]{background:var(--accent);border-color:var(--accent)}
.stButton button[kind="primary"]:hover{background:var(--accent-dk);border-color:var(--accent-dk)}
button:focus-visible,[role="slider"]:focus-visible{outline:2px solid var(--accent)!important;outline-offset:2px}

/* ---------- 13. màn hình nhỏ ---------- */
@media (max-width:640px){
  .block-container{padding-left:1rem;padding-right:1rem;padding-top:1.2rem}
  .ph-step{display:none}.ph-badge{width:28px;height:28px;font-size:14px;grid-row:1}
  .stApp h1.ph-title{font-size:19px!important}.ph .lead{grid-column:1/4}
  [data-testid="stMetricValue"],[data-testid="stMetricValue"] div{font-size:22px!important}
  .cards,.errgrid{grid-template-columns:1fr}
  :is(.st-key-feat_intro,.st-key-eval_intro,.st-key-concl_lim) .card+.card{border-left:none;padding-left:0}
  .flow{flex-direction:column}
  .flow .fa{width:auto;flex-direction:row;padding:4px;border:none;border-top:1px solid var(--line-2);
    border-bottom:1px solid var(--line-2)}
  .flow .fa::after{content:"↓"}
}
</style>""", unsafe_allow_html=True)

STEPS = ["Prediction Demo", "1. Data Source", "2. EDA", "3. Preprocessing", "4. Feature",
         "5. Model", "6. Evaluation", "7. Error Analysis", "8. Conclusion"]
EXAMPLES = {
    "Spam - trúng thưởng": "Free entry! Call 09061701461 now to claim your £500 prize. Text STOP to opt out.",
    "Spam - khuyến mãi": "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt WIN to 80086",
    "Ham - đời thường": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "Ham - ngắn": "Ok lar... Joking wif u oni...",
    # Tin tự viết, NGOÀI dataset - để thử khả năng tổng quát hóa
    "[Ngoài dataset] Spam - giả mạo ngân hàng": "Your bank account has been suspended. Verify now at http://secure-bank-login.xyz to avoid closure.",
    "[Ngoài dataset] Spam - phí giao hàng": "We tried to deliver your package but failed. Pay the 1.99 redelivery fee at bit.ly/3xYz to reschedule.",
    "[Ngoài dataset] Spam - iPhone miễn phí": "Congratulations! You have been selected to receive a free iPhone 15. Click here to claim your gift today.",
    "[Ngoài dataset] Spam - việc làm online": "Earn $500 per day working from home! No experience needed. Message us on WhatsApp now.",
    "[Ngoài dataset] Spam - cảnh báo tài khoản": "URGENT: unusual activity detected on your account. Confirm your identity immediately: tinyurl.com/acc-verify",
    "[Ngoài dataset] Ham - nhắc lịch khám": "Your appointment with Dr. Smith is confirmed for Tuesday at 3:30 PM. Reply C to confirm or call the clinic to reschedule.",
    "[Ngoài dataset] Ham - giao hàng thật": "Your Royal Mail parcel will be delivered tomorrow between 9am and 1pm. Track it on the Royal Mail app.",
    "[Ngoài dataset] Ham - mã xác thực": "Your verification code is 482913. Do not share it with anyone.",
    "[Ngoài dataset] Ham - họp công ty": "Hi Anna, the meeting has moved to 2pm in Room 4. Please bring the printed report. Thanks, Mark",
}


# ---------------------------------------------------------------- dữ liệu / mô hình
@st.cache_data
def get_data():
    cfg = DataConfig()
    df = load_dataset(cfg)
    raw_n = len(load_dataset(replace(cfg, drop_duplicates=False)))
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    return cfg, df, raw_n, X_train, X_test, y_train, y_test


@st.cache_resource
def get_models():
    return {p.stem: joblib.load(p) for p in sorted(OUT_MODELS.glob("*.joblib"))}


@st.cache_data
def get_csv(name):
    path = OUT_RESULTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def test_scores(model_id):
    pipe = get_models()[model_id]
    return pipe.predict_proba(get_data()[4])[:, 1]


metrics = get_csv("metrics.csv")
models = get_models()
if metrics is None or not models:
    st.error("Chưa có mô hình. Chạy `python -m spam_filter run` rồi mở lại trang này.")
    st.stop()
cfg, df, raw_n, X_train, X_test, y_train, y_test = get_data()
ids = metrics["id"].tolist()
best_file = OUT_RESULTS / "best_model.txt"
best_id = best_file.read_text().strip() if best_file.exists() else ids[0]
M = metrics.set_index("id")
specs = {s.id: s for s in get_experiments(None)}


def pick_model(label="Mô hình", key=None):
    return st.selectbox(label, ids, index=ids.index(best_id), key=key,
                        format_func=lambda i: f"{i} - {M.loc[i, 'description']}" + (" (tốt nhất)" if i == best_id else ""))


def fig_style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                      title_font=dict(family="Be Vietnam Pro, sans-serif", size=15, color="#1A1A1A"))
    return fig


def batch_test(mid, thr):
    """Dán nhiều tin (mỗi dòng một tin, tùy chọn tiền tố spam| hoặc ham|) -> bảng dự đoán."""
    st.subheader("Test hàng loạt")
    with st.container(border=True, key="batch_tool"):  # khung thao tác
        st.caption("Dán nhiều tin nhắn, mỗi dòng một tin; thêm nhãn mong đợi ở đầu dòng bằng `spam|` hoặc `ham|` "
                   "để đếm số tin đoán đúng.")
        batch = st.text_area("Danh sách tin", height=150, key="batch",
                             placeholder="spam|You won a free prize, call 09061234567\nham|See you at lunch tomorrow")
    rows = []
    for line in filter(str.strip, batch.splitlines()):
        head, sep, rest = line.partition("|")
        if sep and head.strip().lower() in ("spam", "ham"):
            rows.append((head.strip().lower(), rest.strip()))
        else:
            rows.append(("", line.strip()))
    if not rows:
        return
    probs = models[mid].predict_proba([m for _, m in rows])[:, 1]
    pred = np.where(probs >= thr, "spam", "ham")
    out = pd.DataFrame({"tin": [m for _, m in rows], "mong đợi": [w or "–" for w, _ in rows],
                        "P(spam)": probs, "dự đoán": pred})
    out["đúng?"] = ["–" if not w else "✅" if w == q else "❌" for (w, _), q in zip(rows, pred)]
    st.dataframe(out.style.format({"P(spam)": "{:.2f}"}), width="stretch", hide_index=True)
    judged = out[out["mong đợi"] != "–"]
    if len(judged):
        st.write(f"Đúng **{(judged['đúng?'] == '✅').sum()}/{len(judged)}** tin có nhãn mong đợi "
                 f"(mô hình {mid}, ngưỡng {thr:.2f}).")


# ---------------------------------------------------------------- điều hướng
_b = M.loc[best_id]


def _go(step):
    st.session_state.nav = step


def _keep_nav():
    """Bấm lại mục đang mở sẽ bỏ chọn nó (tính năng của st.pills) → chọn lại để dải navy không biến mất."""
    if st.session_state.get("nav") is None:
        st.session_state.nav = st.session_state.get("_page", STEPS[0])


for _k in ("nav", "_page"):  # tên mục cũ còn lưu trong phiên đang mở (vd "Demo dự đoán") -> về trang đầu
    if st.session_state.get(_k) not in (None, *STEPS):
        st.session_state[_k] = STEPS[0]
st.session_state.setdefault("nav", STEPS[0])
ICONS = {STEPS[0]: ":material/mail:", STEPS[1]: ":material/database:", STEPS[2]: ":material/query_stats:",
         STEPS[3]: ":material/cleaning_services:", STEPS[4]: ":material/data_array:",
         STEPS[5]: ":material/model_training:", STEPS[6]: ":material/fact_check:",
         STEPS[7]: ":material/report:", STEPS[8]: ":material/flag:"}
with st.sidebar:
    # logo: phong thư có dấu tích (thư đã được kiểm tra / lọc)
    st.markdown('<div class="side-brand"><span class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M22 13V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v12c0 1.1.9 2 2 2h8"/>'
                '<path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/><path d="m16 19 2 2 4-4"/></svg></span>'
                '<span class="bt">Spam Filter<small>SMS Spam Classification</small></span></div>',
                unsafe_allow_html=True)
    _sel = st.pills("Navigation", STEPS, key="nav", label_visibility="collapsed", on_change=_keep_nav,
                    format_func=lambda o: f"{ICONS[o]} {o}")
    st.markdown(
        # Khối điểm nhấn: F1 to ở giữa, mô hình ở trên, Precision/Recall chia đôi bên dưới
        '<div class="side-sum"><div class="t">Test set results: ' + f'{len(X_test):,} messages</div>'
        '<div class="sc">'
        f'<div class="sc-m" title="Chọn theo F1-spam cross-validation trên tập train, không dùng tập test">'
        f'<span>Best model</span><b>{best_id}</b></div>'
        f'<div class="sc-f" title="Trung bình điều hòa của Precision và Recall cho lớp spam">'
        f'<span>F1 (spam)</span><b>{_b["f1"]:.3f}</b></div>'
        '<div class="sc-pr">'
        f'<div title="Trong các tin bị báo là spam, bao nhiêu % đúng là spam (cao = ít chặn nhầm tin thật)">'
        f'<span>Precision</span><b>{_b["precision"]:.3f}</b></div>'
        f'<div title="Trong các tin spam thật, bao nhiêu % bắt được (cao = ít spam lọt qua)"><span>Recall</span>'
        f'<b>{_b["recall"]:.3f}</b></div></div></div></div>', unsafe_allow_html=True)
page = _sel or st.session_state.get("_page", STEPS[0])  # bấm lại mục đang chọn không làm mất trang
st.session_state["_page"] = page

LEADS = [
    "Nhập hoặc chọn một tin nhắn tiếng Anh, mô hình sẽ cho biết đó là spam hay tin bình thường và vì sao.",
    "Dữ liệu lấy từ đâu, làm sạch thế nào và chia train/test ra sao.",
    "Phân bố lớp, độ dài tin nhắn và các dấu hiệu thường gặp ở tin spam.",
    "Các bước làm sạch văn bản và tác động của từng bước lên kết quả.",
    "Cách biến văn bản thành vector số: Bag-of-Words, TF-IDF, n-gram.",
    "Các thuật toán đã thử, tham số, và vì sao chọn mô hình tốt nhất.",
    "Chất lượng trên tập test: Precision, Recall, F1, confusion matrix và ROC.",
    "Những tin mô hình đoán sai, và vì sao.",
    "Tóm tắt kết quả, kiểm tra độ tin cậy và hướng phát triển.",
]


BEST_CELL = "background-color:#D9DDE2;color:#1A1A1A;font-weight:600"


def table_style(styler, best_cols):
    """Kiểu chung cho bảng so sánh: ô cao nhất mỗi cột nổi bật, dòng mô hình được chọn tô nền."""
    df = styler.data
    if "id" in df.columns:  # dòng mô hình được chọn: nền xanh rất nhạt, mã id đậm (áp trước để ô cao nhất đè lên)
        styler = styler.apply(lambda row: ["background-color:#F0F0F0" if row["id"] == best_id
                                           else "" for _ in row], axis=1)
        # mã id của mô hình được chọn: thêm vạch đậm bên trái ô để mắt tìm thấy dòng này ngay
        styler = styler.map(lambda v: "color:#1A1A1A;font-weight:600;box-shadow:inset 3px 0 0 #2B2B2B"
                            if v == best_id else "", subset=["id"])
    return styler.highlight_max(subset=best_cols, props=BEST_CELL).hide(axis="index")


def page_head(title):
    """Thanh tiêu đề trang: ô số bước | tên trang + câu dẫn | tiến độ Bước x / 8 (chữ giữ nguyên)."""
    num, sep, rest = title.partition(" · ")
    is_step = bool(sep) and num.isdigit()
    name = rest if is_step else title
    # tách "Tên tiếng Anh - phần tiếng Việt" (dấu gạch ngang): phần sau dấu gạch hiển thị nhạt hơn.
    # Giữ nguyên đúng chữ và dấu gạch như đã viết.
    m = re.match(r"(.+?)(\s+-\s*)(.+)", name)
    title_html = (f'{html.escape(m[1])}<span class="vi">{html.escape(m[2])}{html.escape(m[3])}</span>'
                  if m else html.escape(name))
    if is_step:
        badge = f'<div class="ph-badge">{num}</div>'
        total = len(STEPS) - 1
        dots = "".join(f'<i class="{"on" if k <= int(num) else ""}"></i>' for k in range(1, total + 1))
        side = f'<div class="ph-step"><small>Bước {num} / {total}</small><div class="ph-dots">{dots}</div></div>'
    else:  # trang Demo: biểu tượng phong thư thay cho số, không có tiến độ
        badge = ('<div class="ph-badge"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/></svg></div>')
        side = ""
    st.markdown(f'<div class="ph">{badge}<div class="ph-main"><h1 class="ph-title">{title_html}</h1>'
                f'<p class="lead">{LEADS[STEPS.index(page)]}</p></div>{side}</div>', unsafe_allow_html=True)


# ---- thành phần hiển thị chữ (giữ nguyên nội dung, chỉ đổi cách trình bày)
def cards(items, cols=2):
    """items: [(nhãn hoặc "", nội dung HTML[, "wide"])] -> lưới thẻ."""
    html_ = "".join(f'<div class="card{" wide" if len(it) > 2 else ""}">'
                    + (f'<div class="lbl">{it[0]}</div>' if it[0] else "")
                    + f'<div class="body">{it[1]}</div></div>' for it in items)
    st.markdown(f'<div class="cards" style="--cols:{cols}">{html_}</div>', unsafe_allow_html=True)


def note(label, body):
    """Khung ghi chú (nhãn tùy chọn)."""
    lbl = f'<div class="lbl">{label}</div>' if label else ""
    st.markdown(f'<div class="note">{lbl}<div class="body">{body}</div></div>', unsafe_allow_html=True)


def strip(items, lead=None):
    """Dải số liệu ngắn: [(nhãn, giá trị)], có thể kèm câu dẫn ở ô đầu."""
    head = f'<span class="lead-cell">{lead}</span>' if lead else ""
    st.markdown('<div class="strip">' + head + "".join(f'<span><small>{k}</small><b>{v}</b></span>' for k, v in items)
                + '</div>', unsafe_allow_html=True)


def cmp_card(label, old, new, delta=None):
    """Thẻ so sánh trước -> sau (delta: chuỗi chênh lệch, nếu có)."""
    tail = f'<span class="delta">{delta}</span>' if delta else ""
    return (label, f'<span class="cmp">{old}<span class="arr">→</span><b>{new}</b>{tail}</span>')


# ================================================================ DEMO
if page == STEPS[0]:
    page_head("Kiểm tra một tin nhắn")
    c_in, c_out = st.columns([3, 2])
    with c_in, st.container(border=True, key="demo_in"):  # khung thao tác: chọn/nhập tin + mô hình + ngưỡng
        ex = st.selectbox("Ví dụ mẫu", ["(tự nhập)"] + list(EXAMPLES), index=1,
                          help="Chọn “(tự nhập)” để gõ tin nhắn của bạn.")
        text = st.text_area("Nhập tin nhắn (tiếng Anh)", height=130, key=f"t_{ex}",
                            value="" if ex == "(tự nhập)" else EXAMPLES[ex])
        # mô hình + ngưỡng xếp dọc: tên mô hình dài hiện đủ, không bị cắt
        mid = pick_model(key="demo_model")
        thr = st.slider("Ngưỡng quyết định", 0.05, 0.95, 0.5, 0.05,
                        help="Tin bị coi là spam nếu P(spam) ≥ ngưỡng. Hạ ngưỡng → bắt nhiều spam hơn "
                             "nhưng dễ chặn nhầm ham.")
    if not text.strip():
        st.info("Nhập một tin nhắn để xem dự đoán.")
        batch_test(mid, thr)
        st.stop()

    pipe = models[mid]
    p = float(pipe.predict_proba([text])[0, 1])
    is_spam = p >= thr
    with c_out:
        col = SPAM if is_spam else HAM
        st.markdown(
            '<div class="res">'
            f'<span class="res-tag" style="background:{col}">{"SPAM" if is_spam else "HAM"}</span>'
            f'<div class="res-prob"><small>Xác suất spam</small><b>{p * 100:.1f}%</b></div>'
            f'<div class="res-bar"><i style="width:{p * 100:.1f}%;background:{col}"></i>'
            f'<u style="left:{thr * 100:.0f}%" title="Ngưỡng {thr:.2f}"></u></div>'
            # nhãn ngưỡng ở hàng riêng; dịch theo tỉ lệ thr để không tràn ra ngoài ở 2 đầu
            f'<div class="res-scale"><span>0%</span><span class="thr" style="left:{thr * 100:.0f}%;'
            f'transform:translateX(-{thr * 100:.0f}%)">ngưỡng {thr:.2f}</span>'
            '<span>100%</span></div>'
            '</div>', unsafe_allow_html=True)

    # Giải thích: đóng góp của từng từ (chỉ Naive Bayes có log-prob)
    clf, vec, clean = pipe.named_steps["clf"], pipe.named_steps["vec"], pipe.named_steps["clean"]
    st.subheader("Mô hình nhìn thấy gì?")
    if hasattr(clf, "feature_log_prob_"):
        diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        vocab = vec.vocabulary_
        chunks = []
        for w in text.split():
            toks = clean.transform([w])[0].split()
            s = sum(diff[vocab[t]] for t in toks if t in vocab)
            if abs(s) < 0.3:
                chunks.append(html.escape(w))
            else:
                col = "196,78,82" if s > 0 else "76,114,176"
                chunks.append(f'<mark style="background:rgba({col},{min(abs(s) / 5, 1) * .65 + .1:.2f})" '
                              f'title="{s:+.2f}">{html.escape(w)}</mark>')
        st.markdown('<div class="msg">' + " ".join(chunks) + "</div>", unsafe_allow_html=True)
        st.markdown('<div class="legend"><span><i style="background:rgba(196,78,82,.6)"></i>Đỏ: từ đẩy về spam</span>'
                    '<span><i style="background:rgba(76,114,176,.6)"></i>xanh: từ đẩy về ham</span>'
                    '<span class="hint">càng đậm càng mạnh (rê chuột để xem điểm log-odds).</span></div>',
                    unsafe_allow_html=True)
        x = vec.transform(clean.transform([text])).toarray()[0]
        names = np.array(vec.get_feature_names_out())
        contrib = x * diff
        top = np.argsort(-np.abs(contrib))[:10]
        top = top[contrib[top] != 0]
        if len(top):
            bar = go.Figure(go.Bar(x=contrib[top][::-1], y=names[top][::-1], orientation="h",
                                   marker_color=[SPAM if v > 0 else HAM for v in contrib[top][::-1]]))
            bar.update_layout(title="Các đặc trưng ảnh hưởng nhiều nhất (← ham | spam →)")
            st.plotly_chart(fig_style(bar, 330), width="stretch")
    else:
        st.info("Mô hình này (K-NN / Decision Tree) không có trọng số từ để tô màu. "
                "Chọn mô hình Naive Bayes (vd E5) để xem giải thích.")
    with st.expander("Văn bản sau tiền xử lý (đầu vào thật của vectorizer)"):
        st.code(clean.transform([text])[0])

    known = [w for w in clean.transform([text])[0].split() if w in vec.vocabulary_]
    if len(known) < 0.3 * max(len(text.split()), 1):
        st.warning("Hầu hết từ trong tin này không có trong từ vựng của mô hình (chỉ học tiếng Anh) "
                   "→ kết quả không đáng tin, nhất là với tiếng Việt.")

    batch_test(mid, thr)


# ================================================================ 1. DATA SOURCE
elif page == STEPS[1]:
    page_head("1 · Data Source - Nguồn dữ liệu")
    # Dòng chảy dữ liệu: tin gốc -> bỏ trùng -> chia train/test
    st.markdown(
        '<div class="flow">'
        f'<div class="fs"><small>Tin gốc</small><b>{raw_n:,}</b></div>'
        f'<div class="fa"><span>-{raw_n - len(df)} tin trùng</span></div>'
        f'<div class="fs"><small>Sau khi bỏ trùng</small><b>{len(df):,}</b></div>'
        '<div class="fa"></div>'
        f'<div class="fs split"><div><small>Train</small><b>{len(X_train):,}</b></div>'
        f'<div><small>Test</small><b>{len(X_test):,}</b></div></div>'
        '</div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        cards([
            ("Nguồn", f"SMS Spam Collection (UCI) - file <code>{html.escape(cfg.path.name)}</code>, "
                      "dạng <code>label&lt;TAB&gt;text</code>."),
            ("Nhãn", "<code>ham=0</code>, <code>spam=1</code> (spam là lớp dương để Precision/Recall có ý nghĩa)."),
            ("Bỏ tin trùng trước khi chia",
             '→ tránh cùng một tin nằm cả ở train và test (rò rỉ).'),
            ("Chia dữ liệu", f"<code>test_size={cfg.test_size}</code>, <code>stratify</code> theo nhãn, "
                             f"<code>random_state={cfg.random_state}</code>."),
            ("Giả định/giới hạn", "tin tiếng Anh (Anh, ~2012), dữ liệu nhỏ; spam hiện nay có thể khác.", "wide"),
        ])
    with right:
        split = pd.DataFrame({"Tập": ["Train", "Train", "Test", "Test"],
                              "Nhãn": ["ham", "spam", "ham", "spam"],
                              "Số tin": [(y_train == 0).sum(), (y_train == 1).sum(),
                                         (y_test == 0).sum(), (y_test == 1).sum()]})
        f = px.bar(split, x="Tập", y="Số tin", color="Nhãn", barmode="stack",
                   color_discrete_map={"ham": HAM, "spam": SPAM}, title="Chia train/test (giữ nguyên tỉ lệ lớp)")
        st.plotly_chart(fig_style(f), width="stretch")
    st.subheader("Xem thử dữ liệu")
    show = df.assign(nhãn=df["label"].map({0: "ham", 1: "spam"}))[["nhãn", "text"]]
    with st.container(border=True, key="ds_tool"):  # khung thao tác: lọc + bảng
        which = st.radio("Lọc", ["tất cả", "ham", "spam"], horizontal=True)
        if which != "tất cả":
            show = show[show["nhãn"] == which]
        st.dataframe(show.sample(min(200, len(show)), random_state=1), width="stretch", hide_index=True)


# ================================================================ 2. EDA
elif page == STEPS[2]:
    page_head("2 · EDA - Khám phá dữ liệu")
    d = df.assign(nhãn=df["label"].map({0: "ham", 1: "spam"}), chars=df["text"].str.len(),
                  words=df["text"].str.split().str.len())
    cmap = {"ham": HAM, "spam": SPAM}
    a, b = st.columns(2)
    with a:
        cnt = d["nhãn"].value_counts().rename_axis("nhãn").reset_index(name="số tin")
        f = px.pie(cnt, names="nhãn", values="số tin", hole=.5, color="nhãn", color_discrete_map=cmap,
                   title="Mất cân bằng lớp")
        st.plotly_chart(fig_style(f), width="stretch")
        note(None, f"Spam chỉ {d['label'].mean():.1%} → mô hình luôn đoán “ham” đã đạt "
                         f"<b>{1 - d['label'].mean():.1%} accuracy</b> nhưng không bắt được spam nào. "
                         "Vì vậy phải dùng Precision / Recall / F1 của lớp spam.")
    with b:
        f = px.histogram(d, x="chars", color="nhãn", nbins=50, barmode="overlay", opacity=.65,
                         histnorm="probability density", color_discrete_map=cmap,
                         title="Độ dài tin nhắn (ký tự)")
        st.plotly_chart(fig_style(f), width="stretch")
        g = d.groupby("nhãn")[["chars", "words"]].mean().round(1)
        note(None, f"Spam dài hơn: trung bình {g.loc['spam', 'chars']} ký tự vs {g.loc['ham', 'chars']} của ham.")

    st.subheader("Dấu hiệu spam - tỉ lệ tin có đặc điểm này")
    feats = {
        "có chữ số": d["text"].str.contains(r"\d"),
        "có ≥ 5 chữ số liền (số điện thoại/mã)": d["text"].str.contains(r"\d{5,}"),
        "có URL": d["text"].str.contains(r"http|www\.|\.com|\.co\.uk", case=False),
        "có £ hoặc $": d["text"].str.contains(r"[£$€]"),
        "có dấu !": d["text"].str.contains("!"),
        "có từ VIẾT HOA toàn bộ (≥3 chữ)": d["text"].str.contains(r"\b[A-Z]{3,}\b"),
    }
    rate = pd.DataFrame([{"đặc điểm": k, "nhãn": lab, "tỉ lệ": v[d["nhãn"] == lab].mean()}
                         for k, v in feats.items() for lab in ("ham", "spam")])
    f = px.bar(rate, y="đặc điểm", x="tỉ lệ", color="nhãn", barmode="group", orientation="h",
               color_discrete_map=cmap)
    f.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_style(f, 380), width="stretch")

    st.subheader("Từ phổ biến theo lớp")
    with st.container(border=True, key="eda_tool"):  # khung thao tác
        n = st.slider("Số từ hiển thị", 5, 30, 15)
    for col, lab in zip(st.columns(2), ("ham", "spam")):
        top = pd.DataFrame(_top_words(d.loc[d["nhãn"] == lab, "text"], n), columns=["từ", "số lần"])
        f = px.bar(top[::-1], x="số lần", y="từ", orientation="h", title=f"Top từ - {lab}",
                   color_discrete_sequence=[cmap[lab]])
        col.plotly_chart(fig_style(f, 120 + 22 * n), width="stretch")
    note("Giả thuyết ban đầu", "spam chứa “free, call, txt, claim, prize…”, số điện thoại, ký hiệu tiền, "
                               "và dài hơn ham.")


# ================================================================ 3. PREPROCESSING
elif page == STEPS[3]:
    page_head("3 · Preprocessing - Tiền xử lý")
    note(None, "Với spam, URL / số / tiền là tín hiệu quan trọng → thay bằng token thay vì xóa. "
                    "Bật/tắt từng bước để xem văn bản thay đổi:")
    with st.container(border=True, key="pre_tool"):  # công cụ thử: ô tick + tin nhắn + Trước/Sau
        c = st.columns(5)
        opts = dict(
            replace_url=c[0].checkbox("URL → urltoken", True),
            replace_money=c[1].checkbox("Tiền → moneytoken", True),
            replace_number=c[2].checkbox("Số → numtoken", True),
            remove_punct=c[3].checkbox("Bỏ dấu câu", True),
            remove_stopwords=c[4].checkbox("Bỏ stopword", False))
        sample = st.text_area("Tin nhắn", EXAMPLES["Spam - trúng thưởng"] + " Visit www.prize-now.com", height=90)
        out = TextCleaner(**opts).transform([sample])[0]
        # Trước -> Sau, cùng kiểu "dòng chảy" với trang Data Source; tô nhẹ các token được thay thế
        out_html = " ".join(f'<mark>{html.escape(w)}</mark>' if w in ("urltoken", "moneytoken", "numtoken")
                            else html.escape(w) for w in out.split())
        st.markdown(
            '<div class="flow xform">'
            f'<div class="fs"><small>Trước</small><div class="tx">{html.escape(sample)}</div></div>'
            '<div class="fa"></div>'
            f'<div class="fs"><small>Sau</small><div class="tx">{out_html}</div></div>'
            '</div>', unsafe_allow_html=True)

    st.subheader("Ablation - bằng chứng tác động lên kết quả")
    abl = [i for i in ("E2", "E5", "E6") if i in M.index]
    t = M.loc[abl, ["description", "cv_f1_train", "f1", "precision", "recall"]].reset_index()
    with st.container(key="tbl_abl"):
        st.table(table_style(t.style.format({c: "{:.4f}" for c in t.columns[2:]}), list(t.columns[2:])))
    long = t.melt(id_vars=["id", "description"], value_vars=["cv_f1_train", "f1"], var_name="chỉ số")
    f = px.bar(long, x="id", y="value", color="chỉ số", barmode="group", title="F1-spam: CV (train) vs test")
    f.update_yaxes(range=[max(0, long["value"].min() - .05), 1])
    st.plotly_chart(fig_style(f), width="stretch")


# ================================================================ 4. FEATURE
elif page == STEPS[4]:
    page_head("4 · Feature - Biểu diễn văn bản")
    rows = []
    for i in ids:
        v = specs[i].vectorizer if i in specs else {}
        vec = models[i].named_steps["vec"]
        rows.append({"id": i, "loại": v.get("kind", "bow").upper(),
                     "n-gram": str(tuple(v.get("ngram_range", (1, 1)))), "min_df": v.get("min_df", 1),
                     "kích thước từ vựng": len(vec.vocabulary_), "F1 test": M.loc[i, "f1"]})
    t = pd.DataFrame(rows)
    # Giải thích khái niệm trước, rồi mới tới bảng + biểu đồ
    with st.container(key="feat_intro"):
        cards([
            ("", "<b>BoW</b> đếm số lần từ xuất hiện; <b>TF-IDF</b> giảm trọng số từ xuất hiện ở mọi tin."),
            ("", "<b>(1,2)-gram</b> giữ được cụm như “call now”, “free entry” nhưng làm từ vựng phình ra → "
                 "<code>min_df=2</code> lọc cụm hiếm."),
            ("", 'Vectorizer nằm trong <code>Pipeline</code> nên chỉ fit trên train, '
                 "không rò rỉ sang test."),
        ], cols=3)
    left, right = st.columns([11, 9])
    with left:
        with st.container(key="tbl_feat"):
            st.table(table_style(t.style.format({"F1 test": "{:.4f}"}), ["F1 test"]))
    with right:
        f = px.scatter(t, x="kích thước từ vựng", y="F1 test", color="loại", text="id",
                       title="Từ vựng lớn hơn có tốt hơn?")
        f.update_traces(textposition="top center", marker_size=12)
        st.plotly_chart(fig_style(f, 380), width="stretch")


# ================================================================ 5. MODEL
elif page == STEPS[5]:
    page_head("5 · Model - Thuật toán & tham số")
    NAMES = {"nb": "Multinomial Naive Bayes", "knn": "K-NN (cosine)", "tree": "Decision Tree"}
    t = pd.DataFrame([{"id": i, "thuật toán": NAMES.get(specs[i].model, specs[i].model),
                       "tham số": str(specs[i].model_params) if specs[i].model_params else "mặc định",
                       "mô tả": specs[i].description, "CV F1 (train)": M.loc[i, "cv_f1_train"],
                       "F1 test": M.loc[i, "f1"]} for i in ids if i in specs])
    # Kết luận của trang đặt lên đầu (cùng vị trí với khung ý tưởng ở trang 3)
    note(None, f"Chọn <b>{best_id}</b> vì F1-spam qua cross-validation trên tập train "
                                  "cao nhất (không nhìn tập test → tránh chọn thiên lệch).")
    # Bảng tĩnh (tự xuống dòng) để hiện đủ cột "tham số" dài, không phải cuộn ngang
    with st.container(key="tbl_model"):
        st.table(table_style(t.style.format({"CV F1 (train)": "{:.4f}", "F1 test": "{:.4f}"}),
                             ["CV F1 (train)", "F1 test"]))
    long = t.melt(id_vars=["id"], value_vars=["CV F1 (train)", "F1 test"], var_name="chỉ số")
    f = px.bar(long, x="id", y="value", color="chỉ số", barmode="group")
    f.update_yaxes(range=[max(0, long["value"].min() - .05), 1])
    st.plotly_chart(fig_style(f), width="stretch")
    seeds = get_csv("stability_f1.csv")
    if seeds is not None:
        f = px.box(seeds, x="id", y="f1", points="all", title="Độ ổn định: F1 qua nhiều lần chia ngẫu nhiên")
        st.plotly_chart(fig_style(f, 380), width="stretch")


# ================================================================ 6. EVALUATION
elif page == STEPS[6]:
    page_head("6 · Evaluation - Đánh giá")
    mid = pick_model(key="eval_model")
    r = M.loc[mid]
    for col, (k, lab) in zip(st.columns(5), [("accuracy", "Accuracy"), ("precision", "Precision"),
                                             ("recall", "Recall"), ("f1", "F1-spam"), ("roc_auc", "ROC-AUC")]):
        col.metric(lab, f"{r[k]:.3f}")
    with st.container(key="eval_intro"):
        cards([("", "<b>Precision</b> cao = ít ham bị chặn nhầm (FP)."),
               ("", "<b>Recall</b> cao = ít spam lọt qua (FN)."),
               ("", 'FP thường nghiêm trọng hơn (mất tin quan trọng).')], cols=3)

    # Công cụ: kéo ngưỡng → thấy ngay Precision/Recall/F1 ngay bên dưới, rồi tới biểu đồ
    with st.container(border=True, key="eval_tool"):
        thr = st.slider("Ngưỡng quyết định", 0.05, 0.95, 0.5, 0.05)
        score = test_scores(mid)
        pred = (score >= thr).astype(int)
        strip([("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}"),
               ("Recall", f"{recall_score(y_test, pred):.3f}"), ("F1", f"{f1_score(y_test, pred):.3f}")],
              lead="Ở ngưỡng này:")
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    a, b = st.columns([2, 3])
    with a:
        f = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Dự đoán ham", "Dự đoán spam"],
                                 y=["Thực ham", "Thực spam"],
                                 text=[[f"TN {tn}", f"FP {fp}"], [f"FN {fn}", f"TP {tp}"]],
                                 texttemplate="%{text}", colorscale="Blues", showscale=False,
                                 textfont={"size": 18}))
        f.update_yaxes(autorange="reversed")
        f.update_layout(title=f"Confusion matrix (ngưỡng {thr:.2f})")
        st.plotly_chart(fig_style(f, 340), width="stretch")
    with b:
        rows = [{"ngưỡng": t, "Precision": precision_score(y_test, score >= t, zero_division=0),
                 "Recall": recall_score(y_test, score >= t), "F1": f1_score(y_test, score >= t)}
                for t in np.arange(0.05, 0.96, 0.05)]
        f = px.line(pd.DataFrame(rows).melt("ngưỡng", var_name="chỉ số"), x="ngưỡng", y="value", color="chỉ số",
                    title="Đánh đổi Precision ↔ Recall theo ngưỡng")
        f.add_vline(x=thr, line_dash="dash")
        st.plotly_chart(fig_style(f, 340), width="stretch")

    st.subheader("So sánh tất cả mô hình")
    fr, fpr_fig = go.Figure(), go.Figure()
    for i in ids:
        s = test_scores(i)
        fpr, tpr, _ = roc_curve(y_test, s)
        fr.add_trace(go.Scatter(x=fpr, y=tpr, name=i, mode="lines"))
        pr, rc, _ = precision_recall_curve(y_test, s)
        fpr_fig.add_trace(go.Scatter(x=rc, y=pr, name=i, mode="lines"))
    fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="đoán mò", line=dict(dash="dot", color="gray")))
    fr.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR")
    fpr_fig.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_style(fr, 380), width="stretch")
    c2.plotly_chart(fig_style(fpr_fig, 380), width="stretch")
    cols = ["id", "description", "accuracy", "precision", "recall", "f1", "roc_auc"]
    # Bảng tĩnh như trang 5: rộng hết khung, cột mô tả xuống dòng, số căn phải
    with st.container(key="tbl_eval"):
        st.table(table_style(metrics[cols].style.format({c: "{:.4f}" for c in cols[2:]}), cols[2:]))


# ================================================================ 7. ERROR ANALYSIS
elif page == STEPS[7]:
    page_head("7 · Error Analysis -Phân tích lỗi")
    mid = pick_model(key="err_model")
    pipe = models[mid]
    err = error_examples(X_test, y_test, pipe.predict(X_test))
    # Mỗi loại lỗi: con số + nhận xét của chính nó nằm chung một ô
    boxes = "".join(
        f'<div class="errbox {cls}"><div class="lbl">{lab}</div><div class="num">{n}</div>'
        f'<div class="why">{why}</div></div>' for cls, lab, n, why in [
            ("fp", "FP - ham bị chặn nhầm", int((err.error == "FP").sum()),
             "FP thường là tin ham có “free”, “call”, số điện thoại…"),
            ("fn", "FN - spam lọt qua", int((err.error == "FN").sum()),
             "FN thường là spam viết khéo, ngắn, không có từ khóa điển hình.")])
    st.markdown(f'<div class="errgrid">{boxes}</div>', unsafe_allow_html=True)
    # Khung xem lỗi: chọn loại → bảng tin hiện đủ cả câu (không bị cắt như bảng cuộn)
    with st.container(border=True, key="err_tool"):
        kind = st.radio("Xem lỗi", ["FN", "FP"], horizontal=True,
                        format_func=lambda k: "FN - spam bị bỏ sót" if k == "FN" else "FP - ham bị báo nhầm")
        st.table(err[err.error == kind][["text", "length"]].style.hide(axis="index"))
    words = top_spam_words(pipe, 15)
    if words is not None:
        st.subheader("Mô hình học được gì? (log P(w|spam) − log P(w|ham))")
        f1_ = px.bar(words[::-1], x="spam_score", y="spam_indicative", orientation="h",
                     title="Từ báo hiệu SPAM", color_discrete_sequence=[SPAM])
        f2_ = px.bar(words[::-1], x="ham_score", y="ham_indicative", orientation="h",
                     title="Từ báo hiệu HAM", color_discrete_sequence=[HAM])
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_style(f1_, 460), width="stretch")
        c2.plotly_chart(fig_style(f2_, 460), width="stretch")
        note(None, "Naive Bayes giả định các từ độc lập nên cộng dồn bằng chứng từng từ.")
    else:
        st.info("Chọn mô hình Naive Bayes để xem từ đặc trưng.")


# ================================================================ 8. CONCLUSION
else:
    page_head("8 · Conclusion - Kết luận")
    r = M.loc[best_id]
    base = M.loc[ids[0]]
    st.markdown(
        '<div class="result"><div class="rhead">'
        f'<span class="rid">{best_id}</span><span class="rdesc">- {html.escape(str(r["description"]))}</span></div>'
        '<div class="rstats">'
        + "".join(f'<span><small>{k}</small><b>{v:.3f}</b></span>' for k, v in
                  [("F1-spam test", r["f1"]), ("Precision", r["precision"]), ("Recall", r["recall"]),
                   ("ROC-AUC", r["roc_auc"])])
        + '</div></div>', unsafe_allow_html=True)
    with st.container(key="concl_cmp"):  # 3 ô so sánh, cùng kiểu ô số liệu các trang trước
        st.markdown(f'<div class="cap-lbl">So với baseline {ids[0]}:</div>', unsafe_allow_html=True)
        cards([cmp_card("F1", f"{base['f1']:.3f}", f"{r['f1']:.3f}", f"({r['f1'] - base['f1']:+.3f})"),
               cmp_card("FP", int(base["FP"]), int(r["FP"])),
               cmp_card("FN", int(base["FN"]), int(r["FN"]))], cols=3)

    st.subheader("Kiểm tra độ tin cậy (rò rỉ dữ liệu / học thuộc)")
    chk = get_csv("sanity_checks.csv")
    if chk is not None:
        # Bảng tĩnh hiện đủ chữ; trạng thái tô màu chữ thay cho biểu tượng
        tone = {"OK": "color:#1E6B3A;font-weight:600", "LƯU Ý": "color:#8A5A00;font-weight:600",
                "LỖI": "color:#B3261E;font-weight:600"}
        with st.container(key="tbl_chk"):
            st.table(chk.style.map(lambda s: tone.get(s, ""), subset=["trạng thái"]).hide(axis="index"))
    else:
        st.info("Chạy `python -m spam_filter check` để tạo bảng này.")
    lc = OUT_FIGURES / "viz_9_learning_curve.png"
    if lc.exists():
        st.image(str(lc), caption="Learning curve", width=620)

    st.subheader("Hạn chế & hướng phát triển")
    with st.container(key="concl_lim"):  # khung xám có vạch ngăn, như đầu trang 4 và 6
        cards([
            ("Hạn chế", "dữ liệu nhỏ (5160 tin), tiếng Anh, spam thay đổi theo thời gian; "
                        "Naive Bayes giả định từ độc lập."),
            ("Ứng dụng thực tế", 'ưu tiên Precision (không chặn nhầm tin quan trọng) → tăng ngưỡng quyết định '
                                 "(thử ở bước 6), tin nghi ngờ chuyển vào thư mục “Spam” thay vì xóa."),
            ("Cải thiện", "class weight, thêm đặc trưng thủ công (độ dài, số chữ số), char n-gram, "
                          "sentence embeddings, cập nhật mô hình định kỳ."),
        ], cols=3)
    with st.expander("Toàn bộ biểu đồ đã xuất (outputs/figures)"):
        imgs = sorted(OUT_FIGURES.glob("*.png"))
        for i in range(0, len(imgs), 3):
            for col, img in zip(st.columns(3), imgs[i:i + 3]):
                col.image(str(img), caption=img.stem)


# ---------------------------------------------------------------- chuyển bước (trước / sau)
st.divider()
_i = STEPS.index(page)
_prev, _, _next = st.columns([2, 1, 2])
if _i > 0:
    _prev.button(f"← {STEPS[_i - 1]}", on_click=_go, args=(STEPS[_i - 1],), width="stretch")
if _i < len(STEPS) - 1:
    _next.button(f"{STEPS[_i + 1]} →", on_click=_go, args=(STEPS[_i + 1],), width="stretch", type="primary")

