"""Giao diện web theo đúng luồng của đề (Hình 1):
Data Source -> EDA -> Preprocessing -> Feature -> Model -> Evaluation -> Error Analysis -> Conclusion

Chạy:  streamlit run app.py     (cần có kết quả: python -m spam_filter run)
"""
import html
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
st.set_page_config(page_title="Lọc tin nhắn rác", page_icon="📩", layout="wide")
st.markdown("""
<style>
.flow{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 14px}
.flow span{padding:5px 11px;border-radius:16px;font-size:13px;background:rgba(128,128,128,.15)}
.flow span.on{background:#C44E52;color:#fff;font-weight:600}
.flow span.done{background:rgba(76,114,176,.35)}
.verdict{padding:18px 22px;border-radius:12px;font-size:26px;font-weight:700;color:#fff}
.msg{line-height:2;font-size:16px;padding:12px 14px;border-radius:10px;background:rgba(128,128,128,.1)}
.msg mark{padding:2px 4px;border-radius:4px;color:inherit}
</style>""", unsafe_allow_html=True)

STEPS = ["🏠 Demo dự đoán", "1. Data Source", "2. EDA", "3. Preprocessing", "4. Feature",
         "5. Model", "6. Evaluation", "7. Error Analysis", "8. Conclusion"]
EXAMPLES = {
    "Spam – trúng thưởng": "Free entry! Call 09061701461 now to claim your £500 prize. Text STOP to opt out.",
    "Spam – khuyến mãi": "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt WIN to 80086",
    "Ham – đời thường": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "Ham – ngắn": "Ok lar... Joking wif u oni...",
    # Tin tự viết, NGOÀI dataset – để thử khả năng tổng quát hóa
    "[Ngoài dataset] Spam – giả mạo ngân hàng": "Your bank account has been suspended. Verify now at http://secure-bank-login.xyz to avoid closure.",
    "[Ngoài dataset] Spam – phí giao hàng": "We tried to deliver your package but failed. Pay the 1.99 redelivery fee at bit.ly/3xYz to reschedule.",
    "[Ngoài dataset] Spam – iPhone miễn phí": "Congratulations! You have been selected to receive a free iPhone 15. Click here to claim your gift today.",
    "[Ngoài dataset] Spam – việc làm online": "Earn $500 per day working from home! No experience needed. Message us on WhatsApp now.",
    "[Ngoài dataset] Spam – cảnh báo tài khoản": "URGENT: unusual activity detected on your account. Confirm your identity immediately: tinyurl.com/acc-verify",
    "[Ngoài dataset] Ham – nhắc lịch khám": "Your appointment with Dr. Smith is confirmed for Tuesday at 3:30 PM. Reply C to confirm or call the clinic to reschedule.",
    "[Ngoài dataset] Ham – giao hàng thật": "Your Royal Mail parcel will be delivered tomorrow between 9am and 1pm. Track it on the Royal Mail app.",
    "[Ngoài dataset] Ham – mã xác thực": "Your verification code is 482913. Do not share it with anyone.",
    "[Ngoài dataset] Ham – họp công ty": "Hi Anna, the meeting has moved to 2pm in Room 4. Please bring the printed report. Thanks, Mark",
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
                        format_func=lambda i: f"{i} – {M.loc[i, 'description']}" + (" ⭐" if i == best_id else ""))


def fig_style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def batch_test(mid, thr):
    """Dán nhiều tin (mỗi dòng một tin, tùy chọn tiền tố spam| hoặc ham|) -> bảng dự đoán."""
    st.divider()
    st.subheader("🧪 Test hàng loạt")
    st.caption("Dán nhiều tin nhắn, **mỗi dòng một tin**; thêm nhãn mong đợi ở đầu dòng bằng `spam|` hoặc `ham|` "
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
st.sidebar.title("📩 Lọc tin nhắn rác")
st.sidebar.caption("Đề tài 3 · Nhập môn Học máy\nSMS Spam Collection (UCI)")
page = st.sidebar.radio("Các bước", STEPS, label_visibility="collapsed")
idx = STEPS.index(page)
st.sidebar.markdown("---")
st.sidebar.metric("Mô hình tốt nhất (CV)", best_id)
st.sidebar.metric("F1-spam trên test", f"{M.loc[best_id, 'f1']:.3f}")

flow = "".join(f'<span class="{"on" if i == idx else "done" if i < idx else ""}">{s}</span>'
               for i, s in enumerate(STEPS[1:], 1))
st.markdown(f'<div class="flow">{flow}</div>', unsafe_allow_html=True)


# ================================================================ DEMO
if page == STEPS[0]:
    st.title("🏠 Demo: tin nhắn này là spam hay ham?")
    c_in, c_out = st.columns([3, 2])
    with c_in:
        ex = st.selectbox("Ví dụ mẫu", ["(tự nhập)"] + list(EXAMPLES))
        text = st.text_area("Nhập tin nhắn (tiếng Anh)", height=130, key=f"t_{ex}",
                            value="" if ex == "(tự nhập)" else EXAMPLES[ex])
        a, b = st.columns(2)
        with a:
            mid = pick_model(key="demo_model")
        with b:
            thr = st.slider("Ngưỡng quyết định (spam nếu P ≥ ngưỡng)", 0.05, 0.95, 0.5, 0.05,
                            help="Hạ ngưỡng → bắt nhiều spam hơn nhưng dễ chặn nhầm ham.")
    if not text.strip():
        st.info("Nhập một tin nhắn để xem dự đoán.")
        batch_test(mid, thr)
        st.stop()

    pipe = models[mid]
    p = float(pipe.predict_proba([text])[0, 1])
    is_spam = p >= thr
    with c_out:
        st.markdown(f'<div class="verdict" style="background:{SPAM if is_spam else HAM}">'
                    f'{"🚨 SPAM" if is_spam else "✅ HAM"}</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Xác suất spam"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": SPAM if is_spam else HAM},
                   "threshold": {"line": {"color": "black", "width": 4}, "value": thr * 100}}))
        st.plotly_chart(fig_style(gauge, 250), width="stretch")

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
        st.caption("🔴 từ đẩy về **spam** · 🔵 từ đẩy về **ham** · càng đậm càng mạnh (rê chuột để xem điểm log-odds).")
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
    st.title("1 · Data Source – nguồn dữ liệu")
    c = st.columns(4)
    c[0].metric("Tin gốc", f"{raw_n:,}")
    c[1].metric("Sau khi bỏ trùng", f"{len(df):,}", f"-{raw_n - len(df)} tin trùng")
    c[2].metric("Train", f"{len(X_train):,}")
    c[3].metric("Test", f"{len(X_test):,}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"""
- **Nguồn:** SMS Spam Collection (UCI) – file `{cfg.path.name}`, dạng `label<TAB>text`.
- **Nhãn:** `ham=0`, `spam=1` (spam là lớp dương để Precision/Recall có ý nghĩa).
- **Bỏ tin trùng *trước khi chia*** → tránh cùng một tin nằm cả ở train và test (rò rỉ).
- **Chia dữ liệu:** `test_size={cfg.test_size}`, `stratify` theo nhãn, `random_state={cfg.random_state}`.
- **Giả định/giới hạn:** tin tiếng Anh (Anh, ~2012), dữ liệu nhỏ; spam hiện nay có thể khác.
""")
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
    which = st.radio("Lọc", ["tất cả", "ham", "spam"], horizontal=True)
    if which != "tất cả":
        show = show[show["nhãn"] == which]
    st.dataframe(show.sample(min(200, len(show)), random_state=1), width="stretch", hide_index=True)


# ================================================================ 2. EDA
elif page == STEPS[2]:
    st.title("2 · EDA – khám phá dữ liệu")
    d = df.assign(nhãn=df["label"].map({0: "ham", 1: "spam"}), chars=df["text"].str.len(),
                  words=df["text"].str.split().str.len())
    cmap = {"ham": HAM, "spam": SPAM}
    a, b = st.columns(2)
    with a:
        cnt = d["nhãn"].value_counts().rename_axis("nhãn").reset_index(name="số tin")
        f = px.pie(cnt, names="nhãn", values="số tin", hole=.5, color="nhãn", color_discrete_map=cmap,
                   title="Mất cân bằng lớp")
        st.plotly_chart(fig_style(f), width="stretch")
        st.caption(f"Spam chỉ {d['label'].mean():.1%} → mô hình luôn đoán “ham” đã đạt "
                   f"**{1 - d['label'].mean():.1%} accuracy** nhưng không bắt được spam nào. "
                   "Vì vậy phải dùng Precision / Recall / F1 của lớp spam.")
    with b:
        f = px.histogram(d, x="chars", color="nhãn", nbins=50, barmode="overlay", opacity=.65,
                         histnorm="probability density", color_discrete_map=cmap,
                         title="Độ dài tin nhắn (ký tự)")
        st.plotly_chart(fig_style(f), width="stretch")
        g = d.groupby("nhãn")[["chars", "words"]].mean().round(1)
        st.caption(f"Spam dài hơn: trung bình {g.loc['spam', 'chars']} ký tự vs {g.loc['ham', 'chars']} của ham.")

    st.subheader("Dấu hiệu spam – tỉ lệ tin có đặc điểm này")
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
    n = st.slider("Số từ hiển thị", 5, 30, 15)
    for col, lab in zip(st.columns(2), ("ham", "spam")):
        top = pd.DataFrame(_top_words(d.loc[d["nhãn"] == lab, "text"], n), columns=["từ", "số lần"])
        f = px.bar(top[::-1], x="số lần", y="từ", orientation="h", title=f"Top từ – {lab}",
                   color_discrete_sequence=[cmap[lab]])
        col.plotly_chart(fig_style(f, 120 + 22 * n), width="stretch")
    st.info("Giả thuyết ban đầu: spam chứa “free, call, txt, claim, prize…”, số điện thoại, ký hiệu tiền, "
            "và dài hơn ham.")


# ================================================================ 3. PREPROCESSING
elif page == STEPS[3]:
    st.title("3 · Preprocessing – tiền xử lý")
    st.markdown("Với spam, **URL / số / tiền là tín hiệu quan trọng** → thay bằng token thay vì xóa. "
                "Bật/tắt từng bước để xem văn bản thay đổi:")
    c = st.columns(5)
    opts = dict(
        replace_url=c[0].checkbox("URL → urltoken", True),
        replace_money=c[1].checkbox("Tiền → moneytoken", True),
        replace_number=c[2].checkbox("Số → numtoken", True),
        remove_punct=c[3].checkbox("Bỏ dấu câu", True),
        remove_stopwords=c[4].checkbox("Bỏ stopword", False))
    sample = st.text_area("Tin nhắn", EXAMPLES["Spam – trúng thưởng"] + " Visit www.prize-now.com", height=90)
    out = TextCleaner(**opts).transform([sample])[0]
    l, r = st.columns(2)
    l.markdown("**Trước**")
    l.code(sample, language=None)
    r.markdown("**Sau**")
    r.code(out, language=None)

    st.subheader("Ablation – bằng chứng tác động lên kết quả")
    abl = [i for i in ("E2", "E5", "E6") if i in M.index]
    t = M.loc[abl, ["description", "cv_f1_train", "f1", "precision", "recall"]].reset_index()
    st.dataframe(t.style.format({c: "{:.4f}" for c in t.columns[2:]}).highlight_max(
        subset=list(t.columns[2:]), color="#c8e6c9"), width="stretch", hide_index=True)
    long = t.melt(id_vars=["id", "description"], value_vars=["cv_f1_train", "f1"], var_name="chỉ số")
    f = px.bar(long, x="id", y="value", color="chỉ số", barmode="group", title="F1-spam: CV (train) vs test")
    f.update_yaxes(range=[max(0, long["value"].min() - .05), 1])
    st.plotly_chart(fig_style(f), width="stretch")


# ================================================================ 4. FEATURE
elif page == STEPS[4]:
    st.title("4 · Feature – biểu diễn văn bản")
    rows = []
    for i in ids:
        v = specs[i].vectorizer if i in specs else {}
        vec = models[i].named_steps["vec"]
        rows.append({"id": i, "loại": v.get("kind", "bow").upper(),
                     "n-gram": str(tuple(v.get("ngram_range", (1, 1)))), "min_df": v.get("min_df", 1),
                     "kích thước từ vựng": len(vec.vocabulary_), "F1 test": M.loc[i, "f1"]})
    t = pd.DataFrame(rows)
    st.dataframe(t.style.format({"F1 test": "{:.4f}"}), width="stretch", hide_index=True)
    f = px.scatter(t, x="kích thước từ vựng", y="F1 test", color="loại", text="id",
                   title="Từ vựng lớn hơn có tốt hơn?")
    f.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig_style(f, 380), width="stretch")
    st.markdown("""
- **BoW** đếm số lần từ xuất hiện; **TF-IDF** giảm trọng số từ xuất hiện ở mọi tin.
- **(1,2)-gram** giữ được cụm như “call now”, “free entry” nhưng làm từ vựng phình ra → `min_df=2` lọc cụm hiếm.
- Vectorizer nằm trong `Pipeline` nên **chỉ fit trên train**, không rò rỉ sang test.
""")


# ================================================================ 5. MODEL
elif page == STEPS[5]:
    st.title("5 · Model – thuật toán & tham số")
    NAMES = {"nb": "Multinomial Naive Bayes", "knn": "K-NN (cosine)", "tree": "Decision Tree"}
    t = pd.DataFrame([{"id": i, "thuật toán": NAMES.get(specs[i].model, specs[i].model),
                       "tham số": str(specs[i].model_params) if specs[i].model_params else "mặc định",
                       "mô tả": specs[i].description, "CV F1 (train)": M.loc[i, "cv_f1_train"],
                       "F1 test": M.loc[i, "f1"]} for i in ids if i in specs])
    st.dataframe(t.style.format({"CV F1 (train)": "{:.4f}", "F1 test": "{:.4f}"}),
                 width="stretch", hide_index=True)
    st.success(f"Chọn **{best_id}** vì F1-spam qua cross-validation trên **tập train** cao nhất "
               "(không nhìn tập test → tránh chọn thiên lệch).")
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
    st.title("6 · Evaluation – đánh giá")
    mid = pick_model(key="eval_model")
    r = M.loc[mid]
    for col, (k, lab) in zip(st.columns(5), [("accuracy", "Accuracy"), ("precision", "Precision"),
                                             ("recall", "Recall"), ("f1", "F1-spam"), ("roc_auc", "ROC-AUC")]):
        col.metric(lab, f"{r[k]:.3f}")
    st.caption("**Precision** cao = ít ham bị chặn nhầm (FP). **Recall** cao = ít spam lọt qua (FN). "
               "FP thường nghiêm trọng hơn (mất tin quan trọng).")

    thr = st.slider("Ngưỡng quyết định", 0.05, 0.95, 0.5, 0.05)
    score = test_scores(mid)
    pred = (score >= thr).astype(int)
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
    st.write(f"Ở ngưỡng này: Precision **{precision_score(y_test, pred, zero_division=0):.3f}**, "
             f"Recall **{recall_score(y_test, pred):.3f}**, F1 **{f1_score(y_test, pred):.3f}**.")

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
    fpr_fig.update_layout(title="Precision–Recall", xaxis_title="Recall", yaxis_title="Precision")
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_style(fr, 380), width="stretch")
    c2.plotly_chart(fig_style(fpr_fig, 380), width="stretch")
    cols = ["id", "description", "accuracy", "precision", "recall", "f1", "roc_auc"]
    st.dataframe(metrics[cols].style.format({c: "{:.4f}" for c in cols[2:]}).highlight_max(
        subset=cols[2:], color="#c8e6c9"), width="stretch", hide_index=True)


# ================================================================ 7. ERROR ANALYSIS
elif page == STEPS[7]:
    st.title("7 · Error Analysis – phân tích lỗi")
    mid = pick_model(key="err_model")
    pipe = models[mid]
    err = error_examples(X_test, y_test, pipe.predict(X_test))
    c = st.columns(2)
    c[0].metric("FP – ham bị chặn nhầm", int((err.error == "FP").sum()))
    c[1].metric("FN – spam lọt qua", int((err.error == "FN").sum()))
    kind = st.radio("Xem lỗi", ["FN", "FP"], horizontal=True,
                    format_func=lambda k: "FN – spam bị bỏ sót" if k == "FN" else "FP – ham bị báo nhầm")
    st.dataframe(err[err.error == kind][["text", "length"]], width="stretch", hide_index=True)
    st.caption("FN thường là spam viết khéo, ngắn, không có từ khóa điển hình. "
               "FP thường là tin ham có “free”, “call”, số điện thoại…")
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
        st.caption("Naive Bayes giả định các từ độc lập nên cộng dồn bằng chứng từng từ.")
    else:
        st.info("Chọn mô hình Naive Bayes để xem từ đặc trưng.")


# ================================================================ 8. CONCLUSION
else:
    st.title("8 · Conclusion – kết luận")
    r = M.loc[best_id]
    st.success(f"**{best_id}** – {r['description']}  \n"
               f"F1-spam test **{r['f1']:.3f}** · Precision **{r['precision']:.3f}** · "
               f"Recall **{r['recall']:.3f}** · ROC-AUC **{r['roc_auc']:.3f}**")
    base = M.loc[ids[0]]
    st.markdown(f"So với baseline **{ids[0]}**: F1 {base['f1']:.3f} → {r['f1']:.3f} "
                f"({r['f1'] - base['f1']:+.3f}), FP {int(base['FP'])} → {int(r['FP'])}, "
                f"FN {int(base['FN'])} → {int(r['FN'])}.")

    st.subheader("Kiểm tra độ tin cậy (rò rỉ dữ liệu / học thuộc)")
    chk = get_csv("sanity_checks.csv")
    if chk is not None:
        icon = {"OK": "✅", "LƯU Ý": "⚠️", "LỖI": "❌"}
        chk = chk.assign(**{"trạng thái": chk["trạng thái"].map(lambda s: f"{icon.get(s, '')} {s}")})
        st.dataframe(chk, width="stretch", hide_index=True)
    else:
        st.info("Chạy `python -m spam_filter check` để tạo bảng này.")
    lc = OUT_FIGURES / "viz_9_learning_curve.png"
    if lc.exists():
        st.image(str(lc), caption="Learning curve", width=620)

    st.subheader("Hạn chế & hướng phát triển")
    st.markdown("""
- **Hạn chế:** dữ liệu nhỏ (5160 tin), tiếng Anh, spam thay đổi theo thời gian; Naive Bayes giả định từ độc lập.
- **Ứng dụng thực tế:** ưu tiên **Precision** (không chặn nhầm tin quan trọng) → tăng ngưỡng quyết định
  (thử ở bước 6), tin nghi ngờ chuyển vào thư mục “Spam” thay vì xóa.
- **Cải thiện:** class weight, thêm đặc trưng thủ công (độ dài, số chữ số), char n-gram, sentence embeddings,
  cập nhật mô hình định kỳ.
""")
    with st.expander("Toàn bộ biểu đồ đã xuất (outputs/figures)"):
        imgs = sorted(OUT_FIGURES.glob("*.png"))
        for i in range(0, len(imgs), 3):
            for col, img in zip(st.columns(3), imgs[i:i + 3]):
                col.image(str(img), caption=img.stem)
