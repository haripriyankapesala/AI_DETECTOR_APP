"""
AI-Generated vs Human Text Detector — Streamlit App
VS Code compatible version

HOW TO RUN:
-----------
1. Put this file, major.pkl, and hybrid_model.pkl in the SAME folder
2. Open terminal in that folder
3. Run:  streamlit run app.py
4. Open the URL shown (http://localhost:8501) in Chrome or Firefox
   Do NOT use VS Code's built-in Simple Browser — use a real browser tab.

INSTALL REQUIREMENTS:
pip install streamlit pandas numpy scikit-learn scipy nltk sentence-transformers lightgbm
"""
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
import re
import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

for _r in ["stopwords", "punkt", "punkt_tab", "averaged_perceptron_tagger"]:
    nltk.download(_r, quiet=True)

from scipy.sparse import csr_matrix, hstack as sp_hstack


# ==============================================================================
# PAGE CONFIG — must be the FIRST streamlit call
# ==============================================================================
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide",
)


# ==============================================================================
# LINGUISTIC FEATURE EXTRACTOR  (exact copy from training code)
# ==============================================================================
class LinguisticFeatureExtractor:
    FEATURE_NAMES = [
        "word_count", "avg_word_length", "sentence_count", "avg_sentence_length",
        "type_token_ratio", "punct_density", "comma_density", "exclamation_density",
        "question_density", "uppercase_ratio", "digit_density", "stopword_ratio",
        "long_word_ratio", "short_word_ratio", "avg_paragraph_length", "paragraph_count",
    ]

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.feature_names_ = self.FEATURE_NAMES

    def extract(self, text: str) -> dict:
        text = str(text)
        words       = word_tokenize(text.lower())
        words_alpha = [w for w in words if w.isalpha()]
        sentences   = sent_tokenize(text)
        paragraphs  = [p.strip() for p in text.split("\n\n") if p.strip()]

        n_words     = max(len(words_alpha), 1)
        n_chars     = max(len(text), 1)
        n_sentences = max(len(sentences), 1)
        alpha_count = sum(1 for c in text if c.isalpha())

        unique_words   = set(words_alpha)
        stopword_count = sum(1 for w in words_alpha if w in self.stop_words)
        long_words     = sum(1 for w in words_alpha if len(w) > 6)
        short_words    = sum(1 for w in words_alpha if len(w) <= 3)
        punct_count    = sum(1 for c in text if c in ".,;:!?'\"-")
        para_wc        = [len(p.split()) for p in paragraphs] if paragraphs else [n_words]

        return {
            "word_count":           n_words,
            "avg_word_length":      sum(len(w) for w in words_alpha) / n_words,
            "sentence_count":       n_sentences,
            "avg_sentence_length":  n_words / n_sentences,
            "type_token_ratio":     len(unique_words) / n_words,
            "punct_density":        punct_count / n_words,
            "comma_density":        text.count(",") / n_words,
            "exclamation_density":  text.count("!") / n_words,
            "question_density":     text.count("?") / n_words,
            "uppercase_ratio":      sum(1 for c in text if c.isupper()) / alpha_count if alpha_count else 0,
            "digit_density":        sum(1 for c in text if c.isdigit()) / n_chars,
            "stopword_ratio":       stopword_count / n_words,
            "long_word_ratio":      long_words / n_words,
            "short_word_ratio":     short_words / n_words,
            "avg_paragraph_length": float(np.mean(para_wc)),
            "paragraph_count":      len(paragraphs) if paragraphs else 1,
        }

    def transform(self, texts):
        records = [self.extract(t) for t in texts]
        df = pd.DataFrame(records)[self.feature_names_]
        return df.values.astype(np.float32)


# ==============================================================================
# MODEL LOADERS — cached so they load only once per session
# ==============================================================================
@st.cache_resource(show_spinner="Loading Traditional model…")
def load_traditional(path: str):
    with open(path, "rb") as f:
        tfidf_vec, clf = pickle.load(f)
    return tfidf_vec, clf


@st.cache_resource(show_spinner="Loading Hybrid model (first load ~30s)…")
def load_hybrid(path: str):
    """
    Load hybrid_model.pkl safely on CPU-only machines.
    The bundle was saved on a GPU machine so torch tensors inside the
    SentenceTransformer must be remapped to CPU on load.
    We monkey-patch torch.load before unpickling so every nested
    torch.load call (including inside SentenceTransformer) uses CPU.
    """
    import torch

    _original_torch_load = torch.load

    def _cpu_torch_load(f, *args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(f, *args, **kwargs)

    torch.load = _cpu_torch_load
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
    finally:
        torch.load = _original_torch_load  # always restore original

    # Move SentenceTransformer model to CPU explicitly
    if "bert_model" in bundle:
        try:
            bundle["bert_model"] = bundle["bert_model"].to(torch.device("cpu"))
        except Exception:
            pass

    return bundle


# ==============================================================================
# PREDICTION HELPERS
# ==============================================================================
def predict_traditional(text: str, tfidf_vec, clf):
    X     = tfidf_vec.transform([text])
    pred  = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]
    return pred, float(max(proba)), proba


def predict_hybrid(text: str, bundle: dict):
    tfidf_vec  = bundle["tfidf_vec"]
    bert_model = bundle["bert_model"]
    ling_ext   = bundle["ling_ext"]
    sc_bert    = bundle["scaler_bert"]
    sc_ling    = bundle["scaler_ling"]
    clf        = bundle["classifier"]

    X_tfidf = tfidf_vec.transform([text])
    X_bert  = sc_bert.transform(bert_model.encode([text], show_progress_bar=False))
    X_ling  = sc_ling.transform(ling_ext.transform([text]))
    X       = sp_hstack([X_tfidf, csr_matrix(X_bert), csr_matrix(X_ling)])

    pred  = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else np.array([0.5, 0.5])
    return pred, float(max(proba)), proba


# ==============================================================================
# UI
# ==============================================================================
st.title("🤖 AI-Generated vs Human Text Detector")
st.markdown("Detect whether text was written by a **Human ✍️** or generated by **AI 🤖**.")
st.divider()
major_path = "major (4).pkl"
# hybrid_path = "hybrid_model (2).pkl"
selected = "Both — Compare side by side"
st.divider()
st.markdown("""
**major.pkl** — tuple:
`(TfidfVectorizer, RandomForest)`

**hybrid_model.pkl** — dict:
`tfidf_vec, bert_model, ling_ext,`
`scaler_bert, scaler_ling, classifier`
    """)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Text", "📄 Batch CSV", "📊 Feature Inspector", "📑 PDF Detector"])


# ══════════════════════════════════════════════════════
# TAB 1 — SINGLE TEXT
# ══════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyse a single piece of text")

    SAMPLES = {
        "— Load a sample —": "",
        "Human sample": (
            "I never really thought much about the stars until that summer my "
            "grandfather took me camping. We lay on our backs in dry grass and he "
            "pointed out Orion, saying every generation of our family had looked at "
            "the same sky. I've never forgotten that quiet moment."
        ),
        "AI sample": (
            "Artificial intelligence represents a transformative paradigm shift in "
            "modern technological development. By leveraging sophisticated machine "
            "learning algorithms and neural network architectures, AI systems are "
            "capable of processing vast datasets and generating contextually relevant "
            "outputs with remarkable accuracy and efficiency across diverse domains."
        ),
    }

    sample_choice = st.selectbox("Load a sample", list(SAMPLES.keys()))

    input_text = st.text_area(
        "Paste or type your text below (50+ words recommended)",
        value=SAMPLES[sample_choice],
        height=200,
    )

    word_count = len(input_text.split()) if input_text.strip() else 0
    st.caption(f"Word count: **{word_count}**")
    if 0 < word_count < 20:
        st.warning("⚠️ Very short text — predictions may be unreliable.")

    if st.button("🔍 Analyse", type="primary"):
        if not input_text.strip():
            st.error("Please enter some text.")
        else:
            results = []

            if "Traditional" in selected or "Both" in selected:
                with st.spinner("Running Traditional model…"):
                    tv, clf = load_traditional(major_path)
                    t0 = time.time()
                    p, c, proba = predict_traditional(input_text, tv, clf)
                    ms = (time.time() - t0) * 1000
                results.append({"name": "Traditional (TF-IDF + RF)", "pred": p,
                                 "conf": c, "proba": proba, "ms": ms})

            if "Traditional" in selected:
                with st.spinner("Running Traditional model…"):
                  tv, clf = load_traditional(major_path)
                  t0 = time.time()
                  p, c, proba = predict_traditional(input_text, tv, clf)
                  ms = (time.time() - t0) * 1000
                results.append({"name": "Traditional (TF-IDF + RF)", "pred": p,
                                "conf": c, "proba": proba, "ms": ms})

            st.divider()
            cols = st.columns(len(results))

            for col, r in zip(cols, results):
                with col:
                    st.markdown(f"**{r['name']}**")
                    label = "🤖 AI-Generated" if r["pred"] == 0 else "✍️ Human-Written"

                    if r["pred"] == 1:
                        st.success(f"## {label}")
                    else:
                        st.error(f"## {label}")

                    st.metric("Confidence",     f"{r['conf']*100:.1f}%")
                    st.metric("Inference time", f"{r['ms']:.0f} ms")

                    st.markdown("**Probability breakdown**")
                    prob_df = pd.DataFrame({
                        "Class":       ["✍️ Human", "🤖 AI"],
                        "Probability": [f"{r['proba'][1]*100:.1f}%",
                                        f"{r['proba'][0]*100:.1f}%"],
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    st.progress(float(r["proba"][1]),
                                text=f"AI probability: {r['proba'][1]*100:.1f}%")

            if len(results) == 2:
                st.divider()
                if results[0]["pred"] == results[1]["pred"]:
                    verdict = "AI-Generated" if results[0]["pred"] == 1 else "Human-Written"
                    st.success(f"✅ Both models agree: **{verdict}**")
                else:
                    st.warning(
                        "⚠️ Models disagree. "
                        "The **Hybrid model** is generally more accurate "
                        "(uses TF-IDF + BERT + linguistic features)."
                    )


# ══════════════════════════════════════════════════════
# TAB 2 — BATCH CSV
# ══════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch prediction from CSV")
    st.markdown("Upload a CSV with a **`text`** column.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    batch_model = st.radio(
        "Model for batch",
        ["Traditional (faster)", "Hybrid (more accurate)"],
        horizontal=True,
    )

    if uploaded:
        df_b = pd.read_csv(uploaded)
        if "text" not in df_b.columns:
            st.error("CSV must have a column named **text**.")
        else:
            st.write(f"**{len(df_b)} rows.** Preview:")
            st.dataframe(df_b.head(), use_container_width=True)

            if st.button("🚀 Run Batch", type="primary"):
                texts_b    = df_b["text"].fillna("").tolist()
                preds_b, confs_b = [], []
                pb = st.progress(0, text="Processing…")

                

              
                tv, clf = load_traditional(major_path)
                for i, txt in enumerate(texts_b):
                  p, c, _ = predict_traditional(txt, tv, clf)
                  preds_b.append(p); confs_b.append(round(c, 4))
                  pb.progress((i + 1) / len(texts_b), text=f"{i+1}/{len(texts_b)}")
                

                pb.empty()
                df_b["prediction"] = preds_b
                df_b["label"]      = ["AI" if p == 1 else "Human" for p in preds_b]
                df_b["confidence"] = confs_b

                ai_n = sum(p == 1 for p in preds_b)
                st.success("✅ Done!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total",    len(preds_b))
                c2.metric("🤖 AI",   ai_n,              f"{ai_n/len(preds_b)*100:.1f}%")
                c3.metric("✍️ Human", len(preds_b)-ai_n, f"{(len(preds_b)-ai_n)/len(preds_b)*100:.1f}%")

                st.dataframe(df_b, use_container_width=True)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=df_b.to_csv(index=False).encode("utf-8"),
                    file_name="ai_detection_results.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════
# TAB 3 — FEATURE INSPECTOR
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("Linguistic Feature Inspector")
    st.markdown(
        "See all **16 linguistic signals** the Hybrid model uses alongside TF-IDF and BERT."
    )

    feat_input = st.text_area("Paste text to inspect", height=180, key="feat_in")

    if st.button("📊 Extract Features") and feat_input.strip():
        ext   = LinguisticFeatureExtractor()
        feats = ext.extract(feat_input)

        st.markdown("#### Feature Values")
        items = list(feats.items())
        for row_start in range(0, len(items), 4):
            cols = st.columns(4)
            for col, (name, val) in zip(cols, items[row_start:row_start + 4]):
                col.metric(name.replace("_", " ").title(), f"{val:.4f}")

        st.divider()

        AI_SIGNALS = {
            "avg_sentence_length":  "Higher → more AI-like",
            "type_token_ratio":     "Higher → more AI-like",
            "long_word_ratio":      "Higher → more AI-like",
            "stopword_ratio":       "Lower  → more AI-like",
            "exclamation_density":  "Lower  → more AI-like",
            "question_density":     "Lower  → more AI-like",
            "uppercase_ratio":      "Lower  → more AI-like",
        }

        feat_rows = [{"Feature": k.replace("_"," ").title(),
                      "Value": round(float(v), 4),
                      "AI Signal": AI_SIGNALS.get(k, "—")}
                     for k, v in feats.items()]
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Normalised Bar Chart")
        chart_df = pd.DataFrame({"Feature": list(feats.keys()),
                                  "Value":   [float(v) for v in feats.values()]})
        max_v = chart_df["Value"].max()
        chart_df["Normalised"] = chart_df["Value"] / max_v if max_v > 0 else chart_df["Value"]
        st.bar_chart(chart_df.set_index("Feature")["Normalised"])

# ══════════════════════════════════════════════════════
# TAB 4 — PDF DETECTOR
# ══════════════════════════════════════════════════════
with tab4:

    st.subheader("Detect AI vs Human Content in PDF")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file:

        text = ""

        # Extract text from PDF
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

        # OCR if scanned
        if len(text.strip()) < 50:

            st.info("Running OCR for scanned PDF...")

            images = convert_from_bytes(pdf_file.read())

            for img in images:
                text += pytesseract.image_to_string(img)

        paragraphs = [
            p.strip()
            for p in re.split(r'\n+', text)
            if len(p.split()) > 6
        ]

        if st.button("Analyse PDF"):

            ai_count = 0
            human_count = 0

            st.subheader("Highlighted Content")

            for para in paragraphs:

                p, c, proba = predict_traditional(para, *load_traditional(major_path))

                if p == 1:

                    human_count += 1

                    st.markdown(
                        f"<div style='background:#d4edda;padding:10px;border-radius:6px;margin-bottom:6px'>"
                        f"<b>Human:</b> {para}</div>",
                        unsafe_allow_html=True
                    )

                else:

                    ai_count += 1

                    st.markdown(
                        f"<div style='background:#f8d7da;padding:10px;border-radius:6px;margin-bottom:6px'>"
                        f"<b>AI:</b> {para}</div>",
                        unsafe_allow_html=True
                    )

            total = ai_count + human_count

            st.divider()
            st.subheader("Document Summary")

            col1, col2 = st.columns(2)

            col1.metric(
                "Human Content",
                human_count,
                f"{(human_count/total)*100:.1f}%"
            )

            col2.metric(
                "AI Content",
                ai_count,
                f"{(ai_count/total)*100:.1f}%"
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("AI Text Detector · major.pkl (Traditional) + hybrid_model.pkl (Hybrid)")