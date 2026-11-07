"""Interactive Decision Tree Classification System (ID3 vs C4.5).

Run with:  streamlit run app.py
"""
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from decision_trees import C45Tree, ID3Tree, to_graphviz, to_text

st.set_page_config(page_title="ID3 vs C4.5 Decision Trees", layout="wide")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def sample_play_tennis():
    rows = [
        ("Sunny", "Hot", "High", "Weak", "No"), ("Sunny", "Hot", "High", "Strong", "No"),
        ("Overcast", "Hot", "High", "Weak", "Yes"), ("Rain", "Mild", "High", "Weak", "Yes"),
        ("Rain", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Cool", "Normal", "Strong", "No"),
        ("Overcast", "Cool", "Normal", "Strong", "Yes"), ("Sunny", "Mild", "High", "Weak", "No"),
        ("Sunny", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Mild", "Normal", "Weak", "Yes"),
        ("Sunny", "Mild", "Normal", "Strong", "Yes"), ("Overcast", "Mild", "High", "Strong", "Yes"),
        ("Overcast", "Hot", "Normal", "Weak", "Yes"), ("Rain", "Mild", "High", "Strong", "No"),
    ]
    return pd.DataFrame(rows, columns=["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"])


@st.cache_data
def sample_iris():
    d = load_iris(as_frame=True)
    df = d.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(d.target_names)))
    return df.drop(columns="target")


# --------------------------------------------------------------------------
# Preprocessing helpers
# --------------------------------------------------------------------------
def fit_bins(X, numeric_cols, n_bins):
    """Quantile bin edges learned on the training set (used by ID3)."""
    edges = {}
    for c in numeric_cols:
        inner = np.unique(np.quantile(X[c], np.linspace(0, 1, n_bins + 1)[1:-1]))
        edges[c] = np.concatenate([[-np.inf], inner, [np.inf]])
    return edges


def apply_bins(X, edges):
    X = X.copy()
    for c, e in edges.items():
        X[c] = pd.cut(X[c], bins=e).astype(str)
    return X


def compute_metrics(y_true, y_pred, n_classes, average):
    kw = dict(average=average, zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, **kw),
        "Recall": recall_score(y_true, y_pred, **kw),
        "F1-score": f1_score(y_true, y_pred, **kw),
        "cm": confusion_matrix(y_true, y_pred, labels=range(n_classes)),
    }


def run_pipeline(df, target, drop_cols, missing, drop_dupes, test_size, seed,
                 n_bins, max_depth, min_split, average):
    d = df.drop(columns=drop_cols, errors="ignore").copy()
    if drop_dupes:
        d = d.drop_duplicates()
    d = d.dropna(subset=[target])
    feats = [c for c in d.columns if c != target]
    if not feats:
        raise ValueError("Select at least one feature column.")

    if missing == "Drop rows with missing values":
        d = d.dropna()
    else:  # impute: median for numeric, mode for categorical
        for c in feats:
            if d[c].isna().any():
                if pd.api.types.is_numeric_dtype(d[c]):
                    d[c] = d[c].fillna(d[c].median())
                else:
                    d[c] = d[c].fillna(d[c].mode().iloc[0])
    if len(d) < 10:
        raise ValueError("Not enough rows left after preprocessing (need at least 10).")

    X = d[feats].copy()
    numeric_cols = [c for c in feats if pd.api.types.is_numeric_dtype(X[c])]
    for c in feats:
        if c not in numeric_cols:
            X[c] = X[c].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(d[target].astype(str))
    classes = list(le.classes_)

    try:
        stratify = y if np.bincount(y).min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)

    edges = fit_bins(Xtr, numeric_cols, n_bins)  # ID3 needs discrete features
    data = {"ID3": (apply_bins(Xtr, edges), apply_bins(Xte, edges)), "C4.5": (Xtr, Xte)}
    models = {
        "ID3": ID3Tree(max_depth=max_depth, min_samples_split=min_split),
        "C4.5": C45Tree(max_depth=max_depth, min_samples_split=min_split),
    }

    out = {"models": {}, "metrics": {}, "test_pred": {}, "reports": {}}
    for name, model in models.items():
        a, b = data[name]
        t0 = time.perf_counter()
        model.fit(a, ytr)
        fit_time = time.perf_counter() - t0
        pred = model.predict(b)
        m = compute_metrics(yte, pred, len(classes), average)
        m["Train accuracy"] = accuracy_score(ytr, model.predict(a))
        m["Depth"], m["Nodes"], m["Leaves"] = model.depth(), model.n_nodes(), model.n_leaves()
        m["Train time (s)"] = fit_time
        out["models"][name], out["metrics"][name], out["test_pred"][name] = model, m, pred
        out["reports"][name] = pd.DataFrame(classification_report(
            yte, pred, labels=range(len(classes)), target_names=classes,
            output_dict=True, zero_division=0)).T

    out.update(classes=classes, edges=edges, numeric_cols=numeric_cols, feats=feats,
               Xtr=Xtr, Xte=Xte, yte=yte, n_rows=len(d), average=average)
    return out


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def plot_cm(cm, classes, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_comparison(metrics):
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (algo, m) in enumerate(metrics.items()):
        bars = ax.bar(x + (k - 0.5) * w, [m[n] for n in names], w, label=algo)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_title("ID3 vs C4.5 (test set)")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("Decision Tree Classification System: ID3 vs C4.5")

st.sidebar.header("1. Dataset")
source = st.sidebar.radio("Source", ["Upload CSV", "Sample: Play Tennis", "Sample: Iris"])
if source == "Upload CSV":
    up = st.sidebar.file_uploader("Classification dataset (.csv)", type="csv")
    df = pd.read_csv(up) if up else None
elif source == "Sample: Play Tennis":
    df = sample_play_tennis()
else:
    df = sample_iris()

if df is None:
    st.info("Upload a CSV file (or pick a sample dataset) in the sidebar to begin.")
    st.stop()

cols = list(df.columns)
target = st.sidebar.selectbox("Target attribute", cols, index=len(cols) - 1)
drop_cols = st.sidebar.multiselect(
    "Columns to drop (e.g. IDs)", [c for c in cols if c != target])

st.sidebar.header("2. Preprocessing")
missing = st.sidebar.radio(
    "Missing values", ["Impute (median / mode)", "Drop rows with missing values"])
drop_dupes = st.sidebar.checkbox("Remove duplicate rows", value=True)
n_bins = st.sidebar.slider("Bins for numeric features (ID3 only)", 2, 10, 3)

st.sidebar.header("3. Training")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
max_depth = st.sidebar.slider("Max depth (0 = unlimited)", 0, 15, 0)
min_split = st.sidebar.slider("Min samples to split", 2, 20, 2)
average = st.sidebar.selectbox("Metric averaging", ["weighted", "macro", "micro"])
train = st.sidebar.button("Train & Evaluate", type="primary")

if train:
    try:
        with st.spinner("Preprocessing and training..."):
            st.session_state["res"] = run_pipeline(
                df, target, drop_cols, missing, drop_dupes, test_size, int(seed),
                n_bins, max_depth or None, min_split, average)
    except Exception as e:
        st.error(f"Could not train models: {e}")
        st.session_state.pop("res", None)

tab_data, tab_id3, tab_c45, tab_eval, tab_cmp, tab_pred = st.tabs(
    ["Data", "ID3 Tree", "C4.5 Tree", "Evaluation", "Comparison", "Predict"])

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(50), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(cols))
    c3.metric("Missing cells", int(df.isna().sum().sum()))
    st.write("**Column types and missing values**")
    st.dataframe(pd.DataFrame({
        "dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
        "unique": df.nunique()}), width="stretch")
    st.write(f"**Target class distribution ({target})**")
    st.bar_chart(df[target].value_counts())

res = st.session_state.get("res")
if res is None:
    for t in (tab_id3, tab_c45, tab_eval, tab_cmp, tab_pred):
        with t:
            st.info("Configure the options in the sidebar and click **Train & Evaluate**.")
    st.stop()

classes = res["classes"]

for tab, name in ((tab_id3, "ID3"), (tab_c45, "C4.5")):
    with tab:
        model, m = res["models"][name], res["metrics"][name]
        c1, c2, c3 = st.columns(3)
        c1.metric("Depth", m["Depth"])
        c2.metric("Nodes", m["Nodes"])
        c3.metric("Leaves", m["Leaves"])
        if name == "ID3":
            st.caption("ID3 works on categorical data, so numeric features were "
                       "discretized into quantile bins before training.")
        else:
            st.caption("C4.5 uses gain ratio and binary thresholds for numeric features.")
        st.graphviz_chart(to_graphviz(model, classes).source, width="stretch")
        with st.expander("Text view of the tree"):
            st.code(to_text(model, classes))

with tab_eval:
    st.caption(f"{res['n_rows']} rows after preprocessing, "
               f"{len(res['yte'])} in the test set, {res['average']}-averaged metrics.")
    cols_eval = st.columns(2)
    for col, name in zip(cols_eval, ("ID3", "C4.5")):
        m = res["metrics"][name]
        with col:
            st.subheader(name)
            a, b = st.columns(2)
            a.metric("Accuracy", f"{m['Accuracy']:.3f}")
            b.metric("Precision", f"{m['Precision']:.3f}")
            a.metric("Recall", f"{m['Recall']:.3f}")
            b.metric("F1-score", f"{m['F1-score']:.3f}")
            st.pyplot(plot_cm(m["cm"], classes, f"{name} confusion matrix"))
            st.write("**Per-class report**")
            st.dataframe(res["reports"][name].round(3), width="stretch")

with tab_cmp:
    keys = ["Accuracy", "Precision", "Recall", "F1-score", "Train accuracy",
            "Depth", "Nodes", "Leaves", "Train time (s)"]
    table = pd.DataFrame({n: {k: res["metrics"][n][k] for k in keys}
                          for n in ("ID3", "C4.5")})
    st.dataframe(table.style.format("{:.4f}"), width="stretch")
    st.pyplot(plot_comparison(res["metrics"]))
    i3, c45 = res["metrics"]["ID3"], res["metrics"]["C4.5"]
    if abs(i3["F1-score"] - c45["F1-score"]) < 1e-9:
        st.info("Both algorithms reach the same F1-score on this split.")
    else:
        w = "ID3" if i3["F1-score"] > c45["F1-score"] else "C4.5"
        st.success(f"**{w}** has the higher F1-score on this test split.")
    gap = lambda m: m["Train accuracy"] - m["Accuracy"]
    st.caption(f"Train-test accuracy gap (overfitting indicator): "
               f"ID3 = {gap(i3):.3f}, C4.5 = {gap(c45):.3f}.")

with tab_pred:
    st.subheader("Predictions on the test set")
    actual = [classes[i] for i in res["yte"]]
    pred_df = res["Xte"].copy().reset_index(drop=True)
    pred_df["Actual"] = actual
    for n in ("ID3", "C4.5"):
        pred_df[f"{n} prediction"] = [classes[i] for i in res["test_pred"][n]]
    st.dataframe(pred_df, width="stretch")
    st.download_button("Download predictions (CSV)", pred_df.to_csv(index=False),
                       "predictions.csv", "text/csv")

    st.subheader("Predict a single new sample")
    with st.form("manual"):
        values = {}
        grid = st.columns(3)
        for i, c in enumerate(res["feats"]):
            with grid[i % 3]:
                if c in res["numeric_cols"]:
                    values[c] = st.number_input(c, value=float(res["Xtr"][c].median()))
                else:
                    values[c] = st.selectbox(c, sorted(res["Xtr"][c].unique()))
        go = st.form_submit_button("Predict")
    if go:
        row = pd.DataFrame([values])
        r3 = res["models"]["ID3"].predict(apply_bins(row, res["edges"]))[0]
        r45 = res["models"]["C4.5"].predict(row)[0]
        a, b = st.columns(2)
        a.metric("ID3 predicts", classes[r3])
        b.metric("C4.5 predicts", classes[r45])