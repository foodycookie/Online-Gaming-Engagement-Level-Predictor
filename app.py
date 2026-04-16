# ----------------------------------------------------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------------------------------------------------

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Online Gaming Engagement Level Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------------------------------------------------------------
# Global CSS
# ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Sora:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
h1,h2,h3,h4 { font-family: 'IBM Plex Mono', monospace !important; }

.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f0f0f0;
    border-bottom: 2px solid #22d3ee;
    padding-bottom: 0.4rem;
    margin-bottom: 0.2rem;
}
.page-sub {
    font-size: 0.82rem;
    color: #888;
    margin-bottom: 1.2rem;
    font-family: 'Sora', sans-serif;
}
.sec {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #22d3ee;
    margin: 1.6rem 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #22d3ee33, transparent);
}
.kcard {
    background: #111827;
    border: 1px solid #1f2937;
    border-top: 2px solid #22d3ee;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.kcard .kl {
    font-size: 0.65rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'IBM Plex Mono', monospace;
}
.kcard .kv {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: #f9fafb;
    margin-top: 2px;
}
.kcard .ksub {
    font-size: 0.7rem;
    color: #9ca3af;
    margin-top: 2px;
}
.info-strip {
    background: #0c1a1f;
    border-left: 3px solid #22d3ee;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0.5rem 0 1rem 0;
    line-height: 1.6;
}
.warn-strip {
    background: #1a1200;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: #d97706;
    margin: 0.5rem 0 1rem 0;
}
.ok-strip {
    background: #021a0c;
    border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: #22c55e;
    margin: 0.5rem 0 1rem 0;
}
.badge-high   { background:#052e16; color:#4ade80; border:1px solid #16a34a; border-radius:6px; padding:0.15rem 0.6rem; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.9rem; }
.badge-medium { background:#1c1400; color:#fbbf24; border:1px solid #d97706; border-radius:6px; padding:0.15rem 0.6rem; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.9rem; }
.badge-low    { background:#1a0505; color:#f87171; border:1px solid #dc2626; border-radius:6px; padding:0.15rem 0.6rem; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.9rem; }
.model-compare-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}
.model-compare-card.winner { border-color: #22d3ee; border-width: 2px; }
.mc-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.6rem;
}
.mc-badge-winner {
    font-size: 0.6rem;
    background: #164e63;
    color: #22d3ee;
    border: 1px solid #22d3ee;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    margin-left: 8px;
    vertical-align: middle;
}
.prob-bar-wrap { display:flex; align-items:center; gap:8px; margin:5px 0; }
.prob-label { width:58px; font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#94a3b8; }
.prob-track { flex:1; background:#1e293b; border-radius:4px; height:14px; overflow:hidden; }
.prob-fill  { height:14px; border-radius:4px; }
.prob-pct   { width:44px; text-align:right; font-family:'IBM Plex Mono',monospace; font-size:0.72rem; }
.baseline-row {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}
.bl-icon { font-size: 1.8rem; line-height: 1; }
.bl-title { font-family:'IBM Plex Mono',monospace; font-size:0.9rem; font-weight:700; margin-bottom:4px; }
.bl-desc  { font-size:0.8rem; color:#94a3b8; line-height:1.5; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------------------
# Load artefacts
# ----------------------------------------------------------------------------------------------------------------------------------

@st.cache_resource
def load_artefacts():
    fc     = joblib.load('model/feature_columns.pkl')
    models = {
        "Decision Tree" : joblib.load('model/Decision Tree.pkl'),
        "Random Forest" : joblib.load('model/Random Forest.pkl'),
        "XGBoost"       : joblib.load('model/XGBoost.pkl'),
    }
    return fc, models

@st.cache_data
def load_data():
    return pd.read_csv('online_gaming_behavior_dataset.csv')

feature_columns, models_dict = load_artefacts()
df_raw = load_data()

LABEL_MAP   = {0: 'Low', 1: 'Medium', 2: 'High'}
LABEL_NAMES = ['Low', 'Medium', 'High']

# ----------------------------------------------------------------------------------------------------------------------------------
# Shared data preparation
# ----------------------------------------------------------------------------------------------------------------------------------

@st.cache_data
def get_encoded(raw):
    df = raw.copy().drop(["PlayerID"], axis=1)
    df = pd.get_dummies(df, columns=["Location"], drop_first=True)
    df = pd.get_dummies(df, columns=["GameGenre"],  drop_first=True)
    df["GameDifficulty"]  = df["GameDifficulty"].map({"Easy": 0, "Medium": 1, "Hard": 2})
    df["EngagementLevel"] = df["EngagementLevel"].map({"Low": 0, "Medium": 1, "High": 2})
    df["Gender"]          = df["Gender"].map({"Male": 1, "Female": 0})
    return df

df_enc = get_encoded(df_raw)

@st.cache_data
def get_split(enc):
    X = enc.drop(columns=['EngagementLevel'])
    y = enc['EngagementLevel']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_test, y_train, y_test = get_split(df_enc)

@st.cache_data
def compute_eval(_mdict, _Xts, _yt):
    out = {}
    for name, mdl in _mdict.items():
        yp = mdl.predict(_Xts)
        out[name] = {
            "y_pred"   : yp,
            "accuracy" : accuracy_score(_yt, yp),
            "precision": precision_score(_yt, yp, average='weighted'),
            "recall"   : recall_score(_yt, yp, average='weighted'),
            "f1"       : f1_score(_yt, yp, average='weighted'),
        }
    return out

eval_results = compute_eval(models_dict, X_test, y_test)

# ----------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------

def cramers_v(c1, c2):
    ct   = pd.crosstab(c1, c2)
    chi2 = chi2_contingency(ct)[0]
    n    = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def eta_squared(num_col, cat_col):
    groups = [g.values for _, g in num_col.groupby(cat_col)]
    gm = num_col.mean()
    ssb = sum(len(g) * (g.mean() - gm)**2 for g in groups)
    sst = sum((x - gm)**2 for g in groups for x in g)
    return ssb / sst if sst else 0

def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

def kcard(col, label, value, sub=""):
    col.markdown(f"""<div class="kcard"><div class="kl">{label}</div>
    <div class="kv">{value}</div><div class="ksub">{sub}</div></div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:700;color:#22d3ee;margin-bottom:2px">
    🎮 Online Gaming Engagement Level Analyser
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("", [
        "🔍 Predictor",
        "🛠 Preprocessing",
        "🔗 Correlation",
        "📈 Model Evaluation",
        "🧪 Parameter Tuning",
    ], label_visibility="collapsed")

# ----------------------------------------------------------------------------------------------------------------------------------
# Page - Predictor
# ----------------------------------------------------------------------------------------------------------------------------------

if page == "🔍 Predictor":
    st.markdown('<div class="page-title">🔍 Online Gaming Engagement Level Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter player attributes, all three models run simultaneously and results are compared side-by-side.</div>', unsafe_allow_html=True)

    # Baseline reference
    sec("Engagement Level Baseline")
    st.markdown("""
    <div class="baseline-row">
        <div class="bl-icon">🔴</div>
        <div>
            <div class="bl-title" style="color:#f87171">Low Engagement</div>
            <div class="bl-desc">Player is rarely active. Characterised by low play time, few sessions per week, minimal achievements, and little investment in the game (low level, no purchases).</div>
        </div>
    </div>
    <div class="baseline-row">
        <div class="bl-icon">🟡</div>
        <div>
            <div class="bl-title" style="color:#fbbf24">Medium Engagement</div>
            <div class="bl-desc">Player is moderately active. Plays regularly but not intensively. Has mid-range level, occasional in-game purchases, and moderate session frequency and duration.</div>
        </div>
    </div>
    <div class="baseline-row">
        <div class="bl-icon">🟢</div>
        <div>
            <div class="bl-title" style="color:#4ade80">High Engagement</div>
            <div class="bl-desc">Highly committed player. Long and frequent sessions, high player level, many achievements unlocked, and likely makes in-game purchases.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    sec("Player Attributes")
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        with c1:
            age             = st.number_input("Age", 15, 49, 25, 1)
            gender          = st.radio("Gender", ["Male", "Female"], horizontal=True)
            location        = st.selectbox("Location", ["Asia", "Europe", "USA", "Other"])
            game_genre      = st.selectbox("Game Genre", ["Action", "RPG", "Simulation", "Sports", "Strategy"])
            game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
        with c2:
            play_time_hours       = st.number_input("Play Time Hours / Day", 0.0, 24.0, 10.0, 0.5)
            in_game_purchases     = st.selectbox("In-Game Purchases", [0, 1], format_func=lambda x: "Yes" if x else "No")
            sessions_per_week     = st.number_input("Sessions Per Week", 0, 19, 5, 1)
            avg_session_duration  = st.number_input("Avg Session Duration (Min)", 10, 179, 100, 5)
            player_level          = st.number_input("Player Level", 1, 99, 20, 1)
            achievements_unlocked = st.number_input("Achievements Unlocked", 0, 49, 15, 1)
        submitted = st.form_submit_button("⚡ Run All 3 Models", use_container_width=True)

    if submitted:
        diff_enc = {"Easy": 0, "Medium": 1, "Hard": 2}
        input_df = pd.DataFrame({
            "Age"                       : [age],
            "Gender"                    : [1 if gender == "Male" else 0],
            "PlayTimeHours"             : [play_time_hours],
            "InGamePurchases"           : [in_game_purchases],
            "GameDifficulty"            : [diff_enc[game_difficulty]],
            "SessionsPerWeek"           : [sessions_per_week],
            "AvgSessionDurationMinutes" : [avg_session_duration],
            "PlayerLevel"               : [player_level],
            "AchievementsUnlocked"      : [achievements_unlocked],
            "Location_Europe"           : [1 if location == "Europe" else 0],
            "Location_Other"            : [1 if location == "Other"  else 0],
            "Location_USA"              : [1 if location == "USA"    else 0],
            "GameGenre_RPG"             : [1 if game_genre == "RPG"        else 0],
            "GameGenre_Simulation"      : [1 if game_genre == "Simulation" else 0],
            "GameGenre_Sports"          : [1 if game_genre == "Sports"     else 0],
            "GameGenre_Strategy"        : [1 if game_genre == "Strategy"   else 0],
        })[feature_columns]

        preds = {}
        for name, mdl in models_dict.items():
            pred  = mdl.predict(input_df)[0]
            probs = mdl.predict_proba(input_df)[0]
            preds[name] = {"label": LABEL_MAP[pred], "probs": probs}

        labels   = [v["label"] for v in preds.values()]
        majority = max(set(labels), key=labels.count)
        agreed   = len(set(labels)) == 1

        sec("Prediction Results - All 3 Models")
        if agreed:
            st.markdown(f'<div class="ok-strip">✅ All 3 models agree: <b>{majority}</b> engagement level.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-strip">⚠️ Models disagree. Majority vote: <b>{majority}</b>.</div>', unsafe_allow_html=True)

        best_model = max(eval_results, key=lambda n: eval_results[n]["f1"])

        cols = st.columns(3)
        for col, (name, res) in zip(cols, preds.items()):
            lbl   = res["label"]
            probs = res["probs"]
            is_best   = (name == best_model)
            badge_cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[lbl]
            icon      = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[lbl]
            winner_tag = '<span class="mc-badge-winner">★ Best F1</span>' if is_best else ''
            card_cls  = "model-compare-card winner" if is_best else "model-compare-card"
            bar_colors = {"Low": "#f87171", "Medium": "#fbbf24", "High": "#4ade80"}

            prob_bars = ""
            for i, ln in enumerate(LABEL_NAMES):
                p  = probs[i]
                bc = bar_colors[ln]
                prob_bars += f"""
                <div class="prob-bar-wrap">
                    <span class="prob-label">{ln}</span>
                    <div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%;background:{bc}"></div></div>
                    <span class="prob-pct" style="color:{bc}">{p*100:.1f}%</span>
                </div>"""

            col.markdown(f"""
            <div class="{card_cls}">
                <div class="mc-name">{name}{winner_tag}</div>
                <div style="font-size:2rem;margin:0.3rem 0">{icon}</div>
                <span class="{badge_cls}">{lbl}</span>
                <div style="margin-top:1rem">{prob_bars}</div>
                <div style="font-size:0.67rem;color:#4b5563;font-family:'IBM Plex Mono',monospace;margin-top:0.8rem">
                    Test F1: {eval_results[name]['f1']:.4f} &nbsp;·&nbsp; Acc: {eval_results[name]['accuracy']:.4f}
                </div>
            </div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------------------
# Page - Preprocessing
# ----------------------------------------------------------------------------------------------------------------------------------

elif page == "🛠 Preprocessing":
    st.markdown('<div class="page-title">🛠 Preprocessing</div>', unsafe_allow_html=True)

    df_nodrop = df_raw.drop(["PlayerID"], axis=1)

    sec("Dataset Shape")
    c1, c2, c3 = st.columns(3)
    kcard(c1, "Total Records",       f"{len(df_raw):,}",         "rows")
    kcard(c2, "Original Features",   str(df_raw.shape[1]),       "incl. PlayerID")
    kcard(c3, "After Drop PlayerID", str(df_nodrop.shape[1]),    "features")

    c1, c2 = st.columns(2)
    kcard(c1, "Shape Before Encoding", f"{df_nodrop.shape[0]} × {df_nodrop.shape[1]}", "rows × cols")
    kcard(c2, "Shape After Encoding",  f"{df_enc.shape[0]} × {df_enc.shape[1]}",       f"+{df_enc.shape[1]-df_nodrop.shape[1]} cols from one-hot")

    sec("Train / Test Split")
    st.markdown('<div class="info-strip">80 / 20 stratified split with <code>random_state=42</code> to preserve class proportions in both sets.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kcard(c1, "X Train", str(X_train.shape), "rows × features")
    kcard(c2, "X Test",  str(X_test.shape),  "rows × features")
    kcard(c3, "y Train", str(y_train.shape), "labels")
    kcard(c4, "y Test",  str(y_test.shape),  "labels")

    sec("Class Imbalance Check (SMOTE)")
    y_full   = df_enc['EngagementLevel']
    counts   = y_full.value_counts().sort_index()
    majority = counts.max()
    ratio    = counts.min() / majority

    imb = pd.DataFrame({
        "Class"             : [LABEL_MAP[k] for k in counts.index],
        "Count"             : counts.values,
        "% of Total"        : (counts.values / len(y_full) * 100).round(2),
        "Ratio vs Majority" : (counts.values / majority).round(4),
    })
    st.dataframe(imb, use_container_width=True, hide_index=True)

    THRESH = 0.50
    if ratio < THRESH:
        st.markdown(f'<div class="warn-strip">⚠️ Minority/Majority ratio = <b>{ratio:.2f}</b> &lt; threshold <b>{THRESH}</b>. SMOTE was applied to the training set to balance classes.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ok-strip">✅ Minority/Majority ratio = <b>{ratio:.2f}</b> ≥ threshold <b>{THRESH}</b>. Classes are sufficiently balanced — SMOTE not applied.</div>', unsafe_allow_html=True)
    
# ----------------------------------------------------------------------------------------------------------------------------------
# Page - Correlation
# ----------------------------------------------------------------------------------------------------------------------------------

elif page == "🔗 Correlation":
    st.markdown('<div class="page-title">🔗 Correlation Analysis</div>', unsafe_allow_html=True)

    df_corr  = df_raw.drop(["PlayerID"], axis=1)
    num_cols = [c for c in df_corr.select_dtypes(include=np.number).columns if c not in ["InGamePurchases", "PlayerLevel"]]
    cat_cols = list(df_corr.select_dtypes(include='object').columns) + ['InGamePurchases', 'PlayerLevel']

    tab1, tab2, tab3 = st.tabs(["Pearson (Num-Num)", "Cramér's V (Cat-Cat)", "Eta² (Num-Cat)"])

    with tab1:
        pearson = df_corr[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pearson, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, ax=ax)
        ax.set_title("Pearson Correlation", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        with st.spinner("Computing Cramér's V…"):
            cv_mat = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
            for a in cat_cols:
                for b in cat_cols:
                    cv_mat.loc[a, b] = cramers_v(df_corr[a], df_corr[b])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cv_mat.astype(float), annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=1, linewidths=0.5, ax=ax)
        ax.set_title("Cramér's V", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        with st.spinner("Computing Eta Squared…"):
            eta_mat = pd.DataFrame(index=num_cols, columns=cat_cols, dtype=float)
            for n in num_cols:
                for c in cat_cols:
                    eta_mat.loc[n, c] = eta_squared(df_corr[n], df_corr[c])
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(eta_mat.astype(float), annot=True, fmt='.2f', cmap='BuGn', vmin=0, vmax=1, linewidths=0.5, ax=ax)
        ax.set_title("Eta Squared", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ----------------------------------------------------------------------------------------------------------------------------------
# Page - Model Evaluation
# ----------------------------------------------------------------------------------------------------------------------------------

elif page == "📈 Model Evaluation":
    st.markdown('<div class="page-title">📈 Model Evaluation</div>', unsafe_allow_html=True)
    
    model_names    = list(eval_results.keys())
    metrics_keys   = ["accuracy", "precision", "recall", "f1"]
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    @st.cache_data
    def get_train_acc(_mdict, _Xtr, _ytr):
        out = {}
        for name, mdl in _mdict.items():
            yp = mdl.predict(_Xtr)
            out[name] = accuracy_score(_ytr, yp)
        return out

    train_accs = get_train_acc(models_dict, X_train, y_train)
    CV_ACCS    = {"Decision Tree": 0.9059, "Random Forest": 0.9086, "XGBoost": 0.9167}
    best_model = max(eval_results, key=lambda n: eval_results[n]["f1"])

    sec("All Metrics Comparison")
    colors_bar = ['#22d3ee', '#818cf8', '#f97316']
    fig, axes  = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Model Performance — All Metrics", fontsize=13, fontweight='bold')
    for ax, mk, ml in zip(axes, metrics_keys, metrics_labels):
        vals = [eval_results[n][mk] for n in model_names]
        bars = ax.bar(model_names, vals, color=colors_bar, edgecolor='none', width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f'{v:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylim(0.5, 1.05); ax.set_title(ml, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    sec("Cross Validation Accuracy and Test Accuracy")
    sum_df = pd.DataFrame({
        "Model"                  : ["Decision Tree", "Random Forest", "XGBoost"],
        "CV Accuracy"            : [0.9059, 0.9086, 0.9167],
        "Test Accuracy"          : [eval_results[n]["accuracy"] for n in ["Decision Tree", "Random Forest", "XGBoost"]],
        "Test F1"                : [eval_results[n]["f1"]       for n in ["Decision Tree", "Random Forest", "XGBoost"]],
        "Δ (CV − Test)"          : [round(0.9059 - eval_results["Decision Tree"]["accuracy"], 4),
                                    round(0.9086 - eval_results["Random Forest"]["accuracy"], 4),
                                    round(0.9167 - eval_results["XGBoost"]["accuracy"],       4)],
    })
    st.dataframe(sum_df.round(4), use_container_width=True, hide_index=True)

    sec("Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight='bold')
    for ax, (name, res) in zip(axes, eval_results.items()):
        cm   = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.4f}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------------------
# Page - Parameter Tuning
# ----------------------------------------------------------------------------------------------------------------------------------

elif page == "🧪 Parameter Tuning":
    st.markdown('<div class="page-title">🧪 Parameter Tuning</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">RandomizedSearchCV · 5-fold CV · n_iter=20 · scoring=accuracy. Range tested vs final best value selected.</div>', unsafe_allow_html=True)

    sec("Model 1 — Decision Tree")
    dt_df = pd.DataFrame([
        {"Parameter": "criterion",         "Range Tested": "gini, entropy",          "Best Value": "entropy", "Explanation": "Entropy chosen; measures information gain at each split"},
        {"Parameter": "max_depth",         "Range Tested": "3, 5, 10, 15, 20, None", "Best Value": "10",      "Explanation": "Limits tree depth to control overfitting"},
        {"Parameter": "min_samples_split", "Range Tested": "2, 5, 10, 20",           "Best Value": "20",      "Explanation": "Higher value prevents splitting on very small nodes"},
        {"Parameter": "min_samples_leaf",  "Range Tested": "1, 2, 5, 10",            "Best Value": "10",      "Explanation": "Each leaf must contain at least 10 samples"},
    ])
    st.dataframe(dt_df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    kcard(c1, "Best CV Accuracy", "0.9059", "5-fold · n_iter=20")
    kcard(c2, "Test Accuracy",    f"{eval_results['Decision Tree']['accuracy']:.4f}", "held-out 20%")
    with st.expander("Final constructor"):
        st.code("DecisionTreeClassifier(\n    criterion='entropy', max_depth=10,\n    min_samples_leaf=10, min_samples_split=20,\n    random_state=42\n)", language="python")

    st.divider()

    sec("Model 2 — Random Forest")
    rf_df = pd.DataFrame([
        {"Parameter": "n_estimators",      "Range Tested": "100, 200, 300",    "Best Value": "200",   "Explanation": "200 trees balances accuracy and compute cost"},
        {"Parameter": "max_depth",         "Range Tested": "None, 10, 20, 30", "Best Value": "30",    "Explanation": "Deep trees allowed; bagging provides built-in regularisation"},
        {"Parameter": "min_samples_split", "Range Tested": "2, 5, 10",         "Best Value": "2",     "Explanation": "Minimum 2 samples to attempt a split"},
        {"Parameter": "min_samples_leaf",  "Range Tested": "1, 2, 4",          "Best Value": "1",     "Explanation": "Leaf can contain a single sample"},
        {"Parameter": "max_features",      "Range Tested": "sqrt, log2",       "Best Value": "log2",  "Explanation": "log2(n_features) per split to reduces correlation between trees"},
    ])
    st.dataframe(rf_df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    kcard(c1, "Best CV Accuracy", "0.9086", "5-fold · n_iter=20")
    kcard(c2, "Test Accuracy",    f"{eval_results['Random Forest']['accuracy']:.4f}", "held-out 20%")
    with st.expander("Final constructor"):
        st.code("RandomForestClassifier(\n    n_estimators=200, max_depth=30,\n    min_samples_split=2, min_samples_leaf=1,\n    max_features='log2', random_state=42\n)", language="python")

    st.divider()

    sec("Model 3 — XGBoost")
    xgb_df = pd.DataFrame([
        {"Parameter": "n_estimators",     "Range Tested": "100, 200, 300",          "Best Value": "200",  "Explanation": "Number of boosting rounds"},
        {"Parameter": "max_depth",        "Range Tested": "3, 5, 7, 9",             "Best Value": "9",    "Explanation": "Deeper trees capture more complex patterns"},
        {"Parameter": "learning_rate",    "Range Tested": "0.01, 0.05, 0.1, 0.2",  "Best Value": "0.1",  "Explanation": "Step-size shrinkage; 0.1 is a common sweet spot"},
        {"Parameter": "subsample",        "Range Tested": "0.7, 0.8, 1.0",          "Best Value": "1.0",  "Explanation": "Use 100% of training rows per tree"},
        {"Parameter": "colsample_bytree", "Range Tested": "0.7, 0.8, 1.0",          "Best Value": "1.0",  "Explanation": "Use 100% of features per tree"},
    ])
    st.dataframe(xgb_df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    kcard(c1, "Best CV Accuracy", "0.9167", "5-fold · n_iter=20")
    kcard(c2, "Test Accuracy",    f"{eval_results['XGBoost']['accuracy']:.4f}", "held-out 20%")
    with st.expander("Final constructor"):
        st.code("XGBClassifier(\n    n_estimators=200, max_depth=9,\n    learning_rate=0.1, subsample=1.0,\n    colsample_bytree=1.0,\n    user_label_encoder=False,\n    eval_metric='mlogloss', random_state=42\n)", language="python")
    
    st.divider()