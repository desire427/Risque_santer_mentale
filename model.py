import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.inspection import permutation_importance
import io
import kagglehub

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maternal Health Risk — ML Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Palette & styles ────────────────────────────────────────────────────────
PALETTE = {
    "bg":         "#0E1117",
    "card":       "#161B22",
    "border":     "#30363D",
    "accent":     "#58A6FF",
    "accent2":    "#3FB950",
    "accent3":    "#D29922",
    "accent4":    "#F85149",
    "text":       "#E6EDF3",
    "muted":      "#8B949E",
    "low":        "#3FB950",
    "mid":        "#D29922",
    "high":       "#F85149",
}

RISK_COLORS = {"low risk": PALETTE["low"], "mid risk": PALETTE["mid"], "high risk": PALETTE["high"]}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: 'Inter', sans-serif;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background-color: {PALETTE['card']};
    border-right: 1px solid {PALETTE['border']};
  }}
  section[data-testid="stSidebar"] * {{ color: {PALETTE['text']} !important; }}

  /* Headings */
  h1, h2, h3 {{ font-family: 'Syne', sans-serif !important; }}

  /* Metric cards */
  [data-testid="metric-container"] {{
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 16px 20px;
  }}
  [data-testid="metric-container"] label {{
    color: {PALETTE['muted']} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {PALETTE['accent']} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.8rem !important;
  }}

  /* Tabs */
  button[data-baseweb="tab"] {{
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: {PALETTE['muted']} !important;
    border-bottom: 2px solid transparent;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PALETTE['accent']} !important;
    border-bottom: 2px solid {PALETTE['accent']} !important;
  }}

  /* Buttons */
  .stButton > button {{
    background: {PALETTE['accent']};
    color: #000;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; color: #000; }}

  /* Selectbox / sliders */
  .stSelectbox > div > div,
  .stSlider > div {{ color: {PALETTE['text']} !important; }}

  /* Divider */
  hr {{ border-color: {PALETTE['border']}; margin: 1.5rem 0; }}

  /* Code mono label */
  .mono {{ font-family: 'DM Mono', monospace; color: {PALETTE['accent']}; }}

  /* Card block */
  .card {{
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }}

  /* Risk badge */
  .badge-low  {{ background:#0d2818; color:{PALETTE['low']};  border:1px solid {PALETTE['low']};  padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}
  .badge-mid  {{ background:#2b2109; color:{PALETTE['mid']};  border:1px solid {PALETTE['mid']};  padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}
  .badge-high {{ background:#2b0a0a; color:{PALETTE['high']}; border:1px solid {PALETTE['high']}; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }}

  /* Dataframe */
  .stDataFrame {{ border: 1px solid {PALETTE['border']}; border-radius: 8px; }}

  /* Header banner */
  .header-banner {{
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }}
  .header-banner::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {PALETTE['accent4']}, {PALETTE['accent3']}, {PALETTE['accent2']}, {PALETTE['accent']});
  }}
  .header-title {{
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: {PALETTE['text']};
    margin: 0 0 0.25rem 0;
    line-height: 1.1;
  }}
  .header-sub {{
    color: {PALETTE['muted']};
    font-size: 0.9rem;
    margin: 0;
    letter-spacing: 0.03em;
  }}
</style>
""", unsafe_allow_html=True)

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load the Kaggle Maternal Health Risk dataset."""
    path = kagglehub.dataset_download("csafrit2/maternal-health-risk-data")
    df = pd.read_csv(f"{path}/Maternal Health Risk Data Set.csv")
    df.columns = [c.strip() for c in df.columns]
    if "RiskLevel" in df.columns:
        df["RiskLevel"] = df["RiskLevel"].str.strip().str.lower()
    return df

@st.cache_data
def prepare_ml(df):
    le = LabelEncoder()
    df2 = df.copy()
    df2["RiskLevel_enc"] = le.fit_transform(df2["RiskLevel"])

    X = df2.drop(["RiskLevel", "RiskLevel_enc"], axis=1)
    y = df2["RiskLevel_enc"]
    classes = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler, le, classes

# ─── Matplotlib dark style ───────────────────────────────────────────────────
def dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE["card"])
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["muted"])
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.title.set_color(PALETTE["text"])
    return fig, ax

def dark_fig_multi(rows, cols, figsize):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor(PALETTE["card"])
    for ax in np.array(axes).flatten():
        ax.set_facecolor(PALETTE["card"])
        ax.tick_params(colors=PALETTE["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.xaxis.label.set_color(PALETTE["muted"])
        ax.yaxis.label.set_color(PALETTE["muted"])
        ax.title.set_color(PALETTE["text"])
    return fig, axes

# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
df = load_data()
X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler, le, classes = prepare_ml(df)

with st.sidebar:
    st.markdown(f"""
    <div style='margin-bottom:1.5rem'>
      <p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                color:{PALETTE["text"]};margin:0'>Maternal Health Risk</p>
      <p style='font-size:0.75rem;color:{PALETTE["muted"]};margin:0'>ML Classification Project</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    page = st.radio("", [
        "Overview & EDA",
        "Feature Analysis",
        "Model Training",
        "Model Comparison",
        "Prediction Tool",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"""
    <div style='font-size:0.8rem;color:{PALETTE["muted"]}'>
      <div style='margin-bottom:4px'>Samples: <span class='mono'>{len(df)}</span></div>
      <div style='margin-bottom:4px'>Features: <span class='mono'>{len(df.columns)-1}</span></div>
      <div>Target classes: <span class='mono'>{len(classes)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    uploaded = st.file_uploader("Upload custom CSV", type=["csv"], help="Must contain the same columns as the Kaggle dataset")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df.columns = [c.strip() for c in df.columns]
            if "RiskLevel" in df.columns:
                df["RiskLevel"] = df["RiskLevel"].str.strip().str.lower()
            X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler, le, classes = prepare_ml(df)
            st.success("Dataset chargé avec succès")
        except Exception as e:
            st.error(f"Erreur: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW & EDA
# ──────────────────────────────────────────────────────────────────────────────
if page == "Overview & EDA":
    st.markdown(f"""
    <div class="header-banner">
      <p class="header-title">Prédiction du Risque Santé Maternelle</p>
      <p class="header-sub">Supervised Machine Learning · Classification · Kaggle Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    counts = df["RiskLevel"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total patients", f"{len(df):,}")
    c2.metric("Faible risque", f"{counts.get('low risk', 0):,}")
    c3.metric("Risque moyen", f"{counts.get('mid risk', 0):,}")
    c4.metric("Risque élevé", f"{counts.get('high risk', 0):,}")

    st.markdown("---")

    col_a, col_b = st.columns([1.4, 1])

    # Distribution RiskLevel
    with col_a:
        st.markdown("#### Distribution des niveaux de risque")
        fig, ax = dark_fig(figsize=(7, 3.5))
        order = ["low risk", "mid risk", "high risk"]
        vals = [counts.get(r, 0) for r in order]
        colors = [RISK_COLORS[r] for r in order]
        bars = ax.barh(order, vals, color=colors, height=0.5, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(v + 8, bar.get_y() + bar.get_height()/2, str(v),
                    va='center', color=PALETTE["text"], fontsize=11, fontfamily='monospace')
        ax.set_xlabel("Nombre de patients", fontsize=10)
        ax.grid(axis='x', color=PALETTE["border"], linewidth=0.5, alpha=0.5)
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    # Pie chart
    with col_b:
        st.markdown("#### Répartition (%)")
        fig, ax = dark_fig(figsize=(4, 3.5))
        ax.set_facecolor(PALETTE["card"])
        wedges, texts, autotexts = ax.pie(
            vals, labels=None, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor=PALETTE["bg"], linewidth=2)
        )
        for at in autotexts:
            at.set_color(PALETTE["bg"]); at.set_fontsize(10); at.set_fontweight('bold')
        patches = [mpatches.Patch(color=colors[i], label=order[i]) for i in range(3)]
        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.08),
                  ncol=3, frameon=False, fontsize=8,
                  labelcolor=PALETTE["text"])
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Aperçu des données")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Statistiques descriptives")
    desc = df.describe().T.round(2)
    st.dataframe(desc, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: FEATURE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Feature Analysis":
    st.markdown(f"""
    <div class="header-banner">
      <p class="header-title">Analyse des Variables</p>
      <p class="header-sub">Distributions, corrélations et séparabilité par classe</p>
    </div>
    """, unsafe_allow_html=True)

    features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    risk_order = ["low risk", "mid risk", "high risk"]

    tab1, tab2, tab3 = st.tabs(["Distributions par risque", "Matrice de corrélation", "Boxplots comparatifs"])

    with tab1:
        fig, axes = dark_fig_multi(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        for i, feat in enumerate(features):
            ax = axes[i]
            for risk in risk_order:
                sub = df[df["RiskLevel"] == risk][feat]
                ax.hist(sub, bins=25, alpha=0.65, color=RISK_COLORS[risk],
                        label=risk, edgecolor="none", density=True)
            ax.set_title(feat, fontsize=11, fontweight='bold')
            ax.set_ylabel("Densité", fontsize=8)
            ax.grid(True, color=PALETTE["border"], linewidth=0.4, alpha=0.5)
            if i == 0:
                ax.legend(fontsize=7, frameon=False, labelcolor=PALETTE["text"])
        fig.tight_layout(pad=2)
        st.pyplot(fig); plt.close()

    with tab2:
        corr = df[features].corr()
        fig, ax = dark_fig(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        # Full heatmap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                    linewidths=0.5, linecolor=PALETTE["bg"],
                    annot_kws={"size": 10, "color": PALETTE["text"]},
                    ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Matrice de corrélation des variables", fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        feat_sel = st.selectbox("Choisir une variable", features)
        fig, ax = dark_fig(figsize=(8, 4.5))
        data_plot = [df[df["RiskLevel"] == r][feat_sel].values for r in risk_order]
        bp = ax.boxplot(data_plot, labels=risk_order, patch_artist=True,
                        medianprops=dict(color=PALETTE["text"], linewidth=2),
                        whiskerprops=dict(color=PALETTE["muted"]),
                        capprops=dict(color=PALETTE["muted"]),
                        flierprops=dict(markerfacecolor=PALETTE["muted"], markersize=4, linestyle='none'))
        for patch, risk in zip(bp['boxes'], risk_order):
            patch.set_facecolor(RISK_COLORS[risk])
            patch.set_alpha(0.7)
        ax.set_title(f"Distribution de {feat_sel} par niveau de risque", fontsize=12)
        ax.set_ylabel(feat_sel)
        ax.grid(True, axis='y', color=PALETTE["border"], linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Model Training":
    st.markdown(f"""
    <div class="header-banner">
      <p class="header-title">Entraînement du Modèle</p>
      <p class="header-sub">Configuration, évaluation et rapport de classification détaillé</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Paramètres")
        model_name = st.selectbox("Algorithme", [
            "Random Forest", "Gradient Boosting", "Logistic Regression",
            "SVM (RBF)", "K-Nearest Neighbors", "Decision Tree"
        ])
        test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20, 5)
        cv_folds  = st.slider("Folds de validation croisée", 3, 10, 5)

        params = {}
        if model_name == "Random Forest":
            params["n_estimators"] = st.slider("Nombre d'arbres", 50, 500, 100, 50)
            params["max_depth"]    = st.selectbox("Profondeur max", [None, 3, 5, 10, 15])
        elif model_name == "Gradient Boosting":
            params["n_estimators"]  = st.slider("Nombre d'arbres", 50, 300, 100, 50)
            params["learning_rate"] = st.select_slider("Taux d'apprentissage", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
        elif model_name == "K-Nearest Neighbors":
            params["n_neighbors"] = st.slider("Nombre de voisins (k)", 1, 20, 5)

        run = st.button("Lancer l'entraînement")

    with col2:
        if run:
            # Re-split with chosen test_size
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y)
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            @st.cache_data
            def fit_model(name, params_str, X_tr_sc, y_tr):
                p = eval(params_str)
                models = {
                    "Random Forest":      RandomForestClassifier(random_state=42, **{k: v for k, v in p.items() if k in ['n_estimators','max_depth']}),
                    "Gradient Boosting":  GradientBoostingClassifier(random_state=42, **{k: v for k, v in p.items() if k in ['n_estimators','learning_rate']}),
                    "Logistic Regression":LogisticRegression(max_iter=1000, random_state=42),
                    "SVM (RBF)":          SVC(kernel='rbf', probability=True, random_state=42),
                    "K-Nearest Neighbors":KNeighborsClassifier(**{k: v for k, v in p.items() if k == 'n_neighbors'}),
                    "Decision Tree":      DecisionTreeClassifier(random_state=42),
                }
                clf = models[name]
                clf.fit(X_tr_sc, y_tr)
                return clf

            with st.spinner("Entraînement en cours..."):
                clf = fit_model(model_name, str(params), X_tr_sc, y_tr)

            y_pred = clf.predict(X_te_sc)
            acc    = accuracy_score(y_te, y_pred)
            f1     = f1_score(y_te, y_pred, average='weighted')
            prec   = precision_score(y_te, y_pred, average='weighted')
            rec    = recall_score(y_te, y_pred, average='weighted')

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, X_tr_sc, y_tr, cv=cv, scoring='accuracy')

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("F1-score", f"{f1:.3f}")
            m3.metric("Précision", f"{prec:.3f}")
            m4.metric("Rappel", f"{rec:.3f}")

            st.markdown(f"""
            <div class='card' style='margin-top:1rem'>
              <span style='color:{PALETTE["muted"]};font-size:0.8rem'>Validation croisée ({cv_folds} folds)</span><br>
              <span style='font-family:DM Mono,monospace;color:{PALETTE["accent2"]};font-size:1.2rem'>
                {cv_scores.mean():.3f} ± {cv_scores.std():.3f}
              </span>
            </div>
            """, unsafe_allow_html=True)

            # Confusion matrix
            st.markdown("#### Matrice de confusion")
            cm = confusion_matrix(y_te, y_pred)
            fig, ax = dark_fig(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes,
                        ax=ax, linewidths=0.5, linecolor=PALETTE["bg"],
                        annot_kws={"size": 13, "weight": "bold"})
            ax.set_xlabel("Prédit", fontsize=10)
            ax.set_ylabel("Réel", fontsize=10)
            ax.set_title("Matrice de confusion", fontsize=12)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

            # Classification report
            st.markdown("#### Rapport de classification")
            report = classification_report(y_te, y_pred, target_names=classes, output_dict=True)
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df, use_container_width=True)

            # Feature importance (if available)
            if hasattr(clf, 'feature_importances_'):
                st.markdown("#### Importance des variables")
                fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
                fig, ax = dark_fig(figsize=(7, 3.5))
                bars = ax.barh(fi.index, fi.values, color=PALETTE["accent"], alpha=0.85, edgecolor="none")
                for bar, v in zip(bars, fi.values):
                    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
                            va='center', color=PALETTE["text"], fontsize=9)
                ax.set_xlabel("Importance (Gini)", fontsize=10)
                ax.grid(axis='x', color=PALETTE["border"], linewidth=0.4, alpha=0.5)
                fig.tight_layout()
                st.pyplot(fig); plt.close()
        else:
            st.info("Configurez les paramètres et cliquez sur **Lancer l'entraînement**.")


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.markdown(f"""
    <div class="header-banner">
      <p class="header-title">Comparaison des Modèles</p>
      <p class="header-sub">Benchmark automatique de tous les algorithmes supervisés</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Lancer le benchmark complet"):
        all_models = {
            "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
            "SVM (RBF)":            SVC(kernel='rbf', probability=True, random_state=42),
            "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=5),
            "Decision Tree":        DecisionTreeClassifier(random_state=42),
        }

        results = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        progress = st.progress(0, text="Entraînement des modèles...")
        for idx, (name, clf) in enumerate(all_models.items()):
            clf.fit(X_train_sc, y_train)
            y_pred = clf.predict(X_test_sc)
            cv_sc = cross_val_score(clf, X_train_sc, y_train, cv=cv, scoring='accuracy')
            results.append({
                "Modèle":       name,
                "Accuracy":     round(accuracy_score(y_test, y_pred), 4),
                "F1 (weighted)":round(f1_score(y_test, y_pred, average='weighted'), 4),
                "CV Mean":      round(cv_sc.mean(), 4),
                "CV Std":       round(cv_sc.std(), 4),
                "Précision":    round(precision_score(y_test, y_pred, average='weighted'), 4),
                "Rappel":       round(recall_score(y_test, y_pred, average='weighted'), 4),
            })
            progress.progress((idx+1)/len(all_models), text=f"Modèle {idx+1}/{len(all_models)} traité")

        progress.empty()
        res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)

        st.markdown("#### Tableau récapitulatif")
        st.dataframe(res_df, use_container_width=True)

        # Bar chart accuracy
        st.markdown("#### Accuracy comparée")
        fig, ax = dark_fig(figsize=(10, 4))
        colors_bar = [PALETTE["accent"] if i == 0 else PALETTE["muted"] for i in range(len(res_df))]
        bars = ax.bar(res_df["Modèle"], res_df["Accuracy"], color=colors_bar, edgecolor="none", width=0.6)
        for bar, v in zip(bars, res_df["Accuracy"]):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
                    ha='center', va='bottom', color=PALETTE["text"], fontsize=9, fontfamily='monospace')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title("Performance sur le jeu de test", fontsize=12)
        ax.grid(True, axis='y', color=PALETTE["border"], linewidth=0.4, alpha=0.5)
        plt.xticks(rotation=20, ha='right', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        # CV comparison
        st.markdown("#### Validation croisée (mean ± std)")
        fig, ax = dark_fig(figsize=(10, 4))
        x = np.arange(len(res_df))
        ax.bar(x, res_df["CV Mean"], color=PALETTE["accent2"], alpha=0.7, width=0.5, edgecolor="none")
        ax.errorbar(x, res_df["CV Mean"], yerr=res_df["CV Std"],
                    fmt='none', color=PALETTE["accent4"], capsize=5, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(res_df["Modèle"], rotation=20, ha='right', fontsize=9)
        ax.set_ylabel("CV Accuracy", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', color=PALETTE["border"], linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        # Best model callout
        best = res_df.iloc[0]
        st.markdown(f"""
        <div class='card' style='border-color:{PALETTE["accent2"]}'>
          <p style='color:{PALETTE["muted"]};font-size:0.8rem;margin:0'>Meilleur modèle</p>
          <p style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;
                    color:{PALETTE["accent2"]};margin:0'>{best["Modèle"]}</p>
          <p style='font-family:DM Mono,monospace;color:{PALETTE["text"]};font-size:0.9rem;margin:0'>
            Accuracy {best["Accuracy"]:.4f} · F1 {best["F1 (weighted)"]:.4f} · CV {best["CV Mean"]:.4f} ± {best["CV Std"]:.4f}
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Cliquez sur **Lancer le benchmark complet** pour comparer tous les modèles.")


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: PREDICTION TOOL
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Prediction Tool":
    st.markdown(f"""
    <div class="header-banner">
      <p class="header-title">Outil de Prédiction Clinique</p>
      <p class="header-sub">Saisir les paramètres patient pour estimer le niveau de risque</p>
    </div>
    """, unsafe_allow_html=True)

    # Train the best model (RF by default)
    @st.cache_resource
    def get_predictor():
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train_sc, y_train)
        return clf

    clf = get_predictor()

    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        st.markdown("#### Paramètres cliniques")
        age_i    = st.slider("Âge (ans)",            10, 70, 28)
        sys_i    = st.slider("Pression systolique",  70, 160, 115, 1)
        dia_i    = st.slider("Pression diastolique", 49, 100, 76, 1)
        bs_i     = st.slider("Glycémie (mmol/L)",    6.0, 19.0, 7.5, 0.1)
        bt_i     = st.slider("Température (°F)",     98.0, 103.0, 98.6, 0.1)
        hr_i     = st.slider("Fréquence cardiaque",  60, 90, 76, 1)
        predict_btn = st.button("Prédire le niveau de risque")

    with col_out:
        if predict_btn:
            sample = np.array([[age_i, sys_i, dia_i, bs_i, bt_i, hr_i]])
            sample_sc = scaler.transform(sample)
            pred_enc  = clf.predict(sample_sc)[0]
            probas    = clf.predict_proba(sample_sc)[0]
            pred_label = le.inverse_transform([pred_enc])[0]

            badge_map = {
                "low risk":  ("badge-low",  "FAIBLE RISQUE"),
                "mid risk":  ("badge-mid",  "RISQUE MODÉRÉ"),
                "high risk": ("badge-high", "RISQUE ÉLEVÉ"),
            }
            badge_cls, badge_txt = badge_map[pred_label]
            color = RISK_COLORS[pred_label]

            st.markdown(f"""
            <div class='card' style='border-color:{color};text-align:center;padding:2rem'>
              <p style='color:{PALETTE["muted"]};font-size:0.8rem;margin-bottom:0.5rem'>Prédiction</p>
              <span class='{badge_cls}' style='font-size:1rem'>{badge_txt}</span>
              <p style='font-family:DM Mono,monospace;font-size:2.5rem;font-weight:700;
                        color:{color};margin:0.5rem 0'>{pred_label.upper()}</p>
              <p style='color:{PALETTE["muted"]};font-size:0.8rem'>Confiance: <b style='color:{PALETTE["text"]}'>{probas.max()*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities
            st.markdown("#### Probabilités par classe")
            fig, ax = dark_fig(figsize=(7, 3))
            cls_labels = [le.inverse_transform([i])[0] for i in range(len(classes))]
            bar_colors = [RISK_COLORS.get(c, PALETTE["accent"]) for c in cls_labels]
            bars = ax.barh(cls_labels, probas, color=bar_colors, edgecolor="none", height=0.45)
            for bar, v in zip(bars, probas):
                ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f"{v*100:.1f}%",
                        va='center', color=PALETTE["text"], fontsize=11, fontfamily='monospace')
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("Probabilité", fontsize=10)
            ax.grid(axis='x', color=PALETTE["border"], linewidth=0.4, alpha=0.5)
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig); plt.close()

            # Résumé des seuils cliniques
            st.markdown("#### Analyse des indicateurs")
            flags = []
            if sys_i > 140: flags.append(("SystolicBP", f"{sys_i} mmHg", "Hypertension sévère", PALETTE["high"]))
            elif sys_i > 130: flags.append(("SystolicBP", f"{sys_i} mmHg", "Hypertension légère", PALETTE["mid"]))
            if dia_i > 90: flags.append(("DiastolicBP", f"{dia_i} mmHg", "Diastolique élevée", PALETTE["high"]))
            if bs_i > 11: flags.append(("BS", f"{bs_i} mmol/L", "Hyperglycémie sévère", PALETTE["high"]))
            elif bs_i > 9: flags.append(("BS", f"{bs_i} mmol/L", "Glycémie élevée", PALETTE["mid"]))
            if bt_i > 100: flags.append(("BodyTemp", f"{bt_i}°F", "Fièvre", PALETTE["mid"]))
            if age_i > 40: flags.append(("Âge", f"{age_i} ans", "Grossesse tardive", PALETTE["mid"]))
            if age_i < 18: flags.append(("Âge", f"{age_i} ans", "Grossesse adolescente", PALETTE["mid"]))

            if flags:
                for feat, val, msg, col in flags:
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:10px;
                                padding:6px 12px;border-left:3px solid {col};
                                margin-bottom:6px;background:{PALETTE["card"]};border-radius:4px'>
                      <span style='font-family:DM Mono,monospace;color:{col};min-width:110px'>{feat}</span>
                      <span style='color:{PALETTE["muted"]};font-size:0.85rem'>{val} — {msg}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:{PALETTE['low']}'>Tous les indicateurs sont dans les normes.</p>",
                            unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='card' style='text-align:center;padding:3rem'>
              <p style='color:{PALETTE["muted"]};font-size:0.95rem'>
                Ajustez les paramètres cliniques à gauche<br>puis cliquez sur <b style='color:{PALETTE["text"]}'>Prédire</b>.
              </p>
            </div>
            """, unsafe_allow_html=True)