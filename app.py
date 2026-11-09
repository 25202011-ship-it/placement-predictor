"""
HR Job Placement Predictor — Streamlit web app
Uses the outputs of:
  • SAS Studio  (PROC GLMSELECT) — regression on expected_ctc_lpa, 7 selection methods
  • SAS Viya    (Model Studio)   — classification on status, champion = Gradient Boosting
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ───────────────────────── EDIT THESE ─────────────────────────
PROJECT_TITLE = "HR Job Placement Predictor"
GROUP_NO = "Group __"
TEAM = ["Member 1", "Member 2", "Member 3", "Member 4", "Member 5"]
COURSE = "Predictive Analytics using SAS — KSOM, KIIT"
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title=PROJECT_TITLE, page_icon="🎯", layout="wide")

BLUE, ORANGE, GREY, GREEN, RED = "#1f4e9c", "#f08a24", "#9aa5b1", "#2e9d5b", "#c8423b"

# ── SAS STUDIO: regression results (PROC GLMSELECT, 70/30 split, seed 2026) ──
REG_RESULTS = pd.DataFrame({
    "Method": ["Forward", "Backward", "Stepwise", "LASSO", "Adaptive LASSO", "Elastic Net", "LAR"],
    "Effects selected": ["Intercept"] * 7,
    "Test RMSE": [3.1157] * 7,
    "Test R²": [-0.0001] * 7,
    "Test MAPE (%)": [41.21] * 7,
})
REG_INTERCEPT = 8.017643   # Parameter estimate, selected model
REG_RMSE = 3.1157

# ── SAS VIYA: classification results (Model Studio, 70/30 split) ──
GB_METRICS = {"AUC": 0.9658, "Accuracy": 0.8996, "F1 score": 0.8318,
              "Misclassification": 0.1004, "KS (Youden)": 0.7968}
GB_TRAIN_AUC = 0.9741
CLS_KS = pd.DataFrame({
    "Model": ["Gradient Boosting", "Random Forest", "Logistic Regression",
              "Decision Tree (full)", "Decision Tree (pruned)"],
    "KS (Youden)": [0.797, 0.745, 0.72, 0.53, 0.53],
})
VAR_IMPORTANCE = pd.DataFrame({
    "Variable": ["Technical score", "Years of experience", "Skills match %", "Expected CTC",
                 "Job role match", "Internship experience", "Communication score",
                 "Previous CTC", "Layoff history", "Relevant experience",
                 "Employment gap (months)", "Competition level", "Bond requirement", "Age",
                 "Career switch willingness", "Relocation willingness", "Notice period",
                 "Aptitude score", "Degree specialization", "SSC %", "Degree %", "HSC %",
                 "Certifications count"],
    "Relative importance": [1.00, 0.60, 0.34, 0.20, 0.20, 0.125, 0.115, 0.105, 0.085, 0.075,
                            0.048, 0.043, 0.035, 0.025, 0.023, 0.022, 0.022, 0.017,
                            0, 0, 0, 0, 0],
})

NUM = ["age_years", "ssc_percentage", "hsc_percentage", "degree_percentage", "technical_score",
       "aptitude_score", "communication_score", "skills_match_percentage", "certifications_count",
       "years_of_experience", "previous_ctc_lpa", "expected_ctc_lpa", "notice_period_days",
       "employment_gap_months"]
CAT = ["degree_specialization", "internship_experience", "career_switch_willingness",
       "relevant_experience", "job_role_match", "competition_level", "bond_requirement",
       "layoff_history", "relocation_willingness"]
INPUTS = NUM + CAT   # the 23 inputs used by the SAS Viya champion


@st.cache_data
def load_data():
    return pd.read_csv("placement_data.csv")


@st.cache_resource
def train_model(df):
    """Gradient boosting replica of the SAS Viya champion, same 70/30 split idea."""
    X, y = df[INPUTS], (df["status"] == "Placed").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=2026, stratify=y)
    pre = ColumnTransformer([("cat", OrdinalEncoder(), CAT)], remainder="passthrough")
    n_cat = len(CAT)
    model = Pipeline([
        ("prep", pre),
        ("gb", HistGradientBoostingClassifier(
            categorical_features=list(range(n_cat)), max_iter=300,
            learning_rate=0.08, random_state=2026)),
    ])
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    check = {"AUC": roc_auc_score(y_te, p),
             "Accuracy": accuracy_score(y_te, p >= 0.5),
             "F1 score": f1_score(y_te, p >= 0.5)}
    return model, check


df = load_data()
model, py_check = train_model(df)
placement_rate = (df["status"] == "Placed").mean()

# ───────────────────────── HEADER ─────────────────────────
st.title(f"🎯 {PROJECT_TITLE}")
st.caption(f"{GROUP_NO} · {COURSE} · Models built in SAS Studio and SAS Viya · "
           f"{len(df):,} candidate records")

tab_pred, tab_perf, tab_ins, tab_data, tab_flow = st.tabs(
    ["🔮 Predict", "📊 Model performance", "💡 Insights", "🗂️ Data", "🧭 Workflow & team"])

# ───────────────────────── PREDICT ─────────────────────────
with tab_pred:
    st.subheader("Enter a candidate profile")
    med = df[NUM].median()

    with st.form("profile"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Skills & assessment**")
            technical_score = st.slider("Technical score", 40.0, 100.0, float(round(med.technical_score)), 0.5)
            aptitude_score = st.slider("Aptitude score", 40.0, 100.0, float(round(med.aptitude_score)), 0.5)
            communication_score = st.slider("Communication score", 45.0, 100.0, float(round(med.communication_score)), 0.5)
            skills_match_percentage = st.slider("Skills match %", 45.0, 100.0, float(round(med.skills_match_percentage)), 0.5)
            certifications_count = st.number_input("Certifications", 0, 8, 1)
        with c2:
            st.markdown("**Education & experience**")
            ssc_percentage = st.slider("SSC %", 50.0, 95.0, 70.0, 0.5)
            hsc_percentage = st.slider("HSC %", 50.0, 95.0, 72.0, 0.5)
            degree_percentage = st.slider("Degree %", 55.0, 95.0, 74.0, 0.5)
            degree_specialization = st.selectbox("Degree specialization", sorted(df.degree_specialization.unique()))
            years_of_experience = st.number_input("Years of experience", 0, 5, 1)
            internship_experience = st.radio("Internship experience", ["Yes", "No"], horizontal=True)
            relevant_experience = st.radio("Experience relevant to role?", ["Relevant", "Not Relevant"], horizontal=True)
        with c3:
            st.markdown("**Job & personal factors**")
            age_years = st.number_input("Age (years)", 21, 34, 27)
            previous_ctc_lpa = st.number_input("Previous CTC (LPA)", 1.0, 14.0, 4.8, 0.1)
            expected_ctc_lpa = st.number_input("Expected CTC (LPA)", 3.0, 25.0, 7.9, 0.1)
            job_role_match = st.radio("Job role match", ["Matched", "Not Matched"], horizontal=True)
            competition_level = st.selectbox("Competition level", ["Low", "Medium", "High"], index=1)
            notice_period_days = st.select_slider("Notice period (days)", [0, 15, 30, 60, 90], 15)
            employment_gap_months = st.slider("Employment gap (months)", 0, 18, 0)
            layoff_history = st.radio("Layoff history", ["No", "Yes"], horizontal=True)
            bond_requirement = st.radio("Bond required", ["Not Required", "Required"], horizontal=True)
            career_switch_willingness = st.radio("Willing to switch career", ["Willing", "Not Willing"], horizontal=True)
            relocation_willingness = st.radio("Willing to relocate", ["Willing", "Not Willing"], horizontal=True)
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    row = pd.DataFrame([{k: v for k, v in locals().items() if k in INPUTS}])[INPUTS]
    prob = float(model.predict_proba(row)[0, 1])

    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### Classification — will this candidate be placed?")
        verdict = "Placed" if prob >= 0.5 else "Not Placed"
        colour = GREEN if prob >= 0.5 else RED
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": f"Prediction: <b style='color:{colour}'>{verdict}</b>"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": colour},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                   "steps": [{"range": [0, 50], "color": "#f6e3e1"},
                             {"range": [50, 100], "color": "#e1f3e8"}]}))
        gauge.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, width="stretch")
        st.caption(f"Model: Gradient Boosting (SAS Viya champion). Cut-off 50%. "
                   f"Average placement rate in the data: {placement_rate:.0%}.")
    with r2:
        st.markdown("#### Regression — expected CTC benchmark")
        diff = expected_ctc_lpa - REG_INTERCEPT
        st.metric("Predicted expected CTC (SAS Studio model)", f"₹ {REG_INTERCEPT:.2f} LPA")
        st.caption(f"This candidate asks ₹ {expected_ctc_lpa:.2f} LPA, which is "
                   f"{abs(diff):.2f} LPA {'above' if diff >= 0 else 'below'} the benchmark.")
        st.info(f"Typical range (±1 RMSE): **₹ {REG_INTERCEPT - REG_RMSE:.2f} – "
                f"{REG_INTERCEPT + REG_RMSE:.2f} LPA**.\n\n"
                "All 7 selection methods kept only the intercept: no profile variable explains "
                "expected CTC, so the best prediction is the market average. Salary expectations "
                "in this data are personal, not driven by skills or academics.")

    st.markdown("#### What-if: how would the placement chance change?")
    w1, w2 = st.columns(2)

    def what_if(var, values):
        rows = pd.concat([row] * len(values), ignore_index=True)
        rows[var] = values
        return model.predict_proba(rows)[:, 1] * 100

    with w1:
        xs = list(range(40, 101, 2))
        fig = px.line(x=xs, y=what_if("technical_score", xs),
                      labels={"x": "Technical score", "y": "Placement chance (%)"})
        fig.add_vline(x=technical_score, line_dash="dot", line_color=GREY)
        fig.update_traces(line_color=BLUE, line_width=3)
        fig.update_layout(height=300, yaxis_range=[0, 100], margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    with w2:
        xs = [round(3 + 0.5 * i, 1) for i in range(43)]
        fig = px.line(x=xs, y=what_if("expected_ctc_lpa", xs),
                      labels={"x": "Expected CTC (LPA)", "y": "Placement chance (%)"})
        fig.add_vline(x=expected_ctc_lpa, line_dash="dot", line_color=GREY)
        fig.update_traces(line_color=ORANGE, line_width=3)
        fig.update_layout(height=300, yaxis_range=[0, 100], margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    st.caption("Dotted line = the value entered above. Other inputs are held constant.")

# ───────────────────────── PERFORMANCE ─────────────────────────
with tab_perf:
    st.subheader("Classification — SAS Viya Model Studio (target: status)")
    cols = st.columns(5)
    for c, (k, v) in zip(cols, GB_METRICS.items()):
        c.metric(k, f"{v:.1%}" if k in ("Accuracy", "Misclassification") else f"{v:.3f}")
    st.caption(f"Champion: Gradient Boosting, holdout (30%) partition, 11,701 rows. "
               f"Training AUC {GB_TRAIN_AUC:.3f} vs holdout {GB_METRICS['AUC']:.3f} → no overfitting.")

    fig = px.bar(CLS_KS.sort_values("KS (Youden)"), x="KS (Youden)", y="Model", orientation="h",
                 text="KS (Youden)", color="Model",
                 category_orders={"Model": CLS_KS.sort_values("KS (Youden)", ascending=False).Model.tolist()},
                 color_discrete_map={m: (ORANGE if m == "Gradient Boosting" else GREY) for m in CLS_KS.Model})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False, height=320, xaxis_range=[0, 1], margin=dict(t=30),
                      title="Model comparison (KS, Youden) — selection criterion in SAS Viya")
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Regression — SAS Studio PROC GLMSELECT (target: expected_ctc_lpa)")
    st.dataframe(REG_RESULTS, hide_index=True, width="stretch")
    st.markdown(
        f"- 70/30 split (27,321 train / 11,684 test), random seed 2026.\n"
        f"- Every method stopped at **Step 0 (intercept only)**: Test RMSE **{REG_RMSE}**, "
        f"R² ≈ **0**, MAPE **41.2%**. The methods tie, so the simplest model (intercept = "
        f"₹ {REG_INTERCEPT:.2f} LPA) is used as the benchmark.\n"
        "- Interpretation: none of the 24 profile variables carries information about expected CTC.")

    with st.expander("Python replica check (used inside this app)"):
        st.write("The SAS Viya gradient boosting model cannot be copied as a formula, so the app "
                 "re-trains the same type of model in Python on the same data and a 70/30 split. "
                 "Its holdout results are close to SAS Viya:")
        st.dataframe(pd.DataFrame({"Metric": list(py_check),
                                   "Python replica": [round(v, 4) for v in py_check.values()],
                                   "SAS Viya": [GB_METRICS[k] for k in py_check]}),
                     hide_index=True)

# ───────────────────────── INSIGHTS ─────────────────────────
with tab_ins:
    st.subheader("What drives placement? (SAS Viya variable importance)")
    vi = VAR_IMPORTANCE.sort_values("Relative importance")
    fig = px.bar(vi, x="Relative importance", y="Variable", orientation="h")
    fig.update_traces(marker_color=BLUE)
    fig.update_layout(height=620, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        bands = pd.cut(df.technical_score, [39, 50, 60, 70, 80, 90, 100],
                       labels=["40–50", "50–60", "60–70", "70–80", "80–90", "90–100"])
        t = df.assign(band=bands).groupby("band", observed=True)["status"].apply(
            lambda s: (s == "Placed").mean() * 100).reset_index(name="Placement rate (%)")
        fig = px.bar(t, x="band", y="Placement rate (%)", labels={"band": "Technical score"},
                     title="Placement rate by technical score")
        fig.update_traces(marker_color=BLUE)
        st.plotly_chart(fig, width="stretch")
    with b:
        t = df.groupby("years_of_experience")["status"].apply(
            lambda s: (s == "Placed").mean() * 100).reset_index(name="Placement rate (%)")
        fig = px.bar(t, x="years_of_experience", y="Placement rate (%)",
                     labels={"years_of_experience": "Years of experience"},
                     title="Placement rate by years of experience")
        fig.update_traces(marker_color=ORANGE)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Business recommendations")
    st.markdown(
        "1. **Screen on skills, not marks.** Technical score, experience and skills match drive "
        "placement; SSC, HSC and degree % have almost no effect.\n"
        "2. **Training ROI.** Raising a candidate's technical score is the single biggest lever "
        "for placement teams and training institutes.\n"
        "3. **Role fit matters.** Matching candidates to roles aligned with their background lifts "
        "placement chances.\n"
        "4. **Salary negotiation.** Expected CTC cannot be predicted from the profile, so HR should "
        "benchmark offers against the market average (~₹ 8 LPA) rather than candidate attributes.\n"
        "5. **Use the score for prioritisation.** Recruiters can rank applicants by predicted "
        "placement probability to focus interview effort.")

# ───────────────────────── DATA ─────────────────────────
with tab_data:
    st.subheader("Dataset overview")
    m = st.columns(4)
    m[0].metric("Records", f"{len(df):,}")
    m[1].metric("Model inputs", len(INPUTS))
    m[2].metric("Placed", f"{placement_rate:.1%}")
    m[3].metric("Missing values", int(df.isna().sum().sum()))
    st.dataframe(df[NUM].describe().T.round(2), width="stretch")
    st.markdown("**Sample rows**")
    st.dataframe(df.head(20), width="stretch")

# ───────────────────────── WORKFLOW & TEAM ─────────────────────────
with tab_flow:
    st.subheader("Analytics workflow and web-app architecture")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
      raw   [label="Raw data\\nHR_Job_Placement.xlsx\\n39,005 rows", fillcolor="#e8eef8"];
      prep  [label="Cleaning & feature\\nengineering", fillcolor="#e8eef8"];
      split [label="70 / 30 split\\ntrain / test", fillcolor="#e8eef8"];
      reg   [label="SAS Studio\\nPROC GLMSELECT\\nFwd · Bwd · Stepwise\\nLASSO · Adaptive · EN · LAR", fillcolor="#fff1e3"];
      cls   [label="SAS Viya Model Studio\\nFull & Pruned Tree · GB\\nForest · Logistic", fillcolor="#fff1e3"];
      regb  [label="Best regression\\n(intercept, RMSE 3.12)", fillcolor="#fde2c4"];
      clsb  [label="Champion: Gradient\\nBoosting (AUC 0.966)", fillcolor="#fde2c4"];
      gh    [label="GitHub repo\\napp.py · data · requirements", fillcolor="#e3f3e9"];
      app   [label="Streamlit Cloud\\nweb app", fillcolor="#e3f3e9"];
      user  [label="User enters\\nprofile → live\\npredictions", shape=ellipse, fillcolor="#ffffff"];
      raw -> prep -> split; split -> reg -> regb; split -> cls -> clsb;
      regb -> gh; clsb -> gh; gh -> app -> user;
    }""", width="stretch")

    st.subheader("Team")
    st.markdown(f"**{GROUP_NO}** — " + " · ".join(TEAM))
    st.markdown(
        "**Documented assumptions**\n"
        "- Derived columns (avg_academic, avg_interview, ctc_diff, bands, status_enc) were excluded "
        "from modelling to avoid duplication and target leakage.\n"
        "- The 30% holdout in SAS Viya is labelled *VALIDATE*; it was used for assessment and "
        "champion selection.\n"
        "- Probability cut-off for 'Placed' = 0.50.")
