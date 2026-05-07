import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from model import train_model, predict_patient, explain_case, get_model_stats

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="StrokeAI — Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Base */
*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main card */
.main-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
}

/* Header hero */
.hero {
    background: linear-gradient(135deg, #1a1f3a 0%, #0f2744 50%, #1a1f3a 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99,179,237,0.05) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}

.hero h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #63b3ed, #90cdf4, #4299e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
}

/* Section labels */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Input styling */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
    border-radius: 10px !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 2.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 4px 20px rgba(49,130,206,0.4) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(49,130,206,0.6) !important;
}

/* Result cards */
.result-critical {
    background: linear-gradient(135deg, rgba(197,48,48,0.2), rgba(197,48,48,0.05));
    border: 2px solid rgba(197,48,48,0.6);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.result-safe {
    background: linear-gradient(135deg, rgba(47,133,90,0.2), rgba(47,133,90,0.05));
    border: 2px solid rgba(47,133,90,0.6);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.result-moderate {
    background: linear-gradient(135deg, rgba(214,158,46,0.2), rgba(214,158,46,0.05));
    border: 2px solid rgba(214,158,46,0.6);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

/* Risk factor badge */
.risk-badge-critical { background: rgba(197,48,48,0.15); border-left: 3px solid #c53030; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.risk-badge-high { background: rgba(221,107,32,0.15); border-left: 3px solid #dd6b20; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.risk-badge-moderate { background: rgba(214,158,46,0.15); border-left: 3px solid #d69e2e; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.risk-badge-low { background: rgba(56,178,172,0.15); border-left: 3px solid #38b2ac; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; }

/* Metric card */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #63b3ed;
}

.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

/* Streamlit elements dark mode fixes */
[data-testid="stMarkdownContainer"] p { color: #cbd5e0; }
label { color: #a0aec0 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD & TRAIN MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    df = pd.read_csv(
        r"C:\Users\Dell\OneDrive - October 6 University Egypt\Desktop\stroke-ai-system\healthcare-dataset-stroke-data.csv"
    )
    train_model(df)
    return get_model_stats()

with st.spinner("🧠 Loading AI Model..."):
    stats = load_model()


# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🧠 StrokeAI Prediction System</h1>
    <p>Expert System with Neural Network • Medical Decision Support</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL STATS BAR
# ============================================================
if stats:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['accuracy']*100:.1f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['roc_auc']:.3f}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['stroke_precision']*100:.1f}%</div>
            <div class="metric-label">Precision</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['stroke_recall']*100:.1f}%</div>
            <div class="metric-label">Recall</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['stroke_f1']*100:.1f}%</div>
            <div class="metric-label">F1-Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# INPUT FORM
# ============================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 1, 100, 55)
    hypertension = st.selectbox("Hypertension", [0, 1],
                                 format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No")
    married = st.selectbox("Ever Married", ["Yes", "No"])

with col_r:
    work_type = st.selectbox("Work Type",
                              ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    glucose = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0,
                               max_value=400.0, value=100.0, step=0.5)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    smoking = st.selectbox("Smoking Status",
                            ["never smoked", "smokes", "formerly smoked", "Unknown"])

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PREDICT BUTTON
# ============================================================
predict_btn = st.button("Analyze Stroke Risk", key="predict_btn")


# ============================================================
# RESULTS
# ============================================================
if predict_btn:
    patient = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking
    }

    with st.spinner("Running neural network analysis..."):
        pred, prob, threshold = predict_patient(patient)
        reasons, overall_risk, risk_score = explain_case(patient)

    st.markdown("---")
    st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

    # --- Result Banner (Based on Neural Network Probability ONLY) ---
    prob_pct = prob * 100

    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        if prob_pct >= 60:
            card_class = "result-critical"
            emoji = "🚨"
            title = "HIGH STROKE RISK"
            color = "#fc8181"
        elif prob_pct >= threshold * 100:
            card_class = "result-moderate"
            emoji = "⚠️"
            title = "MODERATE STROKE RISK"
            color = "#f6e05e"
        else:
            card_class = "result-safe"
            emoji = "✅"
            title = "LOW STROKE RISK"
            color = "#68d391"

        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size:3rem">{emoji}</div>
            <div style="font-size:1.6rem; font-weight:800; color:{color}; margin: 0.5rem 0;">{title}</div>
            <div style="font-size:2.5rem; font-weight:800; color:white;">{prob_pct:.1f}%</div>
            <div style="color:#94a3b8; font-size:0.9rem;">Stroke Probability</div>
            <div style="margin-top:1rem; color:#94a3b8; font-size:0.8rem;">
                Threshold: {threshold:.2f} | Risk Score: {risk_score}/14
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Stroke Risk %", 'font': {'size': 16, 'color': '#94a3b8'}},
            number={'suffix': "%", 'font': {'size': 36, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#4a5568', 'tickfont': {'color': '#94a3b8'}},
                'bar': {'color': color if pred == 1 else "#68d391"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(72,187,120,0.15)'},
                    {'range': [30, 60], 'color': 'rgba(246,224,94,0.15)'},
                    {'range': [60, 100], 'color': 'rgba(252,129,129,0.15)'},
                ],
                'threshold': {
                    'line': {'color': "#63b3ed", 'width': 3},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter'},
            height=250,
            margin=dict(t=30, b=0, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Expert System Explanation ---
    st.markdown("---")
    st.markdown('<div class="section-label">🔬 Expert System Analysis — Risk Factors</div>',
                unsafe_allow_html=True)

    if reasons:
        severity_icons = {
            "critical": ("🔴", "Critical"),
            "high": ("🟠", "High"),
            "moderate": ("🟡", "Moderate"),
            "low": ("🟢", "Low")
        }

        for r in reasons:
            sev = r['severity']
            icon, label = severity_icons.get(sev, ("⚪", "Info"))
            st.markdown(f"""
            <div class="risk-badge-{sev}">
                <strong>{icon} {r['factor']}</strong>
                <span style="float:right; font-size:0.75rem; color:#94a3b8; background:rgba(255,255,255,0.07);
                       padding:2px 8px; border-radius:10px;">{label}</span>
                <br><span style="color:#a0aec0; font-size:0.85rem;">{r['detail']}</span>
            </div>
            """, unsafe_allow_html=True)

        # Risk factor bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        factor_names = [r['factor'].split('(')[0].strip() for r in reasons]
        sev_scores = {"critical": 4, "high": 3, "moderate": 2, "low": 1}
        factor_scores = [sev_scores.get(r['severity'], 1) for r in reasons]
        colors_map = {"critical": "#fc8181", "high": "#f6ad55", "moderate": "#f6e05e", "low": "#68d391"}
        bar_colors = [colors_map.get(r['severity'], "#63b3ed") for r in reasons]

        fig2 = go.Figure(go.Bar(
            x=factor_scores,
            y=factor_names,
            orientation='h',
            marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,0)', width=0)),
            text=[r['severity'].upper() for r in reasons],
            textposition='inside',
            textfont=dict(color='white', size=11, family='Inter')
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(color='#a0aec0', size=12)),
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(150, len(reasons) * 55),
            font=dict(family='Inter')
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.markdown("""
        <div class="risk-badge-low">
            <strong>✅ No significant risk factors identified</strong><br>
            <span style="color:#a0aec0; font-size:0.85rem;">
                The patient's parameters are within normal ranges. Continue regular health monitoring.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # --- Medical Recommendations ---
    st.markdown("---")
    st.markdown('<div class="section-label">💊 Medical Recommendations</div>', unsafe_allow_html=True)

    cols_rec = st.columns(3)
    recs = []

    if pred == 1 or prob_pct >= 30:
        recs = [
            ("🏥", "Consult a Neurologist", "Schedule an immediate appointment with a specialist."),
            ("💊", "Review Medications", "Ensure blood pressure and diabetes medications are optimized."),
            ("🩺", "Regular Monitoring", "Monitor blood pressure daily, glucose weekly."),
            ("🥗", "Lifestyle Changes", "Mediterranean diet, reduce sodium, alcohol, and processed foods."),
            ("🏃", "Physical Activity", "30 min moderate exercise 5× per week (if approved by doctor)."),
            ("🚭", "Quit Smoking", "Cessation programs significantly reduce stroke risk within 2–5 years."),
        ]
    else:
        recs = [
            ("✅", "Maintain Healthy Lifestyle", "Continue your current healthy habits."),
            ("🥗", "Balanced Diet", "Eat a diet rich in fruits, vegetables, and whole grains."),
            ("🏃", "Stay Active", "Maintain regular physical activity for cardiovascular health."),
            ("🩺", "Annual Check-ups", "Regular health screenings are key to early detection."),
            ("😴", "Quality Sleep", "7–9 hours of sleep reduces cardiovascular risk."),
            ("🧘", "Stress Management", "Chronic stress elevates blood pressure. Practice mindfulness."),
        ]

    for i, (icon, title_r, desc) in enumerate(recs):
        with cols_rec[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; margin-bottom:0.8rem; padding:1rem;">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-weight:700; color:#e2e8f0; margin: 0.4rem 0; font-size:0.9rem;">{title_r}</div>
                <div style="color:#718096; font-size:0.8rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 1rem 0;">
     <strong style="color:#718096">Disclaimer:</strong>
    This system is for educational and research purposes only.
    It is NOT a substitute for professional medical diagnosis.
    Always consult a qualified healthcare provider.<br><br>
     <span style="color:#63b3ed">StrokeAI</span> — Expert System with Neural Network &nbsp;|&nbsp;
    Built with MLP Classifier + SMOTE Balancing
</div>
""", unsafe_allow_html=True)