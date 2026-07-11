import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import joblib
from scipy.optimize import differential_evolution
import os

# ---------- 1. PAGE CONFIG & LIGHT THEME CSS ----------

st.set_page_config(
    layout="wide",
    page_title="Recurve Arrow Design Optimizer",
    page_icon="🏹"
)

st.markdown("""
<style>
    /* Apple-like system fonts */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     system-ui, -system-ui, "Segoe UI", Roboto, sans-serif;
        background-color: #0B1120;
        color: #E5E7EB;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .app-header {
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #F9FAFB;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #9CA3AF;
        margin-top: 0.35rem;
    }

    /* Panels / cards */
    .panel {
        background: radial-gradient(circle at top left, #1E293B 0, #020617 55%);
        border-radius: 18px;
        border: 1px solid #111827;
        padding: 20px 22px 22px 22px;
        box-shadow: 0 22px 45px rgba(0,0,0,0.65);
    }

    .stat-card {
        background: #020617;
        border-radius: 16px;
        border: 1px solid #111827;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 22px 45px rgba(0,0,0,0.60);
    }
    .stat-label {
        font-size: 0.78rem;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .stat-main {
        font-size: 2.0rem;
        font-weight: 700;
        color: #F9FAFB;
    }
    .stat-unit {
        font-size: 0.95rem;
        color: #9CA3AF;
        margin-left: 0.2rem;
        font-weight: 500;
    }
    .stat-chip {
        display: inline-block;
        padding: 0.16rem 0.65rem;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-top: 0.45rem;
    }
    .chip-green { background: rgba(16,185,129,0.18); color:#6EE7B7; }
    .chip-amber { background: rgba(245,158,11,0.18); color:#FBBF24; }
    .chip-red   { background: rgba(239,68,68,0.18);  color:#FCA5A5; }

    /* Primary buttons */
    .stButton>button {
        background: linear-gradient(120deg, #2563EB 0%, #4F46E5 50%, #EC4899 100%) !important;
        color: #F9FAFB !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 0.52rem 1.4rem !important;
        font-size: 0.94rem !important;
    }
    .stButton>button:hover {
        filter: brightness(1.08);
    }

    hr {
        border: none;
        border-top: 1px solid #1F2937;
        margin: 1.5rem 0;
    }

    /* BIG current‑spec card */
    .spec-card {
        margin-top: 1.4rem;
        background: #020617;
        border-radius: 16px;
        border: 1px solid #111827;
        padding: 16px 18px 14px 18px;
        box-shadow: inset 0 0 0 1px rgba(148,163,184,0.12);
    }
    .spec-title {
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #9CA3AF;
        margin-bottom: 0.55rem;
        font-weight: 600;
    }
    .spec-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px 18px;
    }
    .spec-label {
        font-size: 0.78rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.1rem;
    }
    .spec-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #E5E7EB;
        letter-spacing: -0.01em;
    }

</style>
""", unsafe_allow_html=True)
# st.markdown("""
# <style>
#     /* Force SF / Apple-style font on main app containers */
#     * {
#         font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
#                      system-ui, -system-ui, "Segoe UI", Roboto, sans-serif !important;
#     }

#     /* Override Streamlit specific text classes */
#     .stMarkdown, .stText, .stRadio, .stSelectbox, .stSlider,
#     div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
#     div[data-testid="stMarkdownContainer"], label, span, p, h1, h2, h3, h4, h5, h6 {
#         font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
#                      system-ui, -system-ui, "Segoe UI", Roboto, sans-serif !important;
#     }
# </style>
# """, unsafe_allow_html=True)



# ---------- 2. BACKEND (MODEL & OPTIMIZER) ----------

@st.cache_resource
def load_brain():
    try:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "archery_model.pkl")
        scaler_path = os.path.join(base_dir, "archery_scaler.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Model files not found. Error: {e}")
        return None, None

model, scaler = load_brain()

def predict_performance(length, point, density, fletch):
    if model is None:
        return 0.0, 0.0
    df = pd.DataFrame([{
        "Arrow_Length_cm": length,
        "Point_Weight_g": point,
        "Shaft_Density_g_cm": density,
        "Fletch_Area_cm2": fletch
    }])
    pred = model.predict(scaler.transform(df))
    return float(pred[0][0]), abs(float(pred[0][1]))

def optimize_arrow(priority: str = "balanced"):
    def objective(p):
        L, P, D, F = p
        speed, group = predict_performance(L, P, D, F)
        if priority == "speed":
            score = -speed + group * 0.1
        elif priority == "accuracy":
            score = group + (100 - speed) * 0.01
        else:
            score = group + (100 - speed) * 0.05
        return score

    bounds = [(58, 82), (5, 10), (0.1, 0.25), (3, 15)]
    with st.spinner("Searching design space..."):
        result = differential_evolution(objective, bounds, maxiter=40, seed=42, workers=1)
    return result.x

# ---------- 3. HEADER ----------

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Recurve Arrow Design Optimizer</div>
        <div class="app-subtitle">
            Multi‑objective tuning of shaft length, mass distribution, density and fletching for high‑performance recurve setups.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- 4. MODE & CONTROL PANEL ----------

mode = st.radio(
    "Mode",
    ["Auto‑Optimize", "Manual Lab"],
    horizontal=True,
    label_visibility="collapsed",
)

if "params" not in st.session_state:
    st.session_state.params = [65.0, 7.5, 0.150, 9.0]

col_ctrl, col_stats = st.columns([1.1, 2.2])

with col_ctrl:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Control panel")

    if mode == "Auto‑Optimize":
        priority = st.selectbox(
            "Optimization objective",
            ["Balanced profile", "Maximum velocity", "Maximum precision"],
            index=0,
        )
        map_pr = {
            "Balanced profile": "balanced",
            "Maximum velocity": "speed",
            "Maximum precision": "accuracy",
        }
        if st.button("Run optimizer", use_container_width=True):
            L, P, D, F = optimize_arrow(map_pr[priority])
            st.session_state.params = [L, P, D, F]
            st.success("Design updated from optimization run.")
    else:
        L0, P0, D0, F0 = st.session_state.params
        L = st.slider("Shaft length (cm)", 58.0, 82.0, float(L0))
        P = st.slider("Point weight (g)", 5.0, 10.0, float(P0))
        D = st.slider("Shaft density (g/cm\u00b3)", 0.10, 0.25, float(D0), format="%.3f")
        F = st.slider("Fletching area (cm²)", 3.0, 15.0, float(F0))
        st.session_state.params = [L, P, D, F]

    L, P, D, F = st.session_state.params
        # Big spec card using full width
    st.markdown(f"""
    <div class="spec-card">
        <div class="spec-title">Current design parameters</div>
        <div class="spec-grid">
            <div>
                <div class="spec-label">Length</div>
                <div class="spec-value">{L:.1f} cm</div>
            </div>
            <div>
                <div class="spec-label">Point mass</div>
                <div class="spec-value">{P:.1f} g</div>
            </div>
            <div>
                <div class="spec-label">Shaft density</div>
                <div class="spec-value">{D:.3f} g/cm³</div>
            </div>
            <div>
                <div class="spec-label">Fletching area</div>
                <div class="spec-value">{F:.1f} cm²</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------- 5. PERFORMANCE METRICS ----------

with col_stats:
    L, P, D, F = st.session_state.params
    speed, group = predict_performance(L, P, D, F)
    total_mass = L * D + P + 1.5
    foc = (P / total_mass) * 100

    st.markdown("#### Performance metrics")

    def chip_class(val, good, mid_low=None, higher_is_better=True):
        if higher_is_better:
            if val >= good:
                return "chip-green", "HIGH"
            elif mid_low is not None and val >= mid_low:
                return "chip-amber", "MEDIUM"
            else:
                return "chip-red", "LOW"
        else:
            if val <= good:
                return "chip-green", "TIGHT"
            elif mid_low is not None and val <= mid_low:
                return "chip-amber", "MODERATE"
            else:
                return "chip-red", "WIDE"

    c1, c2, c3, c4 = st.columns(4)

    cls, txt = chip_class(speed, good=80, mid_low=70, higher_is_better=True)
    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Velocity</div>
                <div class="stat-main">{speed:.1f}<span class="stat-unit"> m/s</span></div>
                <span class="stat-chip {cls}">{txt}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    cls, txt = chip_class(group, good=15, mid_low=30, higher_is_better=False)
    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Group size</div>
                <div class="stat-main">{group:.1f}<span class="stat-unit"> mm</span></div>
                <span class="stat-chip {cls}">{txt}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Total mass</div>
                <div class="stat-main">{total_mass:.1f}<span class="stat-unit"> g</span></div>
                <span class="stat-chip chip-amber">REFERENCE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">FOC balance</div>
                <div class="stat-main">{foc:.1f}<span class="stat-unit"> %</span></div>
                <span class="stat-chip chip-amber">BALANCE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- 6. TARGET SIMULATION (CENTERED) ----------

st.markdown("---")
st.subheader("Virtual grouping at 70 m")

left, mid, right = st.columns([1, 2, 1])

with mid:
    sigma = group / 2.0
    hits_x = np.random.normal(0, sigma, 12)
    hits_y = np.random.normal(0, sigma, 12)

    fig_tgt, ax_tgt = plt.subplots(figsize=(5, 5))
    ax_tgt.set_facecolor("white")

    colors = ["#FEF3C7", "#FEE2E2", "#FECACA", "#BFDBFE", "#E5E7EB"]
    radii = [12.5, 25, 37.5, 50, 62.5]
    for r, col in zip(reversed(radii), reversed(colors)):
        ax_tgt.add_patch(plt.Circle((0, 0), r, color=col, ec="#9ca3af", alpha=0.95))

    ax_tgt.scatter(hits_x, hits_y, c="black", s=35, marker="x", zorder=10)
    ax_tgt.plot([-5, 5], [0, 0], "k-", lw=0.4)
    ax_tgt.plot([0, 0], [-5, 5], "k-", lw=0.4)

    lim = max(70, group * 1.6)
    ax_tgt.set_xlim(-lim, lim)
    ax_tgt.set_ylim(-lim, lim)
    ax_tgt.axis("off")

    st.pyplot(fig_tgt)

# ---------- 7. BLUEPRINT VISUALIZATION ----------

st.markdown("---")
st.subheader("Arrow blueprint")

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

shaft_h = 0.25
vane_len = 3.0 + (F / 8.0)
vane_height = 0.8 + (F / 20.0)
fletch_x = L - 3.5 - vane_len
balance_point = (L / 2) - ((foc - 20) / 100 * L)

ax.add_patch(Rectangle((0, -shaft_h / 2), L, shaft_h, fc="#111827", ec="black", lw=1.3))
ax.add_patch(Rectangle((0, 0.04), L, 0.05, fc="#6B7280", ec="none", alpha=0.35))

point_len = 2.5
ax.add_patch(
    Polygon(
        [[-point_len, 0], [0, shaft_h / 2], [0, -shaft_h / 2]],
        closed=True,
        fc="#D1D5DB",
        ec="#6B7280",
        lw=1.2,
    )
)

ax.add_patch(Rectangle((L, -shaft_h / 2), 1.2, shaft_h, fc="#F97316", ec="#C2410C"))

def draw_swept_vane(x_start, y_start, length, height, flip=False):
    y_dir = 1 if not flip else -1
    sweep = length * 0.25
    verts = [
        (x_start, y_start),
        (x_start + length, y_start),
        (x_start + length + sweep, y_start + height * y_dir),
        (x_start + sweep, y_start + height * y_dir),
        (x_start, y_start),
    ]
    return Polygon(verts, closed=True, facecolor="#22C55E", edgecolor="#15803D", alpha=0.9, lw=1.1)

ax.add_patch(draw_swept_vane(fletch_x, shaft_h / 2, vane_len, vane_height, flip=False))
ax.add_patch(draw_swept_vane(fletch_x, -shaft_h / 2, vane_len, vane_height, flip=True))

plt.text(
    L / 2,
    3.2,
    "Optimal recurve arrow blueprint",
    ha="center",
    fontsize=18,
    fontweight="bold",
    color="#111827",
)

stats_text = f"Predicted speed: {speed:.1f} m/s   |   Predicted group: {group:.1f} mm"
ax.text(
    L / 2,
    2.4,
    stats_text,
    ha="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#DBEAFE", edgecolor="#60A5FA"),
)

ax.axvline(
    x=balance_point,
    color="#DC2626",
    linestyle="--",
    lw=1.8,
    ymax=0.55,
    ymin=0.32,
)
ax.text(
    balance_point,
    1.1,
    f"FOC: {foc:.1f}%",
    color="#B91C1C",
    ha="center",
    fontweight="bold",
    fontsize=10,
)
ax.text(
    balance_point,
    0.8,
    "Balance point",
    color="#B91C1C",
    ha="center",
    fontsize=8,
    style="italic",
)

ax.annotate(
    "",
    xy=(0, -2.0),
    xytext=(L, -2.0),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.4),
)
ax.text(
    L / 2,
    -2.4,
    f"Shaft length: {L:.1f} cm",
    ha="center",
    fontsize=12,
    fontweight="bold",
)

ax.annotate(
    f"Point\n{P:.1f} g",
    xy=(-1, 0),
    xytext=(-8, 1.4),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="black"),
    ha="center",
    fontsize=10,
)

ax.annotate(
    f"Vane area\n{F:.1f} cm²",
    xy=(fletch_x + vane_len / 2, vane_height),
    xytext=(fletch_x, 2.4),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="black"),
    ha="center",
    fontsize=10,
)

ax.text(
    L / 2,
    -0.6,
    f"Shaft density: {D:.3f} g/cm³",
    ha="center",
    fontsize=9,
    color="#6B7280",
)

ax.set_xlim(-10, L + 10)
ax.set_ylim(-3, 4)
ax.axis("off")

st.pyplot(fig)
