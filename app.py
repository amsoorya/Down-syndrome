"""
Fetal Down Syndrome Screening System
=====================================
Uses clinically-validated markers for Trisomy 21 risk estimation:
- Nuchal Translucency (NT) thickness (primary ultrasound marker)
- Nasal bone presence/absence
- Maternal serum markers: free beta-hCG, PAPP-A
- Soft markers: echogenic bowel, short femur, EIF, choroid plexus cysts

Combines First-Trimester Combined Test (FTS) approach used in clinical practice.
Uses a Bayesian risk model based on published likelihood ratios from:
  - FMF (Fetal Medicine Foundation) guidelines
  - Nicolaides et al. (2011) NT-based screening
  - Snijders et al. (1998) age-related risk tables

NOTE: For demonstration/educational purposes only.
      Real screening requires calibrated ultrasound equipment and certified sonographers.
"""

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal T21 Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #58a6ff !important;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .risk-low    { color: #3fb950; font-weight: 600; font-size: 1.4em; }
    .risk-med    { color: #d29922; font-weight: 600; font-size: 1.4em; }
    .risk-high   { color: #f85149; font-weight: 600; font-size: 1.4em; }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78em;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }
    .badge-green  { background: #1a4a2e; color: #3fb950; border: 1px solid #3fb950; }
    .badge-yellow { background: #3d2b0d; color: #d29922; border: 1px solid #d29922; }
    .badge-red    { background: #4a1a1a; color: #f85149; border: 1px solid #f85149; }
    .footnote {
        font-size: 0.75em;
        color: #8b949e;
        border-top: 1px solid #30363d;
        padding-top: 10px;
        margin-top: 20px;
    }
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CLINICAL RISK MODEL (Bayesian / FMF-based)
# ─────────────────────────────────────────────

def age_based_prior(maternal_age: float, gestational_weeks: float) -> float:
    """
    Age-specific background risk for Trisomy 21.
    Based on Snijders et al. (1998) and Hook (1981) tables.
    Returns risk as a fraction (e.g. 0.001 = 1:1000).
    """
    # Polynomial fit to published age-risk data at 10–14 weeks
    # Risk increases exponentially with age
    age_risks = {
        20: 1/1527, 25: 1/1352, 30: 1/895,
        32: 1/660,  34: 1/446,  35: 1/338,
        36: 1/249,  37: 1/185,  38: 1/137,
        39: 1/102,  40: 1/76,   41: 1/57,
        42: 1/43,   43: 1/32,   44: 1/24,
        45: 1/18,   46: 1/14,   47: 1/10,
    }
    ages = sorted(age_risks.keys())
    if maternal_age <= ages[0]:
        return age_risks[ages[0]]
    if maternal_age >= ages[-1]:
        return age_risks[ages[-1]]
    # Linear interpolation in log-risk space
    for i in range(len(ages) - 1):
        if ages[i] <= maternal_age <= ages[i + 1]:
            t = (maternal_age - ages[i]) / (ages[i + 1] - ages[i])
            log_r = (1 - t) * math.log(age_risks[ages[i]]) + t * math.log(age_risks[ages[i + 1]])
            return math.exp(log_r)
    return 1 / 500


def nt_likelihood_ratio(nt_mm: float, crown_rump_mm: float) -> tuple:
    """
    NT-based likelihood ratio using FMF delta-NT approach.
    The NT MoM (multiple of median) is derived from CRL-specific medians.
    
    Reference medians from Snijders et al. 1998 / Nicolaides 2004:
    LR table from Wright et al. 2008, Ultrasound Obstet Gynecol.
    """
    # Expected median NT by CRL (mm) — from FMF reference chart
    # Fitted: median_NT = 0.5445 * CRL^0.2668 (Kagan et al. 2008)
    if crown_rump_mm < 38:
        crown_rump_mm = 38
    if crown_rump_mm > 84:
        crown_rump_mm = 84

    median_nt = 0.5445 * (crown_rump_mm ** 0.2668)
    delta_nt = nt_mm - median_nt  # Difference from expected median

    # LR based on delta-NT (Kagan et al. 2008, Table 3)
    # Approximate look-up / piecewise
    if delta_nt < -0.5:
        lr = 0.14
    elif delta_nt < 0.0:
        lr = 0.40
    elif delta_nt < 0.5:
        lr = 0.75
    elif delta_nt < 1.0:
        lr = 1.5
    elif delta_nt < 1.5:
        lr = 3.0
    elif delta_nt < 2.0:
        lr = 6.0
    elif delta_nt < 2.5:
        lr = 12.0
    elif delta_nt < 3.0:
        lr = 24.0
    elif delta_nt < 4.0:
        lr = 48.0
    else:
        lr = 100.0

    return lr, median_nt, delta_nt


def nasal_bone_lr(nasal_bone: str) -> float:
    """
    Likelihood ratio for nasal bone status.
    Cicero et al. 2001, 2003 — NB absent in ~73% of T21 vs ~1.5% euploid.
    """
    if nasal_bone == "Absent":
        return 73.0 / 1.5
    elif nasal_bone == "Hypoplastic":
        return 3.5
    else:  # Present
        return 0.27


def serum_marker_lr(free_bhcg_mom: float, papp_a_mom: float) -> float:
    """
    Combined likelihood ratio from serum markers.
    Free β-hCG MoM and PAPP-A MoM.
    T21: free β-hCG ↑ (~2.0 MoM), PAPP-A ↓ (~0.4 MoM).
    
    Bivariate Gaussian LR (simplified univariate product approximation).
    Parameters from Wald et al. 2003.
    """
    # T21 distributions (mean MoM in log10 space)
    # log10 MoM ~ N(mean, sd)
    t21_bhcg_mean = math.log10(2.0)
    t21_bhcg_sd   = 0.26
    norm_bhcg_mean = 0.0
    norm_bhcg_sd   = 0.26

    t21_pappa_mean  = math.log10(0.38)
    t21_pappa_sd    = 0.28
    norm_pappa_mean = 0.0
    norm_pappa_sd   = 0.28

    def normal_pdf(x, mu, sigma):
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    log_bhcg  = math.log10(max(free_bhcg_mom, 0.01))
    log_pappa = math.log10(max(papp_a_mom,    0.01))

    lr_bhcg  = normal_pdf(log_bhcg,  t21_bhcg_mean,  t21_bhcg_sd)  / normal_pdf(log_bhcg,  norm_bhcg_mean,  norm_bhcg_sd)
    lr_pappa = normal_pdf(log_pappa, t21_pappa_mean, t21_pappa_sd) / normal_pdf(log_pappa, norm_pappa_mean, norm_pappa_sd)

    return lr_bhcg * lr_pappa


def soft_markers_lr(markers: list) -> float:
    """
    Combined LR from second-trimester soft markers.
    Each marker's isolated LR from Bromley et al. 2002, Smith-Bindman et al. 2001.
    """
    lr_map = {
        "Echogenic intracardiac focus (EIF)": 1.8,
        "Choroid plexus cysts":               1.3,
        "Short femur (< 5th percentile)":     1.6,
        "Echogenic bowel":                    6.7,
        "Pyelectasis":                        1.5,
        "Mild ventriculomegaly":              3.8,
        "Absent/hypoplastic nasal bone":      6.0,
        "Increased nuchal fold (≥6mm)":       11.0,
        "Short humerus":                      2.5,
    }
    combined = 1.0
    for m in markers:
        combined *= lr_map.get(m, 1.0)
    return combined


def compute_risk(prior, lr_nt, lr_nb, lr_serum, lr_soft):
    """Bayesian update: posterior odds = prior odds × product of LRs."""
    prior_odds = prior / (1 - prior)
    posterior_odds = prior_odds * lr_nt * lr_nb * lr_serum * lr_soft
    posterior_risk = posterior_odds / (1 + posterior_odds)
    return posterior_risk


def risk_label(risk: float):
    if risk < 1 / 1000:
        return "LOW", "badge-green", "risk-low"
    elif risk < 1 / 100:
        return "INTERMEDIATE", "badge-yellow", "risk-med"
    else:
        return "HIGH", "badge-red", "risk-high"


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def make_gauge(risk: float):
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    
    log_risk = math.log10(max(risk, 1e-6))
    log_min, log_max = math.log10(1e-5), math.log10(0.5)
    normalized = (log_risk - log_min) / (log_max - log_min)
    normalized = max(0, min(1, normalized))

    # Background arc segments
    theta = np.linspace(0, np.pi, 300)
    x_arc, y_arc = np.cos(theta), np.sin(theta)

    seg_colors = ["#3fb950", "#d29922", "#f85149"]
    seg_labels = ["Low\n(<1:1000)", "Intermediate\n(1:1000–1:100)", "High\n(>1:100)"]
    segs = [(0, 0.45), (0.45, 0.72), (0.72, 1.0)]
    for (lo, hi), col in zip(segs, seg_colors):
        t = np.linspace(lo * np.pi, hi * np.pi, 100)
        ax.fill_between(np.cos(t), np.sin(t) * 0.55, np.sin(t),
                        color=col, alpha=0.18)
        ax.plot(np.cos(t), np.sin(t), color=col, lw=5, alpha=0.7)

    # Needle
    needle_angle = np.pi * (1 - normalized)
    nx, ny = math.cos(needle_angle) * 0.78, math.sin(needle_angle) * 0.78
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#e6edf3", lw=2.5,
                                mutation_scale=18))
    ax.plot(0, 0, "o", color="#58a6ff", ms=8, zorder=5)

    # Risk text
    if risk < 1/1000:
        risk_str = f"1 : {int(round(1/risk)):,}"
        col = "#3fb950"
    elif risk < 1/10:
        risk_str = f"1 : {int(round(1/risk)):,}"
        col = "#d29922"
    else:
        risk_str = f"{risk*100:.1f}%"
        col = "#f85149"

    ax.text(0, -0.25, risk_str, ha="center", va="center", fontsize=18,
            color=col, fontweight="bold",
            fontfamily="monospace")
    ax.text(0, -0.48, "Estimated T21 Risk", ha="center", va="center",
            fontsize=8, color="#8b949e")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.6, 1.1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def make_lr_waterfall(prior, lr_nt, lr_nb, lr_serum, lr_soft, posterior):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    steps = [
        ("Age-based\nprior", prior),
        ("NT\n(LR×{:.1f})".format(lr_nt), prior * lr_nt),
        ("Nasal bone\n(LR×{:.1f})".format(lr_nb), prior * lr_nt * lr_nb),
        ("Serum\n(LR×{:.1f})".format(lr_serum), prior * lr_nt * lr_nb * lr_serum),
        ("Soft\nmarkers\n(LR×{:.1f})".format(lr_soft), posterior),
    ]

    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]

    colors = []
    for v in values:
        if v < 1/1000: colors.append("#3fb950")
        elif v < 1/100: colors.append("#d29922")
        else: colors.append("#f85149")

    bars = ax.bar(range(len(steps)), [v * 1000 for v in values],
                  color=colors, alpha=0.85, edgecolor="#30363d", linewidth=0.8,
                  width=0.55)

    for i, (bar, v) in enumerate(zip(bars, values)):
        lbl = f"1:{int(1/v):,}" if v < 0.05 else f"{v*100:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                lbl, ha="center", va="bottom", fontsize=8,
                color="#e6edf3", fontfamily="monospace")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, fontsize=8, color="#8b949e")
    ax.set_ylabel("Risk per 1,000", color="#8b949e", fontsize=9)
    ax.yaxis.label.set_color("#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.set_facecolor("#161b22")
    ax.set_title("Risk Propagation Through Markers", color="#58a6ff",
                 fontsize=10, fontfamily="monospace", pad=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Patient Parameters")
    st.markdown("---")

    st.markdown("### 👩 Maternal Info")
    maternal_age = st.slider("Maternal Age (years)", 18, 50, 32, 1)

    st.markdown("### 🫄 Gestational Info")
    ga_weeks = st.slider("Gestational Age (weeks)", 10, 22, 12, 1)
    crl_mm = st.slider("Crown-Rump Length (CRL, mm)", 38, 84, 65, 1,
                       help="Measured at 11–14 weeks. Used to derive NT median.")

    st.markdown("### 📡 First-Trimester Ultrasound")
    nt_mm = st.slider("Nuchal Translucency (NT, mm)", 0.5, 8.0, 2.0, 0.1,
                      help="Measured at 11+0 to 13+6 weeks. Normal < 3.5 mm.")
    nasal_bone = st.selectbox("Nasal Bone", ["Present", "Hypoplastic", "Absent"],
                              help="Nasal bone absent in ~73% of T21 fetuses.")

    st.markdown("### 🧪 Serum Markers (if available)")
    use_serum = st.checkbox("Include serum markers", value=True)
    if use_serum:
        free_bhcg = st.slider("Free β-hCG (MoM)", 0.1, 5.0, 1.0, 0.05,
                              help="T21: typically ↑ ~2.0 MoM. Normal: ~1.0 MoM.")
        papp_a = st.slider("PAPP-A (MoM)", 0.1, 3.0, 1.0, 0.05,
                           help="T21: typically ↓ ~0.4 MoM. Normal: ~1.0 MoM.")
    else:
        free_bhcg, papp_a = 1.0, 1.0

    st.markdown("### 🔍 Second-Trimester Soft Markers (optional)")
    available_soft = [
        "Echogenic intracardiac focus (EIF)",
        "Choroid plexus cysts",
        "Short femur (< 5th percentile)",
        "Echogenic bowel",
        "Pyelectasis",
        "Mild ventriculomegaly",
        "Increased nuchal fold (≥6mm)",
        "Short humerus",
    ]
    selected_soft = st.multiselect("Soft markers present", available_soft)


# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────

st.markdown("# 🔬 Fetal Down Syndrome (T21) Screening")
st.markdown("*First-Trimester Combined Test — Bayesian Risk Model*")
st.markdown("---")

# Compute
prior         = age_based_prior(maternal_age, ga_weeks)
lr_nt, median_nt, delta_nt = nt_likelihood_ratio(nt_mm, crl_mm)
lr_nb         = nasal_bone_lr(nasal_bone)
lr_serum      = serum_marker_lr(free_bhcg, papp_a) if use_serum else 1.0
lr_soft       = soft_markers_lr(selected_soft) if selected_soft else 1.0
posterior     = compute_risk(prior, lr_nt, lr_nb, lr_serum, lr_soft)
label, badge_cls, risk_cls = risk_label(posterior)

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    # Gauge
    fig_gauge = make_gauge(posterior)
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close()

    # Risk label badge
    risk_str = f"1:{int(round(1/posterior)):,}" if posterior < 0.05 else f"{posterior*100:.1f}%"
    st.markdown(
        f"""
        <div style="text-align:center; margin: 8px 0 18px 0;">
          <span class="badge {badge_cls}" style="font-size:1.1em; padding:6px 22px;">
            {label} RISK
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Waterfall
    fig_wf = make_lr_waterfall(prior, lr_nt, lr_nb, lr_serum, lr_soft, posterior)
    st.pyplot(fig_wf, use_container_width=True)
    plt.close()

with col2:
    st.markdown("### 📊 Clinical Summary")

    # Prior
    prior_str = f"1:{int(round(1/prior)):,}" if prior < 0.05 else f"{prior*100:.1f}%"
    st.metric("Age-based Background Risk", prior_str,
              help="Based on Snijders et al. 1998 age tables at first trimester")

    # NT analysis
    st.markdown("---")
    st.markdown("**Nuchal Translucency Analysis**")

    nt_cols = st.columns(3)
    nt_cols[0].metric("Measured NT", f"{nt_mm:.1f} mm")
    nt_cols[1].metric("Median for CRL", f"{median_nt:.2f} mm")
    delta_color = "normal" if abs(delta_nt) < 0.5 else ("inverse" if delta_nt > 0 else "normal")
    nt_cols[2].metric("Delta NT", f"{delta_nt:+.2f} mm",
                      delta=f"LR = {lr_nt:.1f}×", delta_color=delta_color)

    if nt_mm >= 3.5:
        st.warning("⚠️ NT ≥ 3.5 mm — significantly elevated. Referral for invasive testing strongly recommended.")
    elif nt_mm >= 2.5:
        st.info("ℹ️ NT in borderline range (2.5–3.5 mm). Combined test interpretation is important.")
    else:
        st.success("✅ NT within normal range for gestational age.")

    # Markers summary
    st.markdown("---")
    st.markdown("**Likelihood Ratios Applied**")

    lr_data = {
        "NT (ultrasound)":        lr_nt,
        "Nasal bone":             lr_nb,
        "Serum markers":          lr_serum if use_serum else None,
        "Soft markers (2nd tri)": lr_soft if selected_soft else None,
    }

    for name, lr in lr_data.items():
        if lr is None:
            continue
        if lr > 2:
            col_txt = "#f85149"
        elif lr < 0.5:
            col_txt = "#3fb950"
        else:
            col_txt = "#8b949e"
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; "
            f"padding:4px 0; border-bottom:1px solid #21262d;'>"
            f"<span style='color:#c9d1d9'>{name}</span>"
            f"<span style='color:{col_txt}; font-family:monospace; font-weight:600'>"
            f"× {lr:.2f}</span></div>",
            unsafe_allow_html=True
        )

    # Final risk
    st.markdown("---")
    posterior_str = f"1:{int(round(1/posterior)):,}" if posterior < 0.05 else f"{posterior*100:.1f}%"
    st.markdown(
        f"""
        <div class="metric-card" style="border-color:#58a6ff; margin-top:10px;">
          <div style="color:#8b949e; font-size:0.85em; margin-bottom:4px;">
            ADJUSTED RISK (Combined)
          </div>
          <div class="{risk_cls}">{posterior_str}</div>
          <div style="color:#8b949e; font-size:0.8em; margin-top:4px;">
            Standard cutoff for further testing: 1:300 (0.33%)
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Recommendation
    st.markdown("### 🏥 Clinical Guidance")
    if posterior >= 1 / 100:
        st.error(
            "**High-Risk Result.** Recommend offering chorionic villus sampling (CVS) "
            "or amniocentesis for definitive karyotyping. Genetic counseling strongly advised."
        )
    elif posterior >= 1 / 300:
        st.warning(
            "**Intermediate/High-Risk.** Consider offering diagnostic testing (CVS or "
            "amniocentesis) or Non-Invasive Prenatal Testing (NIPT) for further risk stratification."
        )
    elif posterior >= 1 / 1000:
        st.info(
            "**Intermediate-Risk.** Routine follow-up recommended. May consider NIPT for "
            "additional reassurance. Second-trimester anatomy scan advised."
        )
    else:
        st.success(
            "**Low-Risk Result.** Routine antenatal care. Second-trimester anatomy scan "
            "at 18–22 weeks recommended. Background risk is not eliminated."
        )

# ─────────────────────────────────────────────
# REFERENCE SECTION
# ─────────────────────────────────────────────

with st.expander("📚 Clinical References & Methodology"):
    st.markdown("""
    **Model Basis**
    
    This tool implements a **First-Trimester Combined Test** using Bayesian likelihood ratio methodology:

    - **Age-based prior**: Snijders RL et al. *Age- and gestation-specific risk for trisomy 21.* 
      Ultrasound Obstet Gynecol 1999;13:167–170.
    - **NT measurement**: Kagan KO et al. *Fetal nuchal translucency at 11–13⁺⁶ weeks.*
      FMF guidelines. Nicolaides KH, 2011.
    - **Nasal bone LR**: Cicero S et al. *Absence of nasal bone in fetuses with trisomy 21.*
      Lancet 2001;358:1665–1667.
    - **Serum markers (free β-hCG, PAPP-A)**: Wald NJ et al. *First and second trimester antenatal
      screening for Down's syndrome.* J Med Screen 2003;10:56–61.
    - **Soft markers (2nd trimester)**: Bromley B et al. *The genetic sonogram update.* 
      Obstet Gynecol 2002;99:435–440. Smith-Bindman R et al. JAMA 2001;285:1044–1055.

    **Limitations**
    - This model provides *risk estimates*, not diagnoses
    - Definitive diagnosis requires invasive testing (CVS, amniocentesis) with karyotyping
    - NT measurement quality is operator- and equipment-dependent
    - Serum MoM values are laboratory- and gestational-age-specific; local median adjustment required
    - NIPT has higher sensitivity/specificity than combined screening but is not diagnostic
    """)

st.markdown(
    """
    <div class="footnote">
    ⚠️ <b>Educational/Research Use Only.</b> This tool is not a medical device and has not been
    validated for clinical use. It is intended for educational demonstration of clinical screening
    methodology. All clinical decisions must be made by qualified healthcare professionals.
    Results must be interpreted in the context of complete clinical history, certified ultrasound
    measurements, and laboratory-specific reference ranges.
    </div>
    """,
    unsafe_allow_html=True
)
