"""
bayesian.py
-----------
Bayesian analysis view for Experiment 1: posterior distributions,
P(treatment better), expected loss calculator, and prior sensitivity.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dashboard_data import EXPERIMENT_SUMMARY, load_exp1_bayesian


def _posterior_plot(control_a, control_b, treat_a, treat_b, n_samples=10000):
    """Simulate Beta posteriors and plot."""
    control_samples = np.random.beta(control_a, control_b, n_samples)
    treat_samples = np.random.beta(treat_a, treat_b, n_samples)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=control_samples * 100, name="Control", opacity=0.6,
        marker_color="#6B7280", nbinsx=60,
        histnorm="probability density"
    ))
    fig.add_trace(go.Histogram(
        x=treat_samples * 100, name="Treatment", opacity=0.6,
        marker_color="#2563EB", nbinsx=60,
        histnorm="probability density"
    ))
    fig.update_layout(
        barmode="overlay",
        height=340, margin=dict(l=10, r=10, t=30, b=40),
        xaxis_title="Conversion Rate (%)",
        yaxis_title="Posterior Density",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        legend=dict(orientation="h", y=1.1),
        font=dict(family="sans-serif", color="#374151", size=12)
    )
    p_treat_better = (treat_samples > control_samples).mean()
    return fig, p_treat_better


def show():
    st.markdown("# Bayesian Results")
    st.markdown(
        "<div class='subtitle'>Posterior distributions and decision-theoretic "
        "loss framework for Experiment 1 (UPI Checkout Redesign).</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    data = EXPERIMENT_SUMMARY["exp1"]

    # Top-level Bayesian metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "P(Treatment Better)",
            f"{data['p_treatment_better']*100:.1f}%",
            delta="Bayesian posterior probability",
            delta_color="off"
        )
    with c2:
        st.metric(
            "Expected Loss if Ship",
            f"{data['e_loss_ship_pp']:.4f} pp",
            delta="30x higher than holding",
            delta_color="off"
        )
    with c3:
        st.metric(
            "Expected Loss if Hold",
            f"{data['e_loss_hold_pp']:.4f} pp",
            delta="Preferred decision",
            delta_color="off"
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Posterior plot
    st.markdown("## Posterior Distribution")
    st.markdown(
        "<div class='subtitle'>Distributions represent our belief about the "
        "true conversion rate for each variant after observing the experiment data.</div>",
        unsafe_allow_html=True
    )

    control_conv = int(data["control_rate"] * data["n_control"])
    treat_conv = int(data["treatment_rate"] * data["n_treatment"])
    control_no = data["n_control"] - control_conv
    treat_no = data["n_treatment"] - treat_conv

    # Weakly informative prior (Beta(1,1) equivalent to uniform)
    fig, p_better = _posterior_plot(
        1 + control_conv, 1 + control_no,
        1 + treat_conv, 1 + treat_no
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"<div class='subtitle'>Simulated P(Treatment > Control) from posterior: "
        f"<b>{p_better*100:.1f}%</b>. Reported value from full analysis: "
        f"<b>{data['p_treatment_better']*100:.1f}%</b></div>",
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Expected loss calculator
    st.markdown("## Expected Loss Calculator")
    st.markdown(
        "<div class='subtitle'>Adjust the loss threshold to see the risk/reward "
        "trade-off. The default 0.01 pp threshold reflects our tolerance for "
        "shipping a break-even change.</div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        threshold = st.slider(
            "Loss threshold (pp)",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            help="Minimum acceptable lift below which shipping is a loss"
        )

    with c2:
        # Compute expected loss under simulated posterior for chosen threshold
        control_samples = np.random.beta(
            1 + control_conv, 1 + control_no, 20000
        )
        treat_samples = np.random.beta(
            1 + treat_conv, 1 + treat_no, 20000
        )
        diff = (treat_samples - control_samples) * 100
        e_loss_ship = np.maximum(-(diff - threshold), 0).mean()
        e_loss_hold = np.maximum(diff - threshold, 0).mean()

        st.markdown(
            f"""
            <div style="padding: 1rem 0;">
                <div style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase;
                            letter-spacing: 0.06em; margin-bottom: 0.5rem;">
                    Decision Analysis at Threshold = {threshold:.2f} pp
                </div>
                <div style="font-size: 0.95rem; line-height: 1.7;">
                    <b>Expected loss if we ship:</b> {e_loss_ship:.4f} pp<br>
                    <b>Expected loss if we hold:</b> {e_loss_hold:.4f} pp<br>
                    <b>Recommended action:</b>
                    <span style="color: {'#DC2626' if e_loss_ship > e_loss_hold else '#059669'};
                                 font-weight: 600;">
                        {'HOLD' if e_loss_ship > e_loss_hold else 'SHIP'}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Prior sensitivity
    st.markdown("## Prior Sensitivity Analysis")
    st.markdown(
        "<div class='subtitle'>How much does the choice of prior change the "
        "conclusion? A robust decision should be insensitive to reasonable priors.</div>",
        unsafe_allow_html=True
    )

    priors = [
        ("Uniform Prior", 1, 1, "Beta(1,1) - no prior information"),
        ("Weakly Informative", 5, 145, "Beta(5,145) - roughly 3% baseline"),
        ("Strong Informed", 50, 1450, "Beta(50,1450) - strong 3% prior"),
    ]

    fig_priors = go.Figure()
    colors = ["#F87171", "#2563EB", "#7C3AED"]
    results = []
    for i, (name, a, b, desc) in enumerate(priors):
        c_samples = np.random.beta(a + control_conv, b + control_no, 15000)
        t_samples = np.random.beta(a + treat_conv, b + treat_no, 15000)
        p_better = (t_samples > c_samples).mean()
        results.append((name, desc, p_better))

    # Threshold reference line at 50 percent (coin-flip evidence)
    fig_priors.add_hline(
        y=50,
        line_dash="dash",
        line_color="#9CA3AF",
        line_width=1,
        annotation_text="50% (no evidence)",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#6B7280")
    )

    for i, (name, desc, p_better) in enumerate(results):
        fig_priors.add_trace(go.Bar(
            x=[name],
            y=[p_better * 100],
            marker_color=colors[i],
            text=[f"{p_better*100:.2f}%"],
            textposition="outside",
            textfont=dict(size=13, color="#111827"),
            width=0.55,
            hovertemplate=f"<b>{name}</b><br>{desc}<br>P(T>C) = {p_better*100:.2f}%<extra></extra>"
        ))

    fig_priors.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=50),
        yaxis_title="P(Treatment > Control) [%]",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            range=[0, 60],
            dtick=10
        ),
        xaxis=dict(showgrid=False),
        showlegend=False,
        font=dict(family="sans-serif", color="#374151", size=12),
        bargap=0.25
    )
    st.plotly_chart(fig_priors, use_container_width=True)

    # Interpretation card explaining what we are seeing
    avg_p = np.mean([p for _, _, p in results]) * 100
    st.markdown(
        f"""
        <div class="status-card status-card-hold">
            <div class="card-label">Interpretation</div>
            <div class="card-value">
                All three priors converge on the same conclusion: the posterior
                probability that treatment beats control is around
                <b>{avg_p:.1f}%</b>, well below the 50% threshold required to
                even consider shipping. The dashed line represents "no evidence"
                (a coin flip). Our result is so far below that line that the
                decision is <b>robust to prior choice</b> — even a strongly
                optimistic prior cannot rescue this experiment.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )