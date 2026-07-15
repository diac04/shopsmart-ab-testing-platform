"""
deep_dive.py
------------
Per-experiment drill-down with funnel visualisation, statistical test
results, confidence interval plots, and day-by-day trend charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dashboard_data import (
    EXPERIMENT_SUMMARY,
    load_exp1_daily_rates,
    load_exp2_method_comparison,
    load_exp3_segmented_corrected,
)


def _plot_ci(lift, lower, upper, x_label="Lift", zero_line=True):
    """Simple confidence interval visualisation."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[lower, upper],
        y=[0, 0],
        mode="lines",
        line=dict(color="#2563EB", width=4),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[lift],
        y=[0],
        mode="markers",
        marker=dict(size=14, color="#1E40AF", line=dict(color="white", width=2)),
        showlegend=False,
        name="Point estimate",
        hovertemplate=f"Point estimate: {lift}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[lower], y=[0], mode="markers",
        marker=dict(size=10, color="#2563EB", symbol="line-ns", line=dict(width=3)),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[upper], y=[0], mode="markers",
        marker=dict(size=10, color="#2563EB", symbol="line-ns", line=dict(width=3)),
        showlegend=False, hoverinfo="skip"
    ))

    if zero_line:
        fig.add_vline(x=0, line_dash="dash", line_color="#9CA3AF", line_width=1)

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis_title=x_label,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="sans-serif", color="#374151", size=12)
    )
    return fig


def _render_exp1(data):
    st.markdown("### Statistical Test Results")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Control Rate", f"{data['control_rate']*100:.3f}%")
    with c2:
        st.metric("Treatment Rate", f"{data['treatment_rate']*100:.3f}%")
    with c3:
        st.metric("Observed Lift", f"{data['lift_pp']:.3f} pp")
    with c4:
        st.metric("p-value", f"{data['p_value']:.3f}")

    st.markdown("### 95 Percent Confidence Interval")
    st.markdown(
        f"<div class='subtitle'>The interval crosses zero, meaning the observed "
        f"difference is not statistically distinguishable from no effect.</div>",
        unsafe_allow_html=True
    )
    fig = _plot_ci(
        data["lift_pp"],
        data["ci_lower_pp"],
        data["ci_upper_pp"],
        x_label="Lift (percentage points)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Day-by-Day Conversion Rate")
    daily_df = load_exp1_daily_rates()
    if daily_df.empty:
        st.info("Daily conversion rate file not found. Skipping trend chart.")
    else:
        daily_df.columns = [c.lower() for c in daily_df.columns]

        control_df = daily_df[daily_df["group"] == "control"].sort_values("day")
        treat_df = daily_df[daily_df["group"] == "treatment"].sort_values("day")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=control_df["day"], y=control_df["rate"] * 100,
            mode="lines+markers", name="Control",
            line=dict(color="#6B7280", width=2),
            marker=dict(size=6)
        ))
        fig2.add_trace(go.Scatter(
            x=treat_df["day"], y=treat_df["rate"] * 100,
            mode="lines+markers", name="Treatment",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=6)
        ))

        fig2.update_layout(
            height=340, margin=dict(l=10, r=10, t=30, b=40),
            xaxis_title="Day",
            yaxis_title="Conversion Rate (%)",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F3F4F6", dtick=1),
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
            legend=dict(orientation="h", y=1.15, x=0),
            font=dict(family="sans-serif", color="#374151", size=12)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "<div class='subtitle'>Treatment (blue) stays consistently at or below "
            "control (grey) across all 14 days, confirming the negative effect is "
            "not a novelty artefact.</div>",
            unsafe_allow_html=True
        )


def _render_exp2(data):
    st.markdown("### Statistical Test Results")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CUPED Lift", f"Rs {data['cuped_lift']:.2f}")
    with c2:
        st.metric("Raw Lift (rejected)", f"Rs {data['raw_lift_rejected']:.2f}")
    with c3:
        st.metric("Bias Removed", f"Rs {data['bias_removed']:.2f}")
    with c4:
        st.metric("Correlation (rho)", f"{data['cuped_rho']:.3f}")

    st.markdown("### 95 Percent Bootstrap Confidence Interval")
    st.markdown(
        "<div class='subtitle'>The lower bound (Rs 8.92) is what we quote to leadership.</div>",
        unsafe_allow_html=True
    )
    fig = _plot_ci(
        data["cuped_lift"],
        data["ci_lower"],
        data["ci_upper"],
        x_label="Lift (Rs per user per month)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Raw vs CUPED Comparison")
    st.markdown(
        "<div class='subtitle'>CUPED corrects for pre-experiment group imbalance, "
        "revealing the true treatment effect.</div>",
        unsafe_allow_html=True
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=["Raw Lift (biased)", "CUPED Lift (corrected)"],
        y=[data["raw_lift_rejected"], data["cuped_lift"]],
        marker_color=["#F87171", "#059669"],
        text=[f"Rs {data['raw_lift_rejected']:.2f}", f"Rs {data['cuped_lift']:.2f}"],
        textposition="outside",
        width=0.5
    ))
    fig2.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=40),
        yaxis_title="Lift (Rs per user per month)",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        showlegend=False,
        font=dict(family="sans-serif", color="#374151", size=12)
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_exp3(data):
    st.markdown("### Statistical Test Results")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall Lift", f"+{data['overall_lift_pp']:.2f} pp")
    with c2:
        st.metric("Relative Lift", f"+{data['overall_lift_pct']:.1f}%")
    with c3:
        st.metric("Z-statistic", f"{data['z_stat']:.2f}")
    with c4:
        st.metric("Interaction p-value", f"{data['interaction_p']:.2f}")

    st.markdown("### Segment Breakdown (Bonferroni-corrected)")
    st.markdown(
        "<div class='subtitle'>Both mobile and desktop segments show statistically "
        "significant lift after correcting for multiple comparisons.</div>",
        unsafe_allow_html=True
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Mobile", "Desktop"],
        y=[data["mobile_lift_pp"], data["desktop_lift_pp"]],
        marker_color=["#2563EB", "#7C3AED"],
        text=[f"+{data['mobile_lift_pp']:.2f} pp", f"+{data['desktop_lift_pp']:.2f} pp"],
        textposition="outside",
        width=0.5
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=40),
        yaxis_title="Lift (percentage points)",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
        showlegend=False,
        font=dict(family="sans-serif", color="#374151", size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interpretation")
    st.markdown(
        f"""
        <div class="status-card">
            <div class="card-label">Overall Result</div>
            <div class="card-value">
                Ship on both mobile and desktop. Mobile lift ({data['mobile_lift_pp']:.2f} pp)
                is directionally larger than desktop ({data['desktop_lift_pp']:.2f} pp),
                but the interaction term is not statistically significant
                (p = {data['interaction_p']:.2f}), so we cannot formally claim
                mobile benefits more than desktop.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show():
    st.markdown("# Experiment Deep Dive")
    st.markdown(
        "<div class='subtitle'>Detailed statistical results and visualisations "
        "for the selected experiment.</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    options = {d["name"]: k for k, d in EXPERIMENT_SUMMARY.items()}
    selection = st.selectbox("Select Experiment", list(options.keys()))
    exp_id = options[selection]
    data = EXPERIMENT_SUMMARY[exp_id]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"## {data['name']}")
    st.markdown(
        f"<div class='subtitle'>{data['phase']} &nbsp;·&nbsp; "
        f"Metric: {data['metric']} &nbsp;·&nbsp; "
        f"Duration: {data['duration_days']} days</div>",
        unsafe_allow_html=True
    )

    if exp_id == "exp1":
        _render_exp1(data)
    elif exp_id == "exp2":
        _render_exp2(data)
    else:
        _render_exp3(data)