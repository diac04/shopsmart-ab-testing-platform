"""
health_monitor.py
-----------------
Statistical health checks: SRM gating status, guardrails, novelty
effect, and sequential testing boundary with cost-of-delay.
"""

import streamlit as st
import plotly.graph_objects as go
from dashboard_data import (
    EXPERIMENT_SUMMARY,
    load_exp1_daily_rates,
    load_exp1_sprt_boundaries,
    load_exp1_novelty,
    load_exp1_power_retro,
)


def show():
    st.markdown("# Statistical Health Monitor")
    st.markdown(
        "<div class='subtitle'>Validity checks that gate every experiment "
        "decision. A failed check invalidates downstream statistics regardless "
        "of how significant they look.</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── SRM gate ──────────────────────────────────────────────────────────
    st.markdown("## Sample Ratio Mismatch (SRM) - Gating Check")
    st.markdown(
        "<div class='subtitle'>SRM means the traffic split deviates from the "
        "planned 50/50 allocation. When present, randomisation cannot be "
        "trusted and every downstream statistic is suspect.</div>",
        unsafe_allow_html=True
    )

    srm_rows = [
        ("UPI Checkout Redesign",
         EXPERIMENT_SUMMARY["exp1"]["srm_flag"],
         f"T/C ratio = {EXPERIMENT_SUMMARY['exp1']['srm_ratio']}, "
         f"chi-square = {EXPERIMENT_SUMMARY['exp1']['srm_chi2']}, p < 0.001"),
        ("Personalized Recommendations", False,
         "50/50 split confirmed (6,000 per arm)"),
        ("Discount Banner Placement", False,
         "50/50 split confirmed (98,772 per arm)"),
    ]

    for name, failed, detail in srm_rows:
        card_class = "status-card-hold" if failed else "status-card-ship"
        tag_class = "status-tag-hold" if failed else "status-tag-ship"
        tag_text = "FAILED - Results Gated" if failed else "Passed"
        st.markdown(
            f"""
            <div class="status-card {card_class}">
                <span class="status-tag {tag_class}">{tag_text}</span>
                <div class="card-title">{name}</div>
                <div class="card-value">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Novelty effect ────────────────────────────────────────────────────
    st.markdown("## Novelty Effect Check (Experiment 1)")
    st.markdown(
        "<div class='subtitle'>If users react to change itself rather than the "
        "change's value, early lift fades. We compare early-window vs "
        "steady-state lift.</div>",
        unsafe_allow_html=True
    )

    daily_df = load_exp1_daily_rates()
    if not daily_df.empty:
        daily_df.columns = [c.lower() for c in daily_df.columns]
        control_df = daily_df[daily_df["group"] == "control"].sort_values("day")
        treat_df = daily_df[daily_df["group"] == "treatment"].sort_values("day")

        merged = control_df.merge(
            treat_df, on="day", suffixes=("_c", "_t")
        )
        merged["lift_pp"] = (merged["rate_t"] - merged["rate_c"]) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged["day"], y=merged["lift_pp"],
            mode="lines+markers", name="Daily lift",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=6)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#9CA3AF", line_width=1)
        fig.add_vrect(
            x0=0.5, x1=6.5,
            fillcolor="#FEF3C7", opacity=0.35, line_width=0,
            annotation_text="Novelty window (excluded)",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#92400E")
        )
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=30, b=40),
            xaxis_title="Day", yaxis_title="Lift (pp)",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F3F4F6", dtick=1),
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
            showlegend=False,
            font=dict(family="sans-serif", color="#374151", size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="status-card">
            <div class="card-label">Finding</div>
            <div class="card-value">
                Steady-state lift (Days 7-14) remains negative at -0.135 pp.
                The treatment is not merely suffering a temporary novelty dip -
                it is genuinely worse at steady state. Novelty test p = 0.397
                (no significant novelty pattern).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Sequential testing ────────────────────────────────────────────────
    st.markdown("## Sequential Testing (SPRT) - Early Stop")
    st.markdown(
        "<div class='subtitle'>The Sequential Probability Ratio Test allows "
        "stopping an experiment as soon as evidence is conclusive, instead of "
        "waiting for the full planned sample.</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("SPRT Decision", "Accepted H0",
                  delta="No effect / negative", delta_color="off")
    with c2:
        st.metric("Stopped At", "n = 600",
                  delta="vs 37,671 planned per group", delta_color="off")
    with c3:
        st.metric("Protected Revenue", "Rs 17.0 L / month",
                  delta="From stopping early", delta_color="off")

    boundaries_df = load_exp1_sprt_boundaries()
    if not boundaries_df.empty:
        st.markdown("### O'Brien-Fleming Boundaries")
        st.dataframe(boundaries_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="status-card">
            <div class="card-label">Cost of Delay Context</div>
            <div class="card-value">
                Counterfactual cost-of-delay was estimated at Rs 2.52 Lakh per month
                (the revenue foregone per month of testing if the treatment had been
                a true winner at the pre-registered MDE). Because the true effect was
                negative, this cost is moot - stopping early both saved testing
                time and protected revenue.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Power retrospective ───────────────────────────────────────────────
    st.markdown("## Power Retrospective (Experiment 1)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Observed Effect vs MDE", "40.5%",
                  delta="Effect much smaller than planned", delta_color="off")
    with c2:
        st.metric("Achieved Power", "10.5%",
                  delta="vs 80% planned", delta_color="off")
    with c3:
        st.metric("n Needed for Observed Effect", "631,448 / group",
                  delta="16.8x the planned sample", delta_color="off")

    st.markdown(
        """
        <div class="status-card">
            <div class="card-label">Lesson</div>
            <div class="card-value">
                The experiment was powered to detect a 0.4 pp lift, but the observed
                effect was only 40.5 percent of that. Detecting an effect this small
                would require over 630,000 users per group - 16.8x what was planned.
                This is a structural argument for pre-registering a realistic MDE
                grounded in prior effect sizes, not aspirational targets.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )