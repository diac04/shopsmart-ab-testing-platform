"""
business_impact.py
------------------
Interactive calculator. Users adjust traffic, AOV, and rollout
assumptions to see projected revenue impact and ROI per experiment.
"""

import streamlit as st
import pandas as pd
from dashboard_data import EXPERIMENT_SUMMARY, COMBINED_IMPACT
import plotly.graph_objects as go


def show():
    st.markdown("# Business Impact Calculator")
    st.markdown(
        "<div class='subtitle'>Adjust business assumptions to see how projected "
        "revenue impact and ROI change. Defaults reflect the assumptions used "
        "in the final analysis.</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    options = {
        "Personalized Recommendations": "exp2",
        "Discount Banner Placement": "exp3",
        "UPI Checkout Redesign (blocked)": "exp1",
    }
    selection = st.selectbox("Select Experiment", list(options.keys()))
    exp_id = options[selection]
    data = EXPERIMENT_SUMMARY[exp_id]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Experiment 2: revenue per user model ─────────────────────────────
    if exp_id == "exp2":
        st.markdown("## Assumptions")
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_users = st.number_input(
                "Monthly active users",
                min_value=100_000, max_value=50_000_000,
                value=10_000_000, step=500_000,
                help="Users exposed to recommendations after full rollout"
            )
        with c2:
            lift_choice = st.select_slider(
                "Lift scenario (Rs per user per month)",
                options=["Conservative (8.92)", "Point estimate (18.90)",
                         "Optimistic (28.91)"],
                value="Conservative (8.92)"
            )
        with c3:
            dev_cost_lakh = st.number_input(
                "Development cost (Rs Lakh)",
                min_value=1, max_value=500, value=20, step=1
            )

        lift_map = {
            "Conservative (8.92)": data["ci_lower"],
            "Point estimate (18.90)": data["cuped_lift"],
            "Optimistic (28.91)": data["ci_upper"],
        }
        lift = lift_map[lift_choice]

        monthly_rev = monthly_users * lift
        annual_rev = monthly_rev * 12
        dev_cost = dev_cost_lakh * 100_000
        payback_days = (dev_cost / monthly_rev) * 30 if monthly_rev > 0 else 0
        roi_pct = ((annual_rev - dev_cost) / dev_cost) * 100 if dev_cost > 0 else 0

        st.markdown("## Projected Impact")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Monthly Revenue", f"Rs {monthly_rev/1e7:.2f} Cr")
        with m2:
            st.metric("Annual Revenue", f"Rs {annual_rev/1e7:.2f} Cr")
        with m3:
            st.metric("Payback Period", f"{payback_days:.1f} days")
        with m4:
            st.metric("First-Year ROI", f"{roi_pct:,.0f}%")

        # Scenario comparison chart
        st.markdown("## Scenario Comparison")
        scenarios = ["Conservative", "Point estimate", "Optimistic"]
        annual_values = [
            monthly_users * data["ci_lower"] * 12 / 1e7,
            monthly_users * data["cuped_lift"] * 12 / 1e7,
            monthly_users * data["ci_upper"] * 12 / 1e7,
        ]
        fig = go.Figure(go.Bar(
            x=scenarios, y=annual_values,
            marker_color=["#059669", "#2563EB", "#7C3AED"],
            text=[f"Rs {v:.1f} Cr" for v in annual_values],
            textposition="outside", width=0.5
        ))
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=30, b=40),
            yaxis_title="Annual Revenue (Rs Crore)",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
            showlegend=False,
            font=dict(family="sans-serif", color="#374151", size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<div class='subtitle'>Policy: the conservative scenario "
            "(95 percent CI lower bound) is the figure quoted to leadership. "
            "The optimistic scenario is shown for range only, never as a "
            "commitment.</div>",
            unsafe_allow_html=True
        )

    # ── Experiment 3: CTR and conversion funnel model ────────────────────
    elif exp_id == "exp3":
        st.markdown("## Assumptions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            monthly_visitors = st.number_input(
                "Monthly banner impressions",
                min_value=50_000, max_value=20_000_000,
                value=400_000, step=50_000
            )
        with c2:
            ctr_lift_pp = st.slider(
                "CTR lift (pp)",
                min_value=0.1, max_value=1.5,
                value=float(data["overall_lift_pp"]), step=0.01
            )
        with c3:
            click_to_conv = st.slider(
                "Click-to-conversion rate (%)",
                min_value=5, max_value=30, value=15, step=1,
                help="ASSUMPTION - validate with GA4 post-launch"
            )
        with c4:
            aov = st.number_input(
                "Average order value (Rs)",
                min_value=200, max_value=10_000, value=1_200, step=100
            )

        extra_clicks = monthly_visitors * (ctr_lift_pp / 100)
        extra_conversions = extra_clicks * (click_to_conv / 100)
        monthly_rev = extra_conversions * aov
        annual_rev = monthly_rev * 12
        dev_cost = data["dev_cost_lakh"] * 100_000
        payback_days = (dev_cost / monthly_rev) * 30 if monthly_rev > 0 else 0

        st.markdown("## Projected Impact")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Extra Clicks / Month", f"{extra_clicks:,.0f}")
        with m2:
            st.metric("Extra Conversions / Month", f"{extra_conversions:,.0f}")
        with m3:
            st.metric("Monthly Revenue", f"Rs {monthly_rev/1e5:.2f} L")
        with m4:
            st.metric("Payback Period", f"{payback_days:.0f} days")

        st.markdown(
            f"""
            <div class="status-card">
                <div class="card-label">Annual Projection</div>
                <div class="card-value">
                    At these assumptions this change generates approximately
                    <b>Rs {annual_rev/1e7:.2f} Crore per year</b> against a one-time
                    development cost of Rs {data['dev_cost_lakh']} Lakh.
                    Note the click-to-conversion rate of {click_to_conv} percent is an
                    assumption pending GA4 validation - the true figure may shift
                    this projection materially.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Experiment 1: avoided-loss model ─────────────────────────────────
    else:
        st.markdown("## Why This Experiment Has No Revenue Projection")
        st.markdown(
            f"""
            <div class="status-card status-card-hold">
                <span class="status-tag status-tag-hold">Do Not Ship</span>
                <div class="card-title">Avoided Loss, Not Revenue</div>
                <div class="card-value">
                    This experiment showed a negative effect ({data['lift_pp']:.3f} pp).
                    The business value here is the loss we avoided by not shipping.
                </div>
                <div class="card-label">Avoided Loss at 50 Percent Rollout</div>
                <div class="card-value">Rs {data['avoided_loss_crore_50pct']:.2f} Crore per year</div>
                <div class="card-label">Avoided Loss at 100 Percent Rollout</div>
                <div class="card-value">Rs {data['avoided_loss_crore_100pct']:.2f} Crore per year</div>
                <div class="card-label">Protected Revenue From Early Stop (SPRT)</div>
                <div class="card-value">Rs {data['protected_revenue_lakh_month']:.2f} Lakh per month</div>
                <div class="card-label">Reliability Warning</div>
                <div class="card-value">
                    A sample ratio mismatch (T/C = {data['srm_ratio']}) was detected,
                    so all figures carry high uncertainty. The experiment must be
                    re-run after the traffic split mechanism is fixed.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )