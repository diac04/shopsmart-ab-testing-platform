"""
executive_summary.py
--------------------
Non-technical stakeholder view. One clean card per experiment with
headline, financial impact, and key caveat in plain language.
"""

import streamlit as st
from dashboard_data import EXPERIMENT_SUMMARY, COMBINED_IMPACT


def show():
    st.markdown("# Executive Summary")
    st.markdown(
        "<div class='subtitle'>Ship decisions and financial impact across three "
        "concurrent experiments</div>",
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Portfolio-level metrics
    st.markdown("## Portfolio Impact")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Annual Revenue Impact",
            f"Rs {COMBINED_IMPACT['shipped_annual_conservative_crore']:.2f} Cr",
            delta="Conservative 95% lower bound",
            delta_color="off"
        )
    with col2:
        st.metric(
            "Downside Avoided",
            f"Rs {COMBINED_IMPACT['avoided_loss_crore']:.2f} Cr",
            delta="Negative rollout blocked",
            delta_color="off"
        )
    with col3:
        st.metric(
            "Total Value Protected",
            f"Rs {COMBINED_IMPACT['total_protection_crore']:.2f} Cr",
            delta="Combined portfolio outcome",
            delta_color="off"
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Experiment Decisions")
    st.markdown(
        "<div class='subtitle'>Each card summarises one concluded experiment. "
        "Financial figures use the lower bound of the 95 percent confidence "
        "interval.</div>",
        unsafe_allow_html=True
    )

    for exp_id, data in EXPERIMENT_SUMMARY.items():
        is_ship = data["status"] == "SHIP"
        card_class = "status-card-ship" if is_ship else "status-card-hold"
        tag_class = "status-tag-ship" if is_ship else "status-tag-hold"
        tag_text = "Ship" if is_ship else "Do Not Ship"
        total_n = data["n_control"] + data["n_treatment"]

        st.markdown(
            f"""
            <div class="status-card {card_class}">
                <span class="status-tag {tag_class}">{tag_text}</span>
                <div class="card-title">{data['name']}</div>
                <div class="card-meta">
                    {data['phase']} &nbsp;·&nbsp; Sample size {total_n:,} &nbsp;·&nbsp;
                    Duration {data['duration_days']} days
                </div>
                <div class="card-label">Result</div>
                <div class="card-value">{data['headline']}</div>
                <div class="card-label">Financial Impact</div>
                <div class="card-value">{data['impact_plain']}</div>
                <div class="card-label">Key Caveat</div>
                <div class="card-value">{data['caveat']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )