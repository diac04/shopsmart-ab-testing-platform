"""
overview.py
-----------
Portfolio-level snapshot of all three experiments with sample sizes,
duration, and colour-coded significance status.
"""

import streamlit as st
import pandas as pd
from dashboard_data import EXPERIMENT_SUMMARY


def show():
    st.markdown("# Experiment Overview")
    st.markdown(
        "<div class='subtitle'>Consolidated view of all three concurrent A/B tests. "
        "Sample sizes, duration, and significance status at a glance.</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # Summary metrics
    total_users = sum(
        d["n_control"] + d["n_treatment"] for d in EXPERIMENT_SUMMARY.values()
    )
    ship_count = sum(1 for d in EXPERIMENT_SUMMARY.values() if d["status"] == "SHIP")
    hold_count = sum(
        1 for d in EXPERIMENT_SUMMARY.values() if d["status"] == "DO NOT SHIP"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Experiments", "3", delta="Concurrent runs", delta_color="off")
    with c2:
        st.metric("Total Users", f"{total_users:,}", delta="Across all arms", delta_color="off")
    with c3:
        st.metric("Ship Decisions", str(ship_count), delta="Approved for rollout", delta_color="off")
    with c4:
        st.metric("Hold Decisions", str(hold_count), delta="Blocked", delta_color="off")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Comparison table
    st.markdown("## Experiment Comparison Table")

    rows = []
    for d in EXPERIMENT_SUMMARY.values():
        rows.append({
            "Experiment": d["name"],
            "Metric": d["metric"],
            "Control (n)": f"{d['n_control']:,}",
            "Treatment (n)": f"{d['n_treatment']:,}",
            "Duration": f"{d['duration_days']} days",
            "Significant": "Yes" if d["significant"] else "No",
            "Direction": d["direction"].capitalize(),
            "Decision": d["status"],
        })

    df = pd.DataFrame(rows)

    def style_decision(val):
        if val == "SHIP":
            return "background-color: #ECFDF5; color: #065F46; font-weight: 600;"
        elif val == "DO NOT SHIP":
            return "background-color: #FEF2F2; color: #991B1B; font-weight: 600;"
        return ""

    def style_direction(val):
        if val == "Positive":
            return "color: #059669; font-weight: 500;"
        elif val == "Negative":
            return "color: #DC2626; font-weight: 500;"
        return ""

    styled = (
        df.style
        .map(style_decision, subset=["Decision"])
        .map(style_direction, subset=["Direction"])
        .set_properties(**{"font-size": "0.9rem", "padding": "0.5rem"})
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Individual experiment summary cards
    st.markdown("## Individual Experiment Snapshots")

    for exp_id, data in EXPERIMENT_SUMMARY.items():
        is_ship = data["status"] == "SHIP"
        card_class = "status-card-ship" if is_ship else "status-card-hold"
        tag_class = "status-tag-ship" if is_ship else "status-tag-hold"
        tag_text = "Ship" if is_ship else "Do Not Ship"

        # Compute key statistic display based on experiment
        if exp_id == "exp1":
            key_stat = f"p = {data['p_value']:.3f} &nbsp;·&nbsp; lift = {data['lift_pp']:.3f} pp"
        elif exp_id == "exp2":
            key_stat = (
                f"CUPED lift = Rs {data['cuped_lift']:.2f} "
                f"[{data['ci_lower']:.2f}, {data['ci_upper']:.2f}]"
            )
        else:
            key_stat = (
                f"Z = {data['z_stat']:.2f} &nbsp;·&nbsp; "
                f"lift = +{data['overall_lift_pp']:.2f} pp"
            )

        st.markdown(
            f"""
            <div class="status-card {card_class}">
                <span class="status-tag {tag_class}">{tag_text}</span>
                <div class="card-title">{data['name']}</div>
                <div class="card-meta">
                    {data['phase']} &nbsp;·&nbsp; Metric: {data['metric']} &nbsp;·&nbsp;
                    {key_stat}
                </div>
                <div class="card-value">{data['headline']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )