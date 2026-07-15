"""
ab_testing_dashboard.py
------------------------
Main entry point for the ShopSmart India A/B Testing Platform dashboard.
"""

import streamlit as st

st.set_page_config(
    page_title="ShopSmart A/B Testing Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        /* Force sidebar to always be visible */
        [data-testid="stSidebar"] {
            min-width: 260px !important;
            max-width: 260px !important;
            width: 260px !important;
            transform: translateX(0px) !important;
            visibility: visible !important;
            display: block !important;
            position: relative !important;
            margin-left: 0 !important;
        }

        [data-testid="stSidebar"] > div {
            width: 260px !important;
            min-width: 260px !important;
        }

        /* Hide the collapse arrow completely */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        button[kind="headerNoPadding"] {
            display: none !important;
            visibility: hidden !important;
        }

        .main {
            margin-left: 0 !important;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #111827;
        }

        h1 {
            font-size: 1.65rem !important;
            font-weight: 600 !important;
            color: #111827 !important;
            letter-spacing: -0.01em;
            margin-bottom: 0.25rem !important;
        }
        h2 {
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            color: #1F2937 !important;
            margin-top: 1.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: #374151 !important;
            margin-top: 1.25rem !important;
        }

        .subtitle {
            color: #6B7280;
            font-size: 0.9rem;
            margin-top: -0.25rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            color: #111827 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #6B7280 !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.78rem !important;
            color: #6B7280 !important;
        }

        .status-card {
            border: 1px solid #E5E7EB;
            border-radius: 6px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.1rem;
            background-color: #FFFFFF;
        }
        .status-card-ship {
            border-left: 3px solid #059669;
        }
        .status-card-hold {
            border-left: 3px solid #DC2626;
        }
        .status-tag {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 3px;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }
        .status-tag-ship {
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
        }
        .status-tag-hold {
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
        }
        .card-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.15rem;
        }
        .card-meta {
            font-size: 0.75rem;
            color: #6B7280;
            margin-bottom: 0.9rem;
        }
        .card-label {
            font-size: 0.7rem;
            font-weight: 600;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.85rem;
            margin-bottom: 0.2rem;
        }
        .card-value {
            font-size: 0.9rem;
            color: #1F2937;
            line-height: 1.55;
        }

        section[data-testid="stSidebar"] {
            background-color: #F9FAFB;
            border-right: 1px solid #E5E7EB;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        hr {
            margin: 1rem 0 !important;
            border: none !important;
            border-top: 1px solid #E5E7EB !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.markdown(
        """
        <div style='padding: 0.2rem 0 0.4rem 0;'>
            <div style='font-size: 1rem; font-weight: 600; color: #111827;'>
                ShopSmart India
            </div>
            <div style='font-size: 0.78rem; color: #6B7280; margin-top: 0.15rem;'>
                A/B Testing Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase; "
        "letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Navigate</div>",
        unsafe_allow_html=True
    )

    page_selection = st.radio(
        "Navigate",
        [
            "Executive Summary",
            "Experiment Overview",
            "Statistical Health Monitor",
            "Experiment Deep Dive",
            "Bayesian Results",
            "Business Impact Calculator"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size: 0.7rem; color: #9CA3AF; line-height: 1.6;'>
            <div>Three concurrent experiments</div>
            <div>Analysis window: 14 days each</div>
        </div>
        """,
        unsafe_allow_html=True
    )


if page_selection == "Executive Summary":
    from views import executive_summary as page
elif page_selection == "Experiment Overview":
    from views import overview as page
elif page_selection == "Statistical Health Monitor":
    from views import health_monitor as page
elif page_selection == "Experiment Deep Dive":
    from views import deep_dive as page
elif page_selection == "Bayesian Results":
    from views import bayesian as page
elif page_selection == "Business Impact Calculator":
    from views import business_impact as page

page.show()