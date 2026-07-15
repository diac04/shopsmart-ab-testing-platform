from pathlib import Path
import sys

import streamlit as st

st.set_page_config(
    page_title="ShopSmart Deployment Test",
    layout="wide"
)

st.title("ShopSmart Deployment Test")
st.success("Streamlit Cloud is working.")

st.write("Python version:", sys.version)
st.write("Current directory:", Path.cwd())
st.write("Application file:", Path(__file__).resolve())

project_root = Path(__file__).resolve().parent.parent

required_files = {
    "Dashboard data": project_root / "phase8" / "dashboard_data.py",
    "Business summary": project_root / "phase7" / "experiment_outputs" / "business_summary.json",
    "Decision table": project_root / "phase7" / "experiment_outputs" / "final_decision_table.csv",
}

st.subheader("Required File Check")

for name, path in required_files.items():
    if path.exists():
        st.success(f"{name}: found")
    else:
        st.error(f"{name}: missing — {path}")