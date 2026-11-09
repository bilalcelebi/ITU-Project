import streamlit as st

data_page = st.Page("load_data.py", title="Load Data")
ml_page = st.Page("do_ml.py", title = "ML Operations")
report_page = st.Page("report.py", title = "Report")

pg = st.navigation(
    {
        "Navigation":[data_page, ml_page, report_page]
    }
)

pg.run()

