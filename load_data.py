import pandas as pd
import numpy as np
import streamlit as st
from data_utils import execute_conversion, execute_drop, execute_filling, execute_outliers, feature_engineering

@st.cache_data
def load_data():

    df = pd.read_csv("./bank_customer_data.csv")

    return df

data_load, view_data, prepare_data = st.tabs(["Data Loading", "Viewing Data", "Preparing Data"], key="active_tab", on_change="rerun")
df = load_data()

if "data" not in st.session_state:
    st.session_state['data'] = df

data_types = ["int64","float64","str","bool","category"]

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Data Loading"

def switch_to_view():
    st.session_state["active_tab"] = "Viewing Data"

def switch_to_prepare():
    st.session_state["active_tab"] = "Preparing Data"

def switch_to_load():
    st.session_state["active_tab"] = "Data Loading"


with data_load:
    st.header("Bank Customer Data")
    st.dataframe(st.session_state['data'])

    st.subheader("Informations about Data")
    info_df = pd.DataFrame({
        'Column': st.session_state['data'].columns,
        'Non-Null Count': st.session_state['data'].notna().sum().values,
        'Dtype': st.session_state['data'].dtypes.values
    }).reset_index(drop=True)

    st.dataframe(info_df)

    st.subheader("Description of Data")
    describe = st.session_state['data'].describe()

    st.dataframe(describe)

    col1, spacer, col2 = st.columns([4, 3, 2])

    with col1:
        st.download_button(
            label = "Download Updated Data",
            data = st.session_state['data'].to_csv(index = False),
            file_name = "updated_data.csv",
            mime="text/csv"
        )

    with col2:
        st.button("Data Conversion", type="primary", on_click=switch_to_view)


with view_data:
    dtypes = st.session_state['data'].dtypes.reset_index()
    dtypes = dtypes.rename(columns={"index":"Variable", 0:"Type"})
    dtypes["Type"] = dtypes["Type"].astype(str)
    
    #PS Kolondaki verilerin opsiyon listesindekilerle ayni olmasi onemli!
    st.header("Data Conversion")
    st.text("You can change Columns Data Types in this table. Click twice on Variable Type row and then you can select Variable Type that you decide.")
    dtypes = st.data_editor(
        dtypes,
        column_config = {
            "Type": st.column_config.SelectboxColumn(
                "Variable Type",
                help = "You can change type of Variable",
                options = data_types,
                required = True
            )
        },
        hide_index = True
    )

    if st.button("Execute Conversion", type = "primary"):
        execute_conversion(st.session_state['data'], dtypes)
        st.rerun()

    st.subheader("Preview Data")
    st.dataframe(st.session_state['data'], hide_index = True)

    col1, col2 = st.columns(2)

    with col1:
        st.button("Back To Loading", type="secondary", on_click=switch_to_load)
    with col2:
        st.button("Continue to Prepare", type="primary", on_click=switch_to_prepare)

with prepare_data:

    st.header("Drop Columns")
    drop_df = pd.DataFrame({
        "Column": st.session_state['data'].columns,
        "Drop": False
    })

    drop_df = st.data_editor(
        drop_df,
        column_config = {
            "Drop": st.column_config.SelectboxColumn(
                "Drop",
                help = "You can Select True if you want Drop the Column",
                options = [True, False],
                required = True
            )
        },
        hide_index = True
    )


    st.subheader("Filling None Variables")
    columns = st.session_state['data'].columns
    prepare_df = pd.DataFrame({
        "Variable":columns,
        "Null Strategy":"Median",
    })

    prepare_df = st.data_editor(
        prepare_df,
        column_config = {
            "Null Strategy": st.column_config.SelectboxColumn(
                "Filling Strategy",
                help = "Choose A Strategy for Filling NaN Variables",
                options = ["Mean", "Median", "Unkown"],
                required = True
            )
        },
        hide_index = True
    )

    st.subheader("Remove Outliers")
    outlier_toggle = st.toggle("Remove Outliers")

    if outlier_toggle:
        remove_outliers = st.multiselect(
            "Which Columns Do You Want Removing Outliers?",
            st.session_state['data'].columns.tolist(),
            default = st.session_state['data'].select_dtypes(include = ['number']).columns.tolist()
        )

    feature_eng_toogle = st.toggle("Make Feature Engineering")

    def execute_actions():
        execute_drop(st.session_state['data'], drop_df)
        execute_filling(st.session_state['data'], prepare_df)

        if outlier_toggle:
            execute_outliers(st.session_state['data'], list(remove_outliers))
        if feature_eng_toogle:
            feature_engineering(st.session_state['data'])


    if st.button("Execute Preperation", type = "primary"):
        execute_actions()
        st.rerun()


    col1, col2 = st.columns(2)

    with col1:
        st.button("Back to Conversion", type="secondary", on_click=switch_to_view)
    with col2:
        if st.button("Continue to ML", type="primary"):
            st.switch_page("do_ml.py")
    