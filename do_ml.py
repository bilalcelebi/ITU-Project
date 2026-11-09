import pandas as pd
import streamlit as st
from skopt.space import Integer, Real, Categorical

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, recall_score
import time
import shap
import numpy as np


lgbm_search_space = {
    "model__n_estimators": Integer(200, 800),
    "model__max_depth": Categorical([3, 5, 7, 9, -1]),
    "model__learning_rate": Real(0.005, 0.1, prior='log-uniform'),
    "model__num_leaves": Integer(15, 127),
    "model__min_child_samples": Integer(5, 50),
    "model__subsample": Real(0.6, 1.0),
    "model__colsample_bytree": Real(0.6, 1.0),
    "model__scale_pos_weight": Real(1.0, 5.0),
    "model__reg_alpha": Real(1e-3, 10.0, prior='log-uniform'),
    "model__reg_lambda": Real(1e-3, 10.0, prior='log-uniform')
}

xgb_search_space = {
    "model__n_estimators": Integer(200, 800),
    "model__max_depth": Integer(3, 9),
    "model__learning_rate": Real(0.005, 0.1, prior='log-uniform'),
    "model__subsample": Real(0.6, 1.0),
    "model__colsample_bytree": Real(0.6, 1.0),
    "model__min_child_weight": Integer(1, 15),
    "model__scale_pos_weight": Real(1.0, 5.0),
    "model__alpha": Real(1e-3, 10.0, prior='log-uniform'),
    "model__lambda": Real(1e-3, 10.0, prior='log-uniform')
}

rf_search_space = {
    "model__n_estimators": Integer(200, 800),
    "model__max_depth": Integer(5, 25),
    "model__min_samples_split": Integer(2, 15),
    "model__min_samples_leaf": Integer(1, 8),
    "model__max_features": Categorical(["sqrt", "log2", None]),
    "model__class_weight": Categorical(["balanced", "balanced_subsample"])
}

dt_space = {
    'model__criterion': Categorical(['gini', 'entropy', 'log_loss']),
    'model__max_depth': Integer(3, 30),
    'model__min_samples_split': Integer(2, 20),
    'model__min_samples_leaf': Integer(1, 20),
    'model__max_features': Categorical(['sqrt', 'log2', None])
}

gbc_space = {
    'model__n_estimators': Integer(100, 500),
    'model__learning_rate': Real(1e-3, 0.3, prior='log-uniform'),
    'model__max_depth': Integer(3, 10),
    'model__min_samples_split': Integer(2, 20),
    'model__min_samples_leaf': Integer(1, 20),
    'model__subsample': Real(0.5, 1.0),
    'model__max_features': Categorical(['sqrt', 'log2', None])
}



lgbm_model = LGBMClassifier(random_state=42, verbose=-1, importance_type='gain')
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
gbc_model = GradientBoostingClassifier(random_state=42)



models = {
    "LightGBM": {"model":lgbm_model, "param_grid":lgbm_search_space},
    "XGBoost": {"model":xgb_model, "param_grid":xgb_search_space},
    "Random Forest": {"model":rf_model, "param_grid":rf_search_space},
    "Gradient Boosting": {"model":gbc_model, "param_grid": gbc_space},
    "Decision Tree": {"model":dt_model, "param_grid":dt_space}
}


if "data" in st.session_state:
    data = st.session_state['data']


prepare_tab, execute_tab, results_tab = st.tabs(["Prepare Data", "Execute ML Operations", "Results and Metrics"], key="active_tab", on_change="rerun")


if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Prepare Data"

def switch_to_execute():
    st.session_state["active_tab"] = "Execute ML Operations"

def switch_to_metrics():
    st.session_state["active_tab"] = "Results and Metrics"

def switch_to_prepare():
    st.session_state["active_tab"] = "Prepare Data"


with prepare_tab:
    num_columns = st.multiselect(
        "Define Numerical Columns",
        options = data.columns.tolist(),
        default = data.select_dtypes(include = ['number']).columns.tolist()
    )

    cat_columns = st.multiselect(
        "Define Categorical Columns",
        options = data.columns.tolist(),
        default = data.select_dtypes(include = ['category', 'object']).columns.tolist()
    )

    target_column = st.selectbox(
        "Select Target Variable",
        options = data.columns.tolist()
    )

    selected_cols = num_columns + cat_columns
    if target_column not in selected_cols:
        selected_cols.append(target_column)

    data = data[selected_cols]

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size_slider = st.slider("Define Test Size", 0, 100, 20, step=5)
        test_size = test_size_slider / 100
    with col2:
        cv_size = st.slider("Define CV Size", 0, 10, 5, step=1)
    with col3:
        stratify_to_target = st.checkbox(f"Stratify to {target_column}")


    train_models = st.multiselect(
        "Select Models for Evaluation",
        options = list(models.keys()),
        default = list(models.keys())
    )

    x = data.drop(columns = [target_column])
    y = data[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        random_state=42,
        test_size = test_size,
        stratify = y if stratify_to_target else None
    )

    st.subheader("Look Into The Train Data")
    train_col, test_col = st.columns([3,1])

    with train_col:
        st.dataframe(x_train)
    with test_col:
        st.dataframe(y_train)


    st.session_state['processed_data'] = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test    
    }

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Data Process", type="secondary"):
            st.switch_page("load_data.py")
    with col2:
        st.button("Continue to Execute", type="primary", on_click=switch_to_execute)



with execute_tab:

    if "job_done" not in st.session_state:
        st.session_state['job_done'] = False

    st.subheader("Train Data")
    st.dataframe(st.session_state['processed_data']['x_train'])

    st.subheader("Test Data")
    st.dataframe(st.session_state['processed_data']['x_test'])

    cv = StratifiedKFold(n_splits = cv_size, random_state = 42, shuffle=True)

    pr_index = 0

    result = []

    best_model = None
    best_score = float('-inf')
    best_model_name = None

    if st.button("Execute Training", type="primary"):
        processor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_columns)
            ],
            remainder="passthrough"
        )
        processor.set_output(transform="pandas")

        start_time = time.time()

        if not train_models:
            st.warning("Select at least 1 Model")
            st.stop()

        progress = st.progress(0, text="Modeller Eğitiliyor...")
        
        for name, model in models.items():
            if name in train_models:
                percentage = (pr_index + 1) / len(train_models)

                pipe = Pipeline([
                    ("preprocessor", processor),
                    ("feature_selection", SelectFromModel(estimator = model['model'], threshold="median")),
                    ("model", model['model'])
                ])

                grid = BayesSearchCV(
                    pipe,
                    model['param_grid'],
                    cv = cv,
                    scoring = "average_precision",
                    random_state = 42,
                    verbose = 0
                )

                grid.fit(st.session_state['processed_data']['x_train'], st.session_state['processed_data']['y_train'])
                score = grid.best_score_

                if score > best_score:
                    best_model = grid.best_estimator_
                    best_score = float(score)
                    best_model_name = name

                result.append({
                    "Model": name,
                    "Score": score
                })

                print(f"Done : {name}")

                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / percentage) / 60

                progress.progress(percentage, text = f"{name} Modeli Eğitildi... - ETA : {estimated_total_time:.1f} Minute")

                pr_index += 1

        progress.empty()
        st.session_state['best_model'] = best_model
        st.session_state['results'] = pd.DataFrame(result)

        st.subheader("Model Scores")
        st.dataframe(st.session_state['results'], hide_index=True)

        if "best_model" in st.session_state and st.session_state['best_model'] is not None:
            best_model = st.session_state['best_model']
            preds = best_model.predict(st.session_state['processed_data']['x_test'])
            pred_proba = best_model.predict_proba(st.session_state['processed_data']['x_test'])[:, 1]

            class_report = classification_report(st.session_state['processed_data']['y_test'], preds, output_dict=True)
            cm_labels = [0,1]
            cm = confusion_matrix(st.session_state['processed_data']['y_test'], preds, labels=cm_labels)
            cmd = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)

            auc_score = roc_auc_score(st.session_state['processed_data']['y_test'], pred_proba)
            acc_score = accuracy_score(st.session_state['processed_data']['y_test'], preds)
            rec_score = recall_score(st.session_state['processed_data']['y_test'], preds, zero_division=1)
            avg_precision = average_precision_score(st.session_state['processed_data']['y_test'], pred_proba)

            supports = best_model.named_steps['feature_selection'].get_support()
            label_names = best_model.named_steps['preprocessor'].get_feature_names_out()
            labels = label_names[supports]

            importances = best_model.named_steps['model'].feature_importances_
            feature_importances = pd.DataFrame({
                "Feature": labels,
                "Importance": importances
            }).reset_index(drop=True)

            shap_test = best_model.named_steps['preprocessor'].transform(st.session_state['processed_data']['x_test'])
            shap_test_selected = best_model.named_steps['feature_selection'].transform(shap_test)
            explainer = shap.TreeExplainer(best_model.named_steps['model'])

            shap_values = explainer.shap_values(shap_test_selected)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            mean_shap_values = np.mean(np.abs(shap_values), axis = 0)
            shap_df = pd.DataFrame({
                "Feature": labels,
                "Shap Values (Mean)": mean_shap_values
            }).reset_index(drop=True)

            st.session_state['job_done'] = True
            st.session_state['cm_matrix'] = cmd
            st.session_state['metrics'] = {
                "auc_score": auc_score,
                "acc_score": acc_score,
                "rec_score": rec_score,
                "avg_precision": avg_precision
            }
            st.session_state['feature_importances'] = feature_importances
            st.session_state['shap_df'] = shap_df
            st.session_state['class_report'] = class_report
        else:
            st.write("There is not best Model")


    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.button("Back to Prepare", type="secondary", on_click=switch_to_prepare)
    with b_col2:
        st.button("Continue to Metrics", type="primary", on_click=switch_to_metrics)


with results_tab:
    if st.session_state['job_done'] != False:
        st.subheader("Confusion Matrix")
        st.dataframe(st.session_state['cm_matrix'])

        st.subheader("Classification Report")
        st.dataframe(st.session_state['class_report'])

        st.subheader("Other Metics")
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.metric(label="AUC Score", value = round(st.session_state['metrics']['auc_score'], 3))
        with col2:
            st.metric(label="Accuracy Score", value = round(st.session_state['metrics']['acc_score'], 3))
        with col3:
            st.metric(label="Recall Score", value = round(st.session_state['metrics']['rec_score'], 3))
        with col4:
            st.metric(label="Avg. Precision", value = round(st.session_state['metrics']['avg_precision'], 3))


        st.subheader("Feature Importances")
        st.bar_chart(st.session_state['feature_importances'], x = "Feature", y = "Importance", horizontal=True)

        st.subheader("Shap Report")
        st.bar_chart(st.session_state['shap_df'], x = "Feature", y = "Shap Values (Mean)", horizontal=True)


        if "results" in st.session_state:
            st.subheader("All Models Results")
            st.dataframe(st.session_state['results'])


    col1, col2 = st.columns(2)

    with col1:
        st.button("Back to Execution", type="secondary", on_click=switch_to_execute)
    with col2:
        if st.button("See the Report", type="primary"):
            st.switch_page("report.py")
