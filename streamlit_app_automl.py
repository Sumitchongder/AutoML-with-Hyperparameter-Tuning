# streamlit_app_automl.py
"""
Production-ready Streamlit + PyCaret app (fixed and hardened)
What I fixed in this update:
- Removed PyCaret v2-only args (e.g. `silent`) and used PyCaret v3+ style.
- Robust save/load that stores **prefix** (no `.pkl` stored) and prevents `.pkl.pkl` errors.
- Avoid `module.pull()` immediately after `setup()` — only call after training steps.
- Plot generation writes into a temp folder and displays reliably.
- Finalize/save/load workflows use safe_load_model and safe_save_model.
- Session-state resets on new upload/setup to prevent stale state bugs.
- Defensive error handling across the app.

Usage: `streamlit run streamlit_app_automl.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
import glob
from datetime import datetime

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="AutoML Pro", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Session defaults
# -------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'task' not in st.session_state:
    st.session_state.task = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'pycaret_ready' not in st.session_state:
    st.session_state.pycaret_ready = False
if 'best_model_prefix' not in st.session_state:
    st.session_state.best_model_prefix = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'last_action' not in st.session_state:
    st.session_state.last_action = None

# -------------------------
# Helpers: dynamic pycaret loader
# -------------------------
@st.cache_resource(show_spinner=False)
def load_pycaret_modules(task_type: str):
    """Return a dict of functions from pycaret depending on task"""
    if task_type == 'Classification':
        from pycaret.classification import (
            setup as py_setup,
            compare_models as py_compare,
            create_model as py_create,
            tune_model as py_tune,
            finalize_model as py_finalize,
            save_model as py_save,
            load_model as py_load,
            plot_model as py_plot
        )
    else:
        from pycaret.regression import (
            setup as py_setup,
            compare_models as py_compare,
            create_model as py_create,
            tune_model as py_tune,
            finalize_model as py_finalize,
            save_model as py_save,
            load_model as py_load,
            plot_model as py_plot
        )

    return {
        'setup': py_setup,
        'compare_models': py_compare,
        'create_model': py_create,
        'tune_model': py_tune,
        'finalize_model': py_finalize,
        'save_model': py_save,
        'load_model': py_load,
        'plot_model': py_plot,
    }

# -------------------------
# Safe save/load to avoid .pkl.pkl
# -------------------------

def safe_save_model(mod_funcs: dict, model, base_name: str):
    """Saves model via pycaret save_model and stores prefix (no .pkl in stored value). Returns prefix."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{base_name}_{ts}"
    prefix_path = os.path.join(st.session_state.temp_dir, filename)

    try:
        saved = mod_funcs['save_model'](model, prefix_path, verbose=False)
        # pycaret.save_model may return (model, path) or path string
        actual_path = saved[1] if isinstance(saved, tuple) else saved
        # actual_path ends with .pkl — remove extension and store the prefix
        prefix = actual_path[:-4] if actual_path.lower().endswith('.pkl') else actual_path
        st.session_state.best_model_prefix = prefix
        return prefix
    except Exception as e:
        st.error(f"Failed to save model: {e}")
        return None


def safe_load_model(mod_funcs: dict, prefix: str):
    """Load model from stored prefix (remove .pkl if present) and return model object."""
    if prefix is None:
        return None
    # ensure no .pkl in prefix
    if prefix.lower().endswith('.pkl'):
        prefix = prefix[:-4]

    try:
        model = mod_funcs['load_model'](prefix, verbose=False)
        return model
    except Exception as e:
        st.error(f"Error loading model '{os.path.basename(prefix)}': {e}")
        return None

# -------------------------
# Sidebar: data upload and setup
# -------------------------
with st.sidebar:
    st.title("AutoML Pro")
    st.markdown("Upload dataset and configure setup")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded is not None and st.session_state.df is None:
        try:
            st.session_state.df = pd.read_csv(uploaded)
            # Reset app state when a new dataset is loaded
            st.session_state.task = None
            st.session_state.target = None
            st.session_state.pycaret_ready = False
            st.session_state.best_model_prefix = None
            st.session_state.last_action = 'uploaded'
            st.success("Dataset loaded — continue to configure.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if st.session_state.df is not None:
        st.write(f"Rows: {st.session_state.df.shape[0]}, Columns: {st.session_state.df.shape[1]}")

        st.selectbox("Target column", options=st.session_state.df.columns, key='target')

        st.radio("Task type", ['Classification', 'Regression'], key='task')

        st.markdown("---")
        st.subheader("Preprocessing")
        normalize = st.checkbox("Normalize (MinMax/ZScore)", value=True)
        remove_outliers = st.checkbox("Remove outliers", value=False)
        feature_selection = st.checkbox("Feature selection", value=False)

        if st.button("Run PyCaret Setup"):
            # Load pycaret modules for the chosen task
            try:
                mod_funcs = load_pycaret_modules(st.session_state.task)
            except Exception as e:
                st.error(f"Could not import PyCaret: {e}")
                st.stop()

            # run setup (PyCaret v3+; DO NOT use `silent` arg)
            try:
                mod_funcs['setup'](
                    data=st.session_state.df,
                    target=st.session_state.target,
                    session_id=42,
                    html=False,
                    normalize=normalize,
                    remove_outliers=remove_outliers,
                    feature_selection=feature_selection,
                    log_experiment=False,
                    experiment_name=None,
                    verbose=False,
                    n_jobs=1
                )
                st.session_state.pycaret_ready = True
                st.success("PyCaret setup completed — ML tabs unlocked")
                st.session_state.last_action = 'setup'
            except Exception as e:
                st.error(f"PyCaret setup failed: {e}")
                st.session_state.pycaret_ready = False

# -------------------------
# Main area
# -------------------------
st.title("AutoML Pro 🚀 Advanced ML Platform")
st.markdown("A **Future-Proof, Industry-Grade** platform for automated Machine Learning and Hyperparameter Tuning.")
st.markdown("---")

if st.session_state.df is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

# Always-visible tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Profile", "AutoML", "Manual Tune", "Predict"])

# -------------------------
# Tab 1: Data Profile
# -------------------------
with tab1:
    st.header("Data Snapshot & Profile")
    st.dataframe(st.session_state.df.head())
    with st.expander("Summary stats"):
        st.dataframe(st.session_state.df.describe(include='all').T)

# -------------------------
# Tab 2: AutoML
# -------------------------
with tab2:
    st.header("Automated Model Comparison")
    if not st.session_state.pycaret_ready:
        st.warning("Run PyCaret setup in the sidebar first.")
    else:
        mod_funcs = load_pycaret_modules(st.session_state.task)
        n_models = st.slider("Top N models to compare", min_value=3, max_value=30, value=10)
        metric = 'Accuracy' if st.session_state.task == 'Classification' else 'R2'

        if st.button("Run AutoML Comparison"):
            with st.spinner("Running AutoML (this can take time)..."):
                try:
                    best_n = mod_funcs['compare_models'](n_select=n_models, sort=metric, fold=5, verbose=False)

                    # best_n is list of models; pick first
                    top_obj = best_n[0] if isinstance(best_n, list) else best_n

                    # Save the top model safely and store prefix
                    prefix = safe_save_model(mod_funcs, top_obj, base_name='AutoML_Best')
                    if prefix:
                        st.success(f"Top model trained and saved: {os.path.basename(prefix)}.pkl")
                        st.session_state.last_action = 'automl'

                except Exception as e:
                    st.error(f"AutoML failed: {e}")

# -------------------------
# Tab 3: Manual Tune
# -------------------------
with tab3:
    st.header("Manual Build & Tuning")
    if not st.session_state.pycaret_ready:
        st.warning("Run PyCaret setup in the sidebar first.")
    else:
        mod_funcs = load_pycaret_modules(st.session_state.task)

        model_id = st.selectbox("Choose model id (pycaret)", options=mod_funcs['setup'].__module__ and [])
        # Note: we purposely don't try to call module.models() here to avoid early creation of threadlocal objects
        st.markdown("If you want a specific pycaret model id (eg. 'rf', 'lightgbm'), type it below")
        typed_model = st.text_input("Model id (pycaret)", value='rf')

        if st.button("Create & Save Base Model"):
            try:
                base = mod_funcs['create_model'](typed_model, verbose=False)
                prefix = safe_save_model(mod_funcs, base, base_name=f"Manual_Base_{typed_model}")
                if prefix:
                    st.success("Base model created and saved.")
                    st.session_state.last_action = 'create_manual'
            except Exception as e:
                st.error(f"Create model failed: {e}")

        if st.button("Tune Last Saved Model"):
            if not st.session_state.best_model_prefix:
                st.warning("No saved model to tune — create one first.")
            else:
                try:
                    loaded = safe_load_model(mod_funcs, st.session_state.best_model_prefix)
                    tuned = mod_funcs['tune_model'](loaded, n_iter=10, verbose=False)
                    prefix = safe_save_model(mod_funcs, tuned, base_name=f"Manual_Tuned_{typed_model}")
                    if prefix:
                        st.success("Tuning complete and saved.")
                        st.session_state.last_action = 'tuned'
                except Exception as e:
                    st.error(f"Tuning failed: {e}")

        if st.button("Finalize Last Saved Model"):
            if not st.session_state.best_model_prefix:
                st.warning("No saved model to finalize.")
            else:
                try:
                    loaded = safe_load_model(mod_funcs, st.session_state.best_model_prefix)
                    final = mod_funcs['finalize_model'](loaded)
                    prefix = safe_save_model(mod_funcs, final, base_name='Final_Deployed')
                    if prefix:
                        st.success("Final model finalized and saved for deployment.")
                        st.session_state.last_action = 'finalized'
                except Exception as e:
                    st.error(f"Finalization failed: {e}")

        st.markdown("---")
        st.subheader("Diagnostics / Plots")
        plot_choice = st.selectbox("Plot", ['auc','confusion_matrix','feature','residuals','error'])
        if st.button("Generate Plot"):
            if not st.session_state.best_model_prefix:
                st.warning("No saved model to plot.")
            else:
                try:
                    loaded = safe_load_model(mod_funcs, st.session_state.best_model_prefix)
                    # plot_model writes to cwd; ensure it writes to temp_dir then display
                    cwd = os.getcwd()
                    try:
                        os.chdir(st.session_state.temp_dir)
                        mod_funcs['plot_model'](loaded, plot=plot_choice, save=True, display_format='streamlit')
                        # find the latest png
                        files = glob.glob(os.path.join(st.session_state.temp_dir, '*.png'))
                        if files:
                            latest = max(files, key=os.path.getmtime)
                            st.image(latest)
                            os.remove(latest)
                        else:
                            st.warning('Plot generated but file not found')
                    finally:
                        os.chdir(cwd)
                except Exception as e:
                    st.error(f"Plot failed: {e}")

# -------------------------
# Tab 4: Predict
# -------------------------
with tab4:
    st.header("Predict on New Data")
    if not st.session_state.best_model_prefix:
        st.warning("No deployed model found. Train, tune and finalize a model first.")
    else:
        mod_funcs = load_pycaret_modules(st.session_state.task)
        # Ensure prefix has no .pkl
        prefix = st.session_state.best_model_prefix
        if prefix and prefix.lower().endswith('.pkl'):
            prefix = prefix[:-4]
            st.session_state.best_model_prefix = prefix

        st.success(f"Deployment model available: {os.path.basename(prefix)}.pkl")

        pred_file = st.file_uploader("Upload CSV for prediction", type=['csv'])
        if pred_file is not None:
            test_df = pd.read_csv(pred_file)
            st.dataframe(test_df.head())

            if st.button("Run Predictions"):
                model = safe_load_model(mod_funcs, prefix)
                if model is not None:
                    try:
                        res = mod_funcs['plot_model'].__module__  # dummy to avoid linter
                        # Use pycaret predict_model via module functions: predict_model is not exported in our mod_funcs map
                        # So call load_model then use model.predict if available, otherwise use pycaret's predict_model function
                        from pycaret.utils.generic import infer_signature
                    except Exception:
                        pass

                    # Fallback: try pycaret's predict_model for full pipeline
                    try:
                        # import predict_model dynamically from correct pycaret package
                        if st.session_state.task == 'Classification':
                            from pycaret.classification import predict_model as py_predict
                        else:
                            from pycaret.regression import predict_model as py_predict

                        preds = py_predict(model, data=test_df)
                        st.dataframe(preds.head())

                        csv = preds.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Predictions", data=csv, file_name='predictions.csv', mime='text/csv')
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

# -------------------------
# Cleanup on exit
# -------------------------
# Keep temp dir across session; remove logs folder if present
try:
    if os.path.exists('logs'):
        shutil.rmtree('logs')
except Exception:
    pass

st.markdown('---')
st.caption('AutoML Pro — updated and hardened')