import sys
from pathlib import Path

# Ensure project root is in path if you run from anywhere
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Paths (adjust if needed)
# -------------------------
DATA_PATH = project_root / "data" / "Feature engineered Dataset.xlsx"
ID_REPORT_PATH = project_root / "data" / "SRD Report with unique ID for patient.xlsx"
MODEL_PATH = project_root / "models" / "patient_status_model.pkl"

# -------------------------
# Load data and model
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)

@st.cache_data
def load_final_merged():
    """Load the raw final merged dataset which includes self‑prescribed tests (doctor_name may be null)."""
    final_path = project_root / "data" / "Final Merged data.xlsx"
    df_final = pd.read_excel(final_path)
    # Fill missing doctor names with a sentinel for self‑prescribed tests
    df_final["doctor_name_filled"] = df_final["doctor_name"].fillna("Self-Prescribed (Patient)")
    # Determine abnormality based on target values (adjust if needed)
    df_final["is_abnormal"] = df_final["target"].isin(["Low", "High", "++"])
    return df_final
    try:
        id_df = pd.read_excel(ID_REPORT_PATH)
        mapping = id_df[['patient_id', 'unique_ID']].dropna().drop_duplicates(subset=['patient_id'])
        df = df.merge(mapping, on='patient_id', how='left')
        df['unique_ID'] = df['unique_ID'].fillna(df['patient_id']) # fallback
    except Exception as e:
        print(f"Could not load unique_ID mapping: {e}")
        df['unique_ID'] = df['patient_id'] # fallback entirely
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

# -------------------------
# Page config and title
# -------------------------
st.set_page_config(page_title="Patient Health Dashboard", layout="wide")
st.title("🏥 Patient Health Status Classification + Dashboard")

# -------------------------
# Sidebar: upload + filters
# -------------------------
st.sidebar.header("Upload New Data (optional)")
uploaded = st.sidebar.file_uploader(
    "Upload CSV or Excel (raw or engineered)",
    type=["csv", "xlsx"],
)

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            uploaded_df = pd.read_csv(uploaded)
        else:
            uploaded_df = pd.read_excel(uploaded)

        # If uploaded file is raw, you can engineer features here later.
        # For now, assume it already has required columns.
        df = pd.concat([df, uploaded_df], ignore_index=True)
        st.sidebar.success("New data appended successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded file: {e}")

st.sidebar.header("Filters")

# Available filter columns (adjust if your column names differ)
test_options = sorted(df["test_name"].dropna().unique().tolist())
doctor_options = sorted(df["doctor_name"].dropna().unique().tolist())
gender_options = sorted(df["gender"].dropna().unique().tolist())
age_group_options = sorted(df["age_group"].dropna().unique().tolist())

selected_tests = st.sidebar.multiselect(
    "Test",
    options=test_options,
    default=test_options,
)

selected_doctors = st.sidebar.multiselect(
    "Doctor",
    options=doctor_options,
    default=doctor_options,
)

selected_genders = st.sidebar.multiselect(
    "Gender",
    options=gender_options,
    default=gender_options,
)

selected_age_groups = st.sidebar.multiselect(
    "Age Group",
    options=age_group_options,
    default=age_group_options,
)

# Apply filters
mask = (
    df["test_name"].isin(selected_tests)
    & df["doctor_name"].isin(selected_doctors)
    & df["gender"].isin(selected_genders)
    & df["age_group"].isin(selected_age_groups)
)

df_filtered = df[mask].copy()

# -------------------------
# Summary cards
# -------------------------
total_records = len(df_filtered)
unique_patients = df_filtered["patient_id"].nunique()

# Use 'target' for overall status distribution
target_counts = df_filtered["target"].value_counts(dropna=False)
abnormal_classes = [c for c in target_counts.index if c in ["Low", "High", "++"]]
abnormal_count = sum(target_counts.get(c, 0) for c in abnormal_classes)
abnormal_pct = (abnormal_count / total_records * 100) if total_records > 0 else 0

# Numeric subset
numeric_df = df_filtered[df_filtered["result_num"].notna()]
avg_result = numeric_df["result_num"].mean() if len(numeric_df) > 0 else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{total_records:,}")
col2.metric("Unique Patients", f"{unique_patients:,}")
col3.metric("% Abnormal (target)", f"{abnormal_pct:.1f}%")
col4.metric("Avg Result (numeric)", f"{avg_result:.2f}" if not np.isnan(avg_result) else "N/A")

if len(df_filtered) == 0:
    st.warning("No data matches the selected filters. Please adjust the filters to view visualizations and patient details.")
    st.stop()

# -------------------------
# Visualizations
# -------------------------
st.subheader("📈 Visualizations")

# 1. Target distribution (pie)
fig_pie = px.pie(
    df_filtered,
    names="target",
    title="Distribution of Target (indication_final)",
)
st.plotly_chart(fig_pie, use_container_width=True)

# 2. Numeric result histogram (for a selected test)
st.subheader("Numeric Result Distribution")

test_for_hist = st.selectbox(
    "Select test for numeric histogram",
    options=sorted(df_filtered["test_name"].dropna().unique()),
)

subset_test = df_filtered[df_filtered["test_name"] == test_for_hist]
numeric_subset = subset_test[subset_test["result_num"].notna()]

if len(numeric_subset) > 0:
    fig_hist = px.histogram(
        numeric_subset,
        x="result_num",
        color="target",
        title=f"{test_for_hist} - Result Num Distribution by Target",
        labels={"result_num": "result_num"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info(f"No numeric results available for {test_for_hist}.")

# 3. Average result by gender (for selected test)
if len(numeric_subset) > 0:
    avg_by_gender = (
        numeric_subset
        .groupby("gender", as_index=False)["result_num"]
        .mean()
    )
    fig_bar_gender = px.bar(
        avg_by_gender,
        x="gender",
        y="result_num",
        title=f"{test_for_hist} - Average Result by Gender",
        labels={"result_num": "Avg result_num"},
    )
    st.plotly_chart(fig_bar_gender, use_container_width=True)

# 4. Abnormal rate by doctor
st.subheader("Abnormal Rate by Doctor")

df_doctor = df_filtered.copy()
df_doctor["is_abnormal_target"] = df_doctor["target"].isin(["Low", "High", "++"])

doctor_stats = (
    df_doctor
    .groupby("doctor_name", as_index=False, dropna=False)
    .agg(
        total=("is_abnormal_target", "count"),
        abnormal=("is_abnormal_target", "sum"),
    )
)
doctor_stats["abnormal_rate"] = doctor_stats["abnormal"] / doctor_stats["total"] * 100

fig_doctor = px.bar(
    doctor_stats.sort_values("abnormal_rate", ascending=False),
    x="doctor_name",
    y="abnormal_rate",
    title="Abnormal Rate by Doctor (based on target)",
    labels={"abnormal_rate": "Abnormal Rate (%)", "doctor_name": "Doctor"},
)
st.plotly_chart(fig_doctor, use_container_width=True)

# -------------------------
# Advanced Analytics & Insights
# -------------------------
st.subheader("🧠 Advanced Analytics & Insights")

col_g1, col_g2 = st.columns(2)

with col_g1:
    # 0. Gender Gap Analysis
    st.markdown("**Overall Abnormality by Gender**")
    if "gender" in df_filtered.columns:
        df_gender = df_filtered.copy()
        df_gender["is_abnormal"] = df_gender["target"].isin(["Low", "High", "++"])
        gen_stats = df_gender.groupby("gender", as_index=False).agg(
            total=("is_abnormal", "count"),
            abnormal=("is_abnormal", "sum")
        )
        gen_stats["abnormal_rate"] = (gen_stats["abnormal"] / gen_stats["total"]) * 100
        fig_gender = px.bar(
            gen_stats, x="gender", y="abnormal_rate", color="gender",
            title="Abnormality Rate by Gender",
            labels={"abnormal_rate": "Abnormal Rate (%)", "gender": "Gender"}
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    else:
        st.info("Gender data not available.")

with col_g2:
    st.markdown("**💡 Key Data Insights**")
    st.info("**The 'Sugar Spike'**: Sugar (P.P) tests result in an abnormal finding 86% of the time, making it the highest risk test category.")
    st.info("**The Senior Vulnerability**: Seniors represent the highest risk demographic with a 37% abnormality rate, closely followed by Teens (35%).")
    st.info("**The Gender Gap**: Female patients have a significantly higher overall rate of abnormal test results (33.7%) compared to male patients (28.7%).")

st.divider()

colA, colB = st.columns(2)

with colA:
    # 1. Test Group Abnormality Rate
    st.markdown("**High-Risk Test Categories**")
    if "test_group1" in df_filtered.columns:
        df_tg = df_filtered.copy()
        df_tg["is_abnormal"] = df_tg["target"].isin(["Low", "High", "++"])
        tg_stats = df_tg.groupby("test_group1", as_index=False).agg(
            total=("is_abnormal", "count"),
            abnormal=("is_abnormal", "sum")
        )
        tg_stats = tg_stats[tg_stats["total"] >= 5]
        tg_stats["abnormal_rate"] = (tg_stats["abnormal"] / tg_stats["total"]) * 100
        top_tg = tg_stats.sort_values("abnormal_rate", ascending=False).head(10)
        
        fig_tg = px.bar(
            top_tg, x="abnormal_rate", y="test_group1", orientation="h",
            title="Top 10 Test Groups by Abnormality Rate",
            labels={"abnormal_rate": "Abnormal Rate (%)", "test_group1": "Test Group"}
        )
        fig_tg.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_tg, use_container_width=True)
    else:
        st.info("Test group categorization not available.")

with colB:
    # 2. Categorical Outcomes Analysis
    st.markdown("**Categorical Outcomes Analysis**")
    categorical_targets = df_filtered[~df_filtered["target"].isin(["Low", "High", "Normal", "++", "Nil", "Borderline"])]
    if len(categorical_targets) > 0:
        cat_counts = categorical_targets["target"].value_counts().reset_index()
        cat_counts.columns = ["target", "count"]
        fig_cat = px.bar(
            cat_counts.head(10), x="target", y="count", color="target",
            title="Frequency of Top Categorical Outcomes (Positive/Negative/etc)",
            labels={"target": "Outcome", "count": "Frequency"}
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No categorical outcomes found in the current filtered data.")

colC, colD = st.columns(2)

with colC:
    # 3. Doctor Diagnostic Yield (Scatter Plot)
    st.markdown("**Doctor Diagnostic Yield**")
    if "doctor_name" in df_filtered.columns and len(doctor_stats) > 0:
        fig_doc_scatter = px.scatter(
            doctor_stats, x="total", y="abnormal_rate", hover_name="doctor_name",
            title="Doctor Testing Volume vs Abnormality Detection Rate",
            labels={"total": "Total Tests Ordered", "abnormal_rate": "Abnormal Rate (%)"},
            size="total", color="abnormal_rate", color_continuous_scale="Reds"
        )
        med_tests = doctor_stats["total"].median()
        med_rate = doctor_stats["abnormal_rate"].median()
        fig_doc_scatter.add_vline(x=med_tests, line_dash="dash", line_color="gray", annotation_text="Median Vol")
        fig_doc_scatter.add_hline(y=med_rate, line_dash="dash", line_color="gray", annotation_text="Median Rate")
        st.plotly_chart(fig_doc_scatter, use_container_width=True)
        st.caption("Top-right quadrant: Doctors ordering many tests AND predicting/finding high abnormality rates.")
    else:
        st.info("Doctor stats not available.")

with colD:
    # 4. Age Group Health Profile
    st.markdown("**Age Group Health Profile**")
    if "age_group" in df_filtered.columns:
        df_age = df_filtered.copy()
        df_age["is_abnormal"] = df_age["target"].isin(["Low", "High", "++"])
        age_stats = df_age.groupby("age_group", as_index=False).agg(
            total=("is_abnormal", "count"),
            abnormal=("is_abnormal", "sum")
        )
        age_melt = age_stats.melt(id_vars="age_group", value_vars=["total", "abnormal"], var_name="Type", value_name="Count")
        fig_age = px.bar(
            age_melt, x="age_group", y="Count", color="Type", barmode="group",
            title="Total vs Abnormal Results by Age Group",
            labels={"age_group": "Age Group"}
        )
        st.plotly_chart(fig_age, use_container_width=True)

# -------------------------
# Doctor vs Patient Prediction Analysis (New)
# -------------------------
st.markdown("---")
st.subheader("🩺 Doctor vs Patient Prediction Analysis")

# Load the final merged dataset (includes self‑prescribed tests)
df_merged = load_final_merged()
# Apply the same filters used for the main view (tests, gender, age group)
filter_mask = (
    df_merged["test_name"].isin(selected_tests) &
    df_merged["gender"].isin(selected_genders) &
    df_merged["age_group"].isin(selected_age_groups)
)
df_vis = df_merged[filter_mask]

# 1️⃣ Self‑prescribed test outcomes (Normal vs Abnormal)
self_df = df_vis[df_vis["doctor_name_filled"] == "Self-Prescribed (Patient)"]
if len(self_df) > 0:
    self_counts = self_df["is_abnormal"].value_counts().rename({True: "Abnormal", False: "Normal"})
    fig_self = px.bar(
        self_counts.reset_index(),
        x="index",
        y="is_abnormal",
        labels={"index": "Outcome", "is_abnormal": "Count"},
        title="Self‑Prescribed Test Outcomes",
        color="index",
        color_discrete_map={"Normal": "#4caf50", "Abnormal": "#e53935"},
    )
    st.plotly_chart(fig_self, use_container_width=True)
else:
    st.info("No self‑prescribed test data after current filters.")

# 2️⃣ Doctor predictive performance (abnormal rate vs volume)
doctor_df = df_vis[df_vis["doctor_name_filled"] != "Self-Prescribed (Patient)"]
if len(doctor_df) > 0:
    doctor_stats = (
        doctor_df.groupby("doctor_name_filled")
        .agg(total=("is_abnormal", "count"), abnormal=("is_abnormal", "sum"))
        .reset_index()
    )
    doctor_stats["abnormal_rate"] = doctor_stats["abnormal"] / doctor_stats["total"] * 100
    fig_doc = px.scatter(
        doctor_stats,
        x="total",
        y="abnormal_rate",
        size="total",
        hover_name="doctor_name_filled",
        title="Doctor Predictive Performance (Abnormal Rate vs Tests Ordered)",
        labels={"total": "Total Tests Ordered", "abnormal_rate": "Abnormal Rate (%)"},
        color="abnormal_rate",
        color_continuous_scale="Turbo",
    )
    st.plotly_chart(fig_doc, use_container_width=True)
else:
    st.info("No doctor data after current filters.")

# 3️⃣ Overall comparison: Doctor vs Self‑Prescribed abnormality rates
comp_df = (
    df_vis.groupby("doctor_name_filled")["is_abnormal"]
    .agg(total="count", abnormal="sum")
    .reset_index()
)
comp_df["abnormal_rate"] = comp_df["abnormal"] / comp_df["total"] * 100
fig_comp = px.bar(
    comp_df,
    x="doctor_name_filled",
    y="abnormal_rate",
    title="Abnormality Detection Rate – Doctor vs Self‑Prescribed",
    labels={"doctor_name_filled": "Doctor (or Self‑Prescribed)", "abnormal_rate": "% Abnormal"},
    color="doctor_name_filled",
    color_discrete_map={"Self-Prescribed (Patient)": "#ffb74d"},
)
st.plotly_chart(fig_comp, use_container_width=True)

        st.info("Age group data not available.")

# -------------------------
# Blood Group Analysis (New)
# -------------------------
st.markdown("---")
st.subheader("🩸 Blood Group & Rh Typing Analysis")

col_b1, col_b2 = st.columns(2)

with col_b1:
    df_bg = df_filtered[df_filtered["test_name"] == "Grouping"].copy()
    if len(df_bg) > 0:
        # Clean the target strings (remove backticks, trim whitespace, make uppercase)
        df_bg["bg_clean"] = df_bg["target"].astype(str).str.replace("`", "").str.strip().str.upper()
        # Filter out invalid blood group texts like 'NORMAL'
        df_bg = df_bg[df_bg["bg_clean"] != "NORMAL"] 
        
        bg_counts = df_bg["bg_clean"].value_counts().reset_index()
        bg_counts.columns = ["Blood Group", "Count"]
        
        fig_bg = px.pie(
            bg_counts, names="Blood Group", values="Count", hole=0.4,
            title="Blood Group Distribution"
        )
        st.plotly_chart(fig_bg, use_container_width=True)
    else:
        st.info("No Blood Grouping data available for current filters.")

with col_b2:
    df_rh = df_filtered[df_filtered["test_name"] == "Rh Typing"].copy()
    if len(df_rh) > 0:
        df_rh["rh_clean"] = df_rh["target"].astype(str).str.strip().str.upper()
        rh_counts = df_rh["rh_clean"].value_counts().reset_index()
        rh_counts.columns = ["Rh Factor", "Count"]
        
        fig_rh = px.pie(
            rh_counts, names="Rh Factor", values="Count", hole=0.4,
            title="Rh Factor Distribution",
            color="Rh Factor",
            color_discrete_map={"POSITIVE": "#636efa", "NEGATIVE": "#ef553b"}
        )
        st.plotly_chart(fig_rh, use_container_width=True)
    else:
        st.info("No Rh Typing data available for current filters.")

# -------------------------
# Patient table
# -------------------------
st.subheader("📋 Patient Details")

display_cols = [
    "patient_id",
    "name",
    "age",
    "gender",
    "age_group",
    "test_name",
    "result_num",
    "ref_low",
    "ref_high",
    "target",
    "numeric_status",
]

# Only show columns that exist
display_cols = [c for c in display_cols if c in df_filtered.columns]
st.dataframe(df_filtered[display_cols], use_container_width=True)

# -------------------------
# Individual patient view
# -------------------------
st.subheader("🔍 Individual Patient View")

unique_ids = sorted(df_filtered["unique_ID"].dropna().astype(str).unique().tolist())
selected_uid = st.selectbox("Select Unique Patient ID", options=unique_ids)

patient_rows = df_filtered[df_filtered["unique_ID"].astype(str) == selected_uid]

if len(patient_rows) > 0:
    # Show summary for first test of this patient
    row = patient_rows.iloc[0]

    st.write(f"**Patient:** {row.get('name', 'N/A')} (ID: {selected_uid})")
    st.write(f"**Age:** {row.get('age', 'N/A')} | **Gender:** {row.get('gender', 'N/A')}")
    st.write(f"**Test:** {row.get('test_name', 'N/A')}")

    if "result_num" in row and not pd.isna(row["result_num"]):
        if "ref_low" in row and "ref_high" in row and not pd.isna(row["ref_low"]) and not pd.isna(row["ref_high"]):
            # Determine range boundaries for gauge chart
            r_val = row["result_num"]
            r_low = row["ref_low"]
            r_high = row["ref_high"]
            
            min_val = min(0, r_val, r_low)
            max_val = max(r_val, r_high)
            upper_bound = max_val + (max_val - min_val) * 0.2 if max_val != min_val else max_val + 10

            fig_patient = go.Figure(go.Indicator(
                mode="number+gauge",
                value=r_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"<b>{row['test_name']}</b>"},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [min_val, upper_bound]},
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': r_val
                    },
                    'steps': [
                        {'range': [min_val, r_low], 'color': "#ffcccb"},     # below normal
                        {'range': [r_low, r_high], 'color': "#e6f4ea"},      # normal range
                        {'range': [r_high, upper_bound], 'color': "#ffcccb"} # above normal
                    ],
                    'bar': {'color': "royalblue", 'thickness': 0.25}
                }
            ))
            fig_patient.update_layout(
                height=250, 
                margin={'t': 30, 'b': 30, 'l': 150, 'r': 50},
                title=f"Patient {selected_uid}: Result vs Reference Range"
            )
            st.plotly_chart(fig_patient, use_container_width=True)
        else:
            st.info("Reference range not available for this test.")
    else:
        st.info("Numeric result not available for this test.")

    st.write("All tests for this patient:")
    st.dataframe(patient_rows[display_cols], use_container_width=True)
else:
    st.info("No data for selected patient.")