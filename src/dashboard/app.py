import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = project_root / "data" / "Feature engineered Dataset.xlsx"
FINAL_MERGED_PATH = project_root / "data" / "Final Merged data.xlsx"
ID_REPORT_PATH = project_root / "data" / "SRD Report with unique ID for patient.xlsx"
MODEL_PATH = project_root / "models" / "patient_status_model.pkl"
SELF_PRESCRIBED = "Self-Prescribed (Patient)"
ABNORMAL_VALUES = {"low", "high", "++"}


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    if "patient_id" in df.columns and ID_REPORT_PATH.exists():
        try:
            id_df = pd.read_excel(ID_REPORT_PATH)
            required = {"patient_id", "unique_ID"}
            if required.issubset(id_df.columns):
                mapping = (
                    id_df[["patient_id", "unique_ID"]]
                    .dropna()
                    .drop_duplicates("patient_id")
                )
                df = df.merge(mapping, on="patient_id", how="left")
        except Exception as exc:
            st.warning(f"Could not load patient ID mapping: {exc}")
    if "unique_ID" not in df.columns and "patient_id" in df.columns:
        df["unique_ID"] = df["patient_id"]
    elif "patient_id" in df.columns:
        df["unique_ID"] = df["unique_ID"].fillna(df["patient_id"])
    return df


@st.cache_data
def load_final_merged():
    """Load the raw dataset, retaining records with a null doctor_name."""
    df = pd.read_excel(FINAL_MERGED_PATH)
    required = {"doctor_name", "indication_final"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Final Merged data is missing columns: {sorted(missing)}")

    df["doctor_name_filled"] = df["doctor_name"].fillna(SELF_PRESCRIBED)
    indication = df["indication_final"].astype("string").str.strip().str.lower()
    df["is_abnormal"] = indication.fillna("").apply(
        lambda value: any(token in value for token in ABNORMAL_VALUES)
    )
    df["cohort"] = np.where(
        df["doctor_name_filled"].eq(SELF_PRESCRIBED),
        "Self-Prescribed",
        "Ordered by Doctor",
    )
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except Exception as exc:
    st.error(f"Error loading data or model: {exc}")
    st.stop()

st.set_page_config(page_title="Patient Health Dashboard", layout="wide")
st.title("🏥 Patient Health Status Classification + Dashboard")

st.sidebar.header("Upload New Data (optional)")
uploaded = st.sidebar.file_uploader(
    "Upload CSV or Excel (raw or engineered)", type=["csv", "xlsx"]
)
if uploaded is not None:
    try:
        uploaded_df = (
            pd.read_csv(uploaded)
            if uploaded.name.lower().endswith(".csv")
            else pd.read_excel(uploaded)
        )
        df = pd.concat([df, uploaded_df], ignore_index=True)
        st.sidebar.success("New data appended successfully.")
    except Exception as exc:
        st.sidebar.error(f"Error loading uploaded file: {exc}")

st.sidebar.header("Filters")

def options(column):
    return sorted(df[column].dropna().unique().tolist()) if column in df else []


test_options = options("test_name")
doctor_options = options("doctor_name")
gender_options = options("gender")
age_group_options = options("age_group")

selected_tests = st.sidebar.multiselect("Test", test_options, default=test_options)
selected_doctors = st.sidebar.multiselect("Doctor", doctor_options, default=doctor_options)
selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
selected_age_groups = st.sidebar.multiselect(
    "Age Group", age_group_options, default=age_group_options
)

mask = pd.Series(True, index=df.index)
for column, values in (
    ("test_name", selected_tests),
    ("doctor_name", selected_doctors),
    ("gender", selected_genders),
    ("age_group", selected_age_groups),
):
    if column in df.columns:
        mask &= df[column].isin(values)

df_filtered = df.loc[mask].copy()
if df_filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

abnormal_mask = df_filtered["target"].astype("string").str.strip().str.lower().isin(
    ABNORMAL_VALUES
)
total_records = len(df_filtered)
unique_patients = df_filtered["patient_id"].nunique() if "patient_id" in df_filtered else 0
abnormal_pct = abnormal_mask.mean() * 100
numeric_df = df_filtered[df_filtered["result_num"].notna()]
avg_result = numeric_df["result_num"].mean() if not numeric_df.empty else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{total_records:,}")
col2.metric("Unique Patients", f"{unique_patients:,}")
col3.metric("% Abnormal (target)", f"{abnormal_pct:.1f}%")
col4.metric("Avg Result (numeric)", f"{avg_result:.2f}" if not np.isnan(avg_result) else "N/A")

# st.subheader("📈 Visualizations")
# st.plotly_chart(
#     px.pie(df_filtered, names="target", title="Distribution of Target (indication_final)"),
#     use_container_width=True,
# )
# -------------------------
# Visualizations
# -------------------------
st.subheader("📈 Visualizations")

# 1. Target distribution (pie) with user-selectable categories
st.markdown("### Target Distribution")

target_options = sorted(df_filtered["target"].dropna().astype(str).unique().tolist())
selected_targets = st.multiselect(
    "Select target categories to display in the pie chart",
    options=target_options,
    default=[],  # all unticked by default
    key="target_pie_select",
)

if selected_targets:
    df_pie = df_filtered[df_filtered["target"].astype(str).isin(selected_targets)]
    if not df_pie.empty:
        fig_pie = px.pie(
            df_pie,
            names="target",
            title="Distribution of Target (indication_final)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data available for the selected target categories.")
else:
    st.info("Select one or more target categories to show the pie chart.")
    
st.subheader("Numeric Result Distribution")
test_for_hist = st.selectbox(
    "Select test for numeric histogram",
    sorted(df_filtered["test_name"].dropna().unique()),
)
subset_test = df_filtered[df_filtered["test_name"] == test_for_hist]
numeric_subset = subset_test[subset_test["result_num"].notna()]
if not numeric_subset.empty:
    st.plotly_chart(
        px.histogram(
            numeric_subset,
            x="result_num",
            color="target",
            title=f"{test_for_hist} - Result Distribution by Target",
        ),
        use_container_width=True,
    )
else:
    st.info(f"No numeric results available for {test_for_hist}.")

if not numeric_subset.empty:
    avg_by_gender = numeric_subset.groupby("gender", as_index=False)["result_num"].mean()
    st.plotly_chart(
        px.bar(
            avg_by_gender,
            x="gender",
            y="result_num",
            title=f"{test_for_hist} - Average Result by Gender",
            labels={"result_num": "Average result"},
        ),
        use_container_width=True,
    )

st.subheader("Abnormal Rate by Doctor")
df_doctor = df_filtered.copy()
df_doctor["is_abnormal_target"] = df_doctor["target"].astype("string").str.strip().str.lower().isin(
    ABNORMAL_VALUES
)
doctor_stats = (
    df_doctor.groupby("doctor_name", as_index=False, dropna=False)
    .agg(total=("is_abnormal_target", "count"), abnormal=("is_abnormal_target", "sum"))
)
doctor_stats["abnormal_rate"] = doctor_stats["abnormal"] / doctor_stats["total"] * 100
st.plotly_chart(
    px.bar(
        doctor_stats.sort_values("abnormal_rate", ascending=False),
        x="doctor_name",
        y="abnormal_rate",
        title="Abnormal Rate by Doctor",
        labels={"abnormal_rate": "Abnormal Rate (%)", "doctor_name": "Doctor"},
    ),
    use_container_width=True,
)

st.subheader("🧠 Advanced Analytics & Insights")
col_g1, col_g2 = st.columns(2)
with col_g1:
    st.markdown("**Overall Abnormality by Gender**")
    gender_data = df_filtered.copy()
    gender_data["is_abnormal"] = gender_data["target"].astype("string").str.strip().str.lower().isin(ABNORMAL_VALUES)
    gender_stats = gender_data.groupby("gender", as_index=False).agg(
        total=("is_abnormal", "count"), abnormal=("is_abnormal", "sum")
    )
    gender_stats["abnormal_rate"] = gender_stats["abnormal"] / gender_stats["total"] * 100
    st.plotly_chart(
        px.bar(gender_stats, x="gender", y="abnormal_rate", color="gender", title="Abnormality Rate by Gender"),
        use_container_width=True,
    )
with col_g2:
    st.markdown("**💡 Key Data Insights**")
    # st.info("Use the charts below to compare abnormality rates. Rates describe observed outcomes, not clinical diagnostic accuracy.")
    st.info("**The 'Sugar Spike'**: Sugar (P.P) tests result in an abnormal finding 86% of the time, making it the highest risk test category.")
    st.info("**The Senior Vulnerability**: Seniors represent the highest risk demographic with a 37% abnormality rate, closely followed by Teens (35%).")
    st.info("**The Gender Gap**: Female patients have a significantly higher overall rate of abnormal test results (33.7%) compared to male patients (28.7%).")

colA, colB = st.columns(2)
with colA:
    st.markdown("**High-Risk Test Categories**")
    if "test_group1" in df_filtered:
        tg = df_filtered.copy()
        tg["is_abnormal"] = tg["target"].astype("string").str.strip().str.lower().isin(ABNORMAL_VALUES)
        tg_stats = tg.groupby("test_group1", as_index=False).agg(total=("is_abnormal", "count"), abnormal=("is_abnormal", "sum"))
        tg_stats = tg_stats[tg_stats["total"] >= 5]
        if not tg_stats.empty:
            tg_stats["abnormal_rate"] = tg_stats["abnormal"] / tg_stats["total"] * 100
            top_tg = tg_stats.sort_values("abnormal_rate", ascending=False).head(10)
            st.plotly_chart(px.bar(top_tg, x="abnormal_rate", y="test_group1", orientation="h", title="Top Test Groups by Abnormality Rate"), use_container_width=True)
with colB:
    st.markdown("**Categorical Outcomes Analysis**")
    categorical_targets = df_filtered[~df_filtered["target"].isin(["Low", "High", "Normal", "++", "Nil", "Borderline"])]
    if not categorical_targets.empty:
        counts = categorical_targets["target"].value_counts().rename_axis("target").reset_index(name="count")
        st.plotly_chart(px.bar(counts.head(10), x="target", y="count", color="target", title="Top Categorical Outcomes"), use_container_width=True)

colC, colD = st.columns(2)
with colC:
    st.markdown("**Doctor Diagnostic Yield**")
    if not doctor_stats.empty:
        fig = px.scatter(doctor_stats, x="total", y="abnormal_rate", hover_name="doctor_name", size="total", color="abnormal_rate", title="Testing Volume vs Abnormality Rate", labels={"total": "Total Tests", "abnormal_rate": "Abnormal Rate (%)"})
        st.plotly_chart(fig, use_container_width=True)
with colD:
    st.markdown("**Age Group Health Profile**")
    if "age_group" in df_filtered:
        age_data = df_filtered.copy()
        age_data["is_abnormal"] = age_data["target"].astype("string").str.strip().str.lower().isin(ABNORMAL_VALUES)
        age_stats = age_data.groupby("age_group", as_index=False).agg(total=("is_abnormal", "count"), abnormal=("is_abnormal", "sum"))
        age_melt = age_stats.melt("age_group", ["total", "abnormal"], var_name="Type", value_name="Count")
        st.plotly_chart(px.bar(age_melt, x="age_group", y="Count", color="Type", barmode="group", title="Total vs Abnormal Results by Age Group"), use_container_width=True)

# -------------------------
# Doctor vs Patient Prediction Analysis
# -------------------------
st.markdown("---")
st.subheader("🧑‍⚕️ Doctor vs Patient (Self-Prescribed) Analysis")
st.caption("Self-prescribed records are rows whose original doctor_name was null. The rate is an observed abnormal-result rate, not a clinical measure of diagnostic accuracy.")

try:
    df_merged = load_final_merged()
    merged_mask = pd.Series(True, index=df_merged.index)
    for column, values in (("test_name", selected_tests), ("gender", selected_genders), ("age_group", selected_age_groups)):
        if column in df_merged.columns:
            merged_mask &= df_merged[column].isin(values)
    df_vis = df_merged.loc[merged_mask].copy()

    if df_vis.empty:
        st.info("No raw records match the current test, gender, and age-group filters.")
    else:
        # Visualization 1
        st.markdown("### 1. Self-Prescribed Outcomes")
        self_df = df_vis[df_vis["cohort"] == "Self-Prescribed"]
        if not self_df.empty:
            self_counts = self_df["is_abnormal"].map({True: "Abnormal", False: "Normal"}).value_counts()
            self_counts = self_counts.reindex(["Normal", "Abnormal"], fill_value=0).rename_axis("Outcome").reset_index(name="Count")
            fig_self = px.bar(self_counts, x="Outcome", y="Count", color="Outcome", title="Self-Prescribed: Normal vs Abnormal", color_discrete_map={"Normal": "#4caf50", "Abnormal": "#e53935"})
            st.plotly_chart(fig_self, use_container_width=True)
        else:
            st.info("No self-prescribed records found.")

        # Visualization 2
        st.markdown("### 2. Doctor Predictive Performance")
        doctor_df = df_vis[df_vis["cohort"] == "Ordered by Doctor"]
        if not doctor_df.empty:
            doctor_perf = doctor_df.groupby("doctor_name_filled", as_index=False).agg(
                total_tests=("is_abnormal", "count"), abnormal_tests=("is_abnormal", "sum")
            )
            doctor_perf["predictive_ratio"] = doctor_perf["abnormal_tests"] / doctor_perf["total_tests"]
            doctor_perf["abnormality_rate"] = doctor_perf["predictive_ratio"] * 100
            min_volume = st.number_input("Minimum tests per doctor", min_value=1, value=5, step=1)
            ranked = doctor_perf[doctor_perf["total_tests"] >= min_volume].sort_values("predictive_ratio", ascending=False)
            if ranked.empty:
                st.info("No doctors meet the selected minimum volume.")
            else:
                top_n = st.slider("Doctors to display", 5, min(50, len(ranked)), min(10, len(ranked))) if len(ranked) >= 5 else 1
                top = ranked.head(top_n)
                fig_doc = px.bar(top, x="doctor_name_filled", y="abnormality_rate", hover_data=["total_tests", "abnormal_tests", "predictive_ratio"], title="Top Doctors by Abnormality Rate", labels={"doctor_name_filled": "Doctor", "abnormality_rate": "Abnormality Rate (%)"}, color="abnormality_rate", color_continuous_scale="Turbo")
                st.plotly_chart(fig_doc, use_container_width=True)
                st.dataframe(top.rename(columns={"doctor_name_filled": "Doctor", "total_tests": "Total Tests", "abnormal_tests": "Abnormal Tests", "predictive_ratio": "Predictive Ratio", "abnormality_rate": "Abnormality Rate (%)"}), use_container_width=True, hide_index=True)
        else:
            st.info("No doctor-ordered records found.")

        # Visualization 3
        st.markdown("### 3. Ultimate Comparison")
        cohort_stats = df_vis.groupby("cohort", as_index=False).agg(
            total_tests=("is_abnormal", "count"), abnormal_tests=("is_abnormal", "sum")
        )
        cohort_stats["abnormality_rate"] = cohort_stats["abnormal_tests"] / cohort_stats["total_tests"] * 100
        cohort_stats["Cohort"] = cohort_stats["cohort"].map({"Ordered by Doctor": "Ordered by Doctor", "Self-Prescribed": "Self-Prescribed"})
        fig_comp = px.bar(cohort_stats, x="Cohort", y="abnormality_rate", text="abnormality_rate", title="Abnormality Detection Rate: Doctor vs Patient", labels={"abnormality_rate": "Abnormality Rate (%)"}, color="Cohort", color_discrete_map={"Ordered by Doctor": "#1976d2", "Self-Prescribed": "#ff9800"})
        fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(cohort_stats[["Cohort", "total_tests", "abnormal_tests", "abnormality_rate"]].rename(columns={"total_tests": "Total Tests", "abnormal_tests": "Abnormal Tests", "abnormality_rate": "Abnormality Rate (%)"}), use_container_width=True, hide_index=True)
except Exception as exc:
    st.error(f"Could not load Doctor vs Patient analysis: {exc}")

st.markdown("---")
st.subheader("🩸 Blood Group & Rh Typing Analysis")
col_b1, col_b2 = st.columns(2)
with col_b1:
    df_bg = df_filtered[df_filtered["test_name"] == "Grouping"].copy()
    if not df_bg.empty:
        df_bg["bg_clean"] = df_bg["target"].astype(str).str.replace("`", "").str.strip().str.upper()
        df_bg = df_bg[df_bg["bg_clean"] != "NORMAL"]
        counts = df_bg["bg_clean"].value_counts().rename_axis("Blood Group").reset_index(name="Count")
        st.plotly_chart(px.pie(counts, names="Blood Group", values="Count", hole=0.4, title="Blood Group Distribution"), use_container_width=True)
    else:
        st.info("No Blood Grouping data available for current filters.")
with col_b2:
    df_rh = df_filtered[df_filtered["test_name"] == "Rh Typing"].copy()
    if not df_rh.empty:
        df_rh["rh_clean"] = df_rh["target"].astype(str).str.strip().str.upper()
        counts = df_rh["rh_clean"].value_counts().rename_axis("Rh Factor").reset_index(name="Count")
        st.plotly_chart(px.pie(counts, names="Rh Factor", values="Count", hole=0.4, title="Rh Factor Distribution"), use_container_width=True)
    else:
        st.info("No Rh Typing data available for current filters.")

st.subheader("📋 Patient Details")
display_cols = [c for c in ["patient_id", "name", "age", "gender", "age_group", "test_name", "result_num", "ref_low", "ref_high", "target", "numeric_status"] if c in df_filtered.columns]
st.dataframe(df_filtered[display_cols], use_container_width=True)

# st.subheader("🔍 Individual Patient View")
# if "unique_ID" in df_filtered:
#     unique_ids = sorted(df_filtered["unique_ID"].dropna().astype(str).unique())
#     selected_uid = st.selectbox("Select Unique Patient ID", unique_ids)
#     patient_rows = df_filtered[df_filtered["unique_ID"].astype(str) == selected_uid]
#     if not patient_rows.empty:
#         row = patient_rows.iloc[0]
#         st.write(f"**Patient:** {row.get('name', 'N/A')} (ID: {selected_uid})")
#         st.write(f"**Age:** {row.get('age', 'N/A')} | **Gender:** {row.get('gender', 'N/A')}")
#         st.write(f"**Test:** {row.get('test_name', 'N/A')}")
#         if pd.notna(row.get("result_num")) and pd.notna(row.get("ref_low")) and pd.notna(row.get("ref_high")):
#             r_val, r_low, r_high = row["result_num"], row["ref_low"], row["ref_high"]
#             min_val, max_val = min(0, r_val, r_low), max(r_val, r_high)
#             upper_bound = max_val + (max_val - min_val) * 0.2 if max_val != min_val else max_val + 10
#             fig_patient = go.Figure(go.Indicator(mode="number+gauge", value=r_val, domain={"x": [0, 1], "y": [0, 1]}, title={"text": str(row["test_name"])}, gauge={"shape": "bullet", "axis": {"range": [min_val, upper_bound]}, "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": r_val}, "steps": [{"range": [min_val, r_low], "color": "#ffcccb"}, {"range": [r_low, r_high], "color": "#e6f4ea"}, {"range": [r_high, upper_bound], "color": "#ffcccb"}], "bar": {"color": "royalblue", "thickness": 0.25}}))
#             fig_patient.update_layout(height=250, margin={"t": 30, "b": 30, "l": 150, "r": 50})
#             st.plotly_chart(fig_patient, use_container_width=True)
#         else:
#             st.info("Numeric result or reference range not available for this test.")
#         st.write("All tests for this patient:")
#         st.dataframe(patient_rows[display_cols], use_container_width=True)
# else:
#     st.info("Unique patient ID is not available.")

# -------------------------
# Individual patient view
# -------------------------
st.subheader("🔍 Individual Patient View")

if "unique_ID" not in df_filtered.columns:
    st.info("Unique patient ID is not available.")
else:
    unique_ids = sorted(df_filtered["unique_ID"].dropna().astype(str).unique().tolist())
    selected_uid = st.selectbox("Select Unique Patient ID", options=unique_ids)

    patient_rows = df_filtered[df_filtered["unique_ID"].astype(str) == selected_uid].copy()

    if patient_rows.empty:
        st.info("No data for selected patient.")
    else:
        # Let user choose which test to view for this patient
        patient_tests = sorted(patient_rows["test_name"].dropna().unique().tolist())
        if not patient_tests:
            st.info("No test names available for this patient.")
        else:
            selected_test = st.selectbox(
                "Select Test",
                options=patient_tests,
                key=f"patient_test_select_{selected_uid}",
            )

            test_rows = patient_rows[patient_rows["test_name"] == selected_test]
            if test_rows.empty:
                st.info("No data for the selected test.")
            else:
                row = test_rows.iloc[0]

                st.write(f"**Patient:** {row.get('name', 'N/A')} (ID: {selected_uid})")
                st.write(f"**Age:** {row.get('age', 'N/A')} | **Gender:** {row.get('gender', 'N/A')}")
                st.write(f"**Test:** {row.get('test_name', 'N/A')}")
                st.write(f"**Result:** {row.get('result_num', 'N/A')} | **Reference Range:** {row.get('ref_low', 'N/A')} - {row.get('ref_high', 'N/A')}")

                # Show gauge only if numeric result and reference range exist
                if (
                    "result_num" in row
                    and pd.notna(row["result_num"])
                    and "ref_low" in row
                    and "ref_high" in row
                    and pd.notna(row["ref_low"])
                    and pd.notna(row["ref_high"])
                ):
                    r_val = row["result_num"]
                    r_low = row["ref_low"]
                    r_high = row["ref_high"]

                    min_val = min(0, r_val, r_low)
                    max_val = max(r_val, r_high)
                    upper_bound = max_val + (max_val - min_val) * 0.2 if max_val != min_val else max_val + 10

                    fig_patient = go.Figure(
                        go.Indicator(
                            mode="number+gauge",
                            value=r_val,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": f"{row['test_name']}"},
                            gauge={
                                "shape": "bullet",
                                "axis": {"range": [min_val, upper_bound]},
                                "threshold": {
                                    "line": {"color": "black", "width": 2},
                                    "thickness": 0.75,
                                    "value": r_val,
                                },
                                "steps": [
                                    {"range": [min_val, r_low], "color": "#ffcccb"},  # below normal
                                    {"range": [r_low, r_high], "color": "#e6f4ea"},  # normal range
                                    {"range": [r_high, upper_bound], "color": "#ffcccb"},  # above normal
                                ],
                                "bar": {"color": "royalblue", "thickness": 0.25},
                            },
                        )
                    )
                    fig_patient.update_layout(
                        height=250,
                        margin={"t": 30, "b": 30, "l": 150, "r": 50},
                    )
                    st.plotly_chart(fig_patient, use_container_width=True)
                else:
                    st.write("**Graph only for Numeric values**")

                st.write("All tests for this patient:")
                display_cols = [
                    c
                    for c in [
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
                    if c in patient_rows.columns
                ]
                st.dataframe(patient_rows[display_cols], use_container_width=True)
