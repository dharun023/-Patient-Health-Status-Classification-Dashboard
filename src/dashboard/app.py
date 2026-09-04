import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = project_root / "data" / "Feature engineered Dataset.xlsx"
FINAL_MERGED_PATH = project_root / "data" / "Final Merged data.xlsx"
SELF_PRESCRIBED = "Self-Prescribed (Patient)"
ABNORMAL_VALUES = {"low", "high", "++", "positive"}
DISPLAY_COLS = [
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


@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)


@st.cache_data
def load_final_merged():
    """Load raw data so null doctor_name records remain available."""
    data = pd.read_excel(FINAL_MERGED_PATH)
    required = {"doctor_name", "indication_final"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Final Merged data is missing required columns: {sorted(missing)}"
        )

    data["doctor_name_filled"] = data["doctor_name"].fillna(SELF_PRESCRIBED)
    indication = (
        data["indication_final"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
    )
    data["is_abnormal"] = indication.apply(
        lambda value: any(token in value for token in ABNORMAL_VALUES)
    )
    data["cohort"] = np.where(
        data["doctor_name_filled"].eq(SELF_PRESCRIBED),
        "Self-Prescribed",
        "Ordered by Doctor",
    )
    return data


try:
    df = load_data()
except Exception as exc:
    st.error(f"Error loading dashboard data: {exc}")
    st.stop()


st.set_page_config(
    page_title="Clinical Diagnostic Laboratory Dashboard",
    layout="wide",
)
st.title("🏥 Dashboard Based on Clinical Diagnostic Laboratory")
st.caption("Descriptive laboratory analytics dashboard.")

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Upload New Data (optional)")
uploaded = st.sidebar.file_uploader(
    "Upload CSV or Excel (raw or engineered)",
    type=["csv", "xlsx"],
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


def get_options(column):
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().unique().tolist())


test_options = get_options("test_name")
doctor_options = get_options("doctor_name")
gender_options = get_options("gender")
age_group_options = get_options("age_group")

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

mask = pd.Series(True, index=df.index)
for column, selected_values in (
    ("test_name", selected_tests),
    ("doctor_name", selected_doctors),
    ("gender", selected_genders),
    ("age_group", selected_age_groups),
):
    if column in df.columns:
        mask &= df[column].isin(selected_values)

df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# -------------------------
# Summary metrics
# -------------------------
if "target" in df_filtered.columns:
    abnormal_mask = (
        df_filtered["target"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(ABNORMAL_VALUES)
    )
else:
    abnormal_mask = pd.Series(False, index=df_filtered.index)

total_records = len(df_filtered)
unique_patients = (
    df_filtered["patient_id"].nunique()
    if "patient_id" in df_filtered.columns
    else 0
)
avg_result = (
    df_filtered.loc[df_filtered["result_num"].notna(), "result_num"].mean()
    if "result_num" in df_filtered.columns
    else np.nan
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{total_records:,}")
col2.metric("Total Patient ID", f"{unique_patients:,}")
col3.metric("% Abnormal", f"{abnormal_mask.mean() * 100:.1f}%")
col4.metric("Avg Result", f"{avg_result:.2f}" if pd.notna(avg_result) else "N/A")

# -------------------------
# Visualizations
# -------------------------
st.subheader(" Visualizations")
# -------------------------
# Distribution of Results
# -------------------------
st.markdown("### Distribution of Results")

# Convert raw target labels to consistent display labels.
TARGET_LABEL_MAP = {
    "normal": "Normal",
    "low": "Low",
    "high": "High",
    "positive": "Positive",
    "negative": "Negative",
    "nil": "Nil",
}

df_target_chart = df_filtered.copy()

# Normalize spacing and capitalization.
df_target_chart["target_clean"] = (
    df_target_chart["target"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Convert normalized values into clean display labels.
df_target_chart["target_display"] = df_target_chart[
    "target_clean"
].map(TARGET_LABEL_MAP)

# Keep only recognized values.
df_target_chart = df_target_chart[
    df_target_chart["target_display"].notna()
].copy()

TARGET_ORDER = [
    "Normal",
    "Low",
    "High",
    "Positive",
    "Negative",
    "Nil",
]

available_target_values = [
    value
    for value in TARGET_ORDER
    if value in df_target_chart["target_display"].unique()
]

if not available_target_values:
    st.warning(
        "No supported target categories were found after applying the current filters."
    )

    with st.expander("Show raw target values found in the dataset"):
        st.write(
            sorted(
                df_filtered["target"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
        )

else:
    selected_targets_for_pie = st.multiselect(
        "Choose target values to display",
        options=available_target_values,
        default=[],
        placeholder="Select one or more result categories",
        key="optimized_target_pie_filter",
    )

    if not selected_targets_for_pie:
        st.info("Select one or more target values to display the pie chart.")

    else:
        pie_data = df_target_chart[
            df_target_chart["target_display"].isin(
                selected_targets_for_pie
            )
        ].copy()

        target_counts = (
            pie_data["target_display"]
            .value_counts()
            .reindex(selected_targets_for_pie, fill_value=0)
            .rename_axis("Target")
            .reset_index(name="Count")
        )

        fig_pie = px.pie(
            target_counts,
            names="Target",
            values="Count",
            title="Distribution of Selected Target Values",
            color="Target",
            color_discrete_map={
                "Normal": "#2E8B57",
                "Low": "#1E88E5",
                "High": "#E53935",
                "Positive": "#D81B60",
                "Negative": "#43A047",
                "Nil": "#757575",
            },
            hole=0.30,
        )

        st.plotly_chart(
            fig_pie,
            width="stretch",
        )

st.subheader("Numeric Result Distribution")
if {"test_name", "result_num"}.issubset(df_filtered.columns):
    histogram_tests = sorted(df_filtered["test_name"].dropna().unique())
    test_for_hist = st.selectbox(
        "Select test for numeric histogram",
        options=histogram_tests,
        key="numeric_histogram_test",
    )

    numeric_subset = df_filtered[
        (df_filtered["test_name"] == test_for_hist)
        & df_filtered["result_num"].notna()
    ].copy()

    if not numeric_subset.empty:
        fig_hist = px.histogram(
            numeric_subset,
            x="result_num",
            color="target",
            title=f"{test_for_hist} - Result Distribution",
            labels={
                "result_num": "Test Result",
                "count": "Number of Records",
                "target": "Target Status",
            },
            barmode="overlay",
            opacity=0.70,
        )
        fig_hist.update_yaxes(
            rangemode="tozero",
            title="Number of Records",
            showline=True,
            linecolor="black",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2,
        )
        fig_hist.update_xaxes(showline=True, linecolor="black")
        fig_hist.update_layout(bargap=0.05, legend_title_text="Target Status")
        st.plotly_chart(fig_hist, width="stretch")
    else:
        st.info(f"No numeric results available for {test_for_hist}.")


if {"result_num", "gender"}.issubset(df_filtered.columns):
    numeric_gender_data = df_filtered[df_filtered["result_num"].notna()].copy()
    if not numeric_gender_data.empty:
        avg_by_gender = (
            numeric_gender_data.groupby("gender", as_index=False)["result_num"]
            .mean()
        )
        fig_gender_avg = px.bar(
            avg_by_gender,
            x="gender",
            y="result_num",
            title="Average Result by Gender",
            labels={"result_num": "Average Result"},
        )
        st.plotly_chart(fig_gender_avg, width="stretch")


st.subheader("### Abnormal Rate by Doctor")
if "doctor_name" in df_filtered.columns:
    df_doctor = df_filtered.copy()
    df_doctor["is_abnormal_target"] = (
        df_doctor["target"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(ABNORMAL_VALUES)
    )

    doctor_stats = (
        df_doctor.groupby("doctor_name", as_index=False, dropna=False)
        .agg(
            total=("is_abnormal_target", "count"),
            abnormal=("is_abnormal_target", "sum"),
        )
    )
    doctor_stats["abnormal_rate"] = (
        doctor_stats["abnormal"] / doctor_stats["total"] * 100
    )

    fig_doctor = px.bar(
        doctor_stats.sort_values("abnormal_rate", ascending=False),
        x="doctor_name",
        y="abnormal_rate",
        title="Abnormal Rate by Doctor",
        labels={"abnormal_rate": "Abnormal Rate (%)", "doctor_name": "Doctor"},
    )
    st.plotly_chart(fig_doctor, width="stretch")
else:
    doctor_stats = pd.DataFrame()
    st.info("Doctor data is not available.")

# -------------------------
# Advanced analytics
# -------------------------
st.subheader(" 🧠 Advanced Analytics & Insights")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("Overall Abnormality by Gender")
    if "gender" in df_filtered.columns:
        gender_data = df_filtered.copy()
        gender_data["is_abnormal"] = (
            gender_data["target"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(ABNORMAL_VALUES)
        )
        gender_stats = (
            gender_data.groupby("gender", as_index=False)
            .agg(
                total=("is_abnormal", "count"),
                abnormal=("is_abnormal", "sum"),
            )
        )
        gender_stats["abnormal_rate"] = (
            gender_stats["abnormal"] / gender_stats["total"] * 100
        )
        st.plotly_chart(
            px.bar(
                gender_stats,
                x="gender",
                y="abnormal_rate",
                color="gender",
                title="Abnormality Rate by Gender",
            ),
            width="stretch",
        )
    else:
        st.info("Gender data is not available.")

with col_g2:
    st.markdown("💡 Key Data Insights**")
    st.info(
        "The dashboard shows descriptive laboratory-result patterns. "
            )
    st.info("The Gender Gap: Female patients have a significantly higher overall rate of abnormal test results (33.7%) compared to male patients (28.7%).")
    st.info("The 'Sugar Spike': Sugar (P.P) tests result in an abnormal finding 86% of the time, making it the highest risk test category.")
    st.info("The Senior Vulnerability: Seniors represent the highest risk demographic with a 37% abnormality rate, closely followed by Teens (35%).")
           


colA, colB = st.columns(2)

with colA:
    # st.markdown("High-Risk Test Categories")
    if "test_group1" not in df_filtered.columns:
        st.warning("The `test_group1` column is not available in the filtered dataset.")
    else:
        tg = df_filtered[["test_group1", "target", "test_name"]].copy()
        tg["test_group1"] = tg["test_group1"].astype("string").str.strip()
        tg["test_name_clean"] = (
            tg["test_name"].astype("string").str.strip().str.lower()
        )

        # Rh Typing is a blood type category, not an abnormality category.
        tg = tg[tg["test_name_clean"] != "rh typing"].copy()
        tg["is_abnormal"] = (
            tg["target"].astype("string").str.strip().str.lower().isin(ABNORMAL_VALUES)
        )
        tg = tg[tg["test_group1"].notna() & (tg["test_group1"] != "")].copy()

        if tg.empty:
            st.info("No valid test-group data is available after the current filters.")
        else:
            tg_stats = (
                tg.groupby("test_group1", as_index=False)
                .agg(
                    total=("is_abnormal", "count"),
                    abnormal=("is_abnormal", "sum"),
                )
            )
            tg_stats["abnormal_rate"] = (
                tg_stats["abnormal"] / tg_stats["total"] * 100
            )
            top_tg = (
                tg_stats.sort_values(
                    ["abnormal_rate", "total"],
                    ascending=[False, False],
                )
                .head(10)
            )
            fig_tg = px.bar(
                top_tg,
                x="abnormal_rate",
                y="test_group1",
                orientation="h",
                text="abnormal_rate",
                hover_data=["total", "abnormal"],
                title="Top 10 Test Groups by Abnormality Rate",
                labels={
                    "test_group1": "Test Group",
                    "abnormal_rate": "Abnormal Rate (%)",
                },
                color="abnormal_rate",
                color_continuous_scale="Reds",
            )
            fig_tg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_tg.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_range=[
                    0,
                    min(110, max(100, top_tg["abnormal_rate"].max() + 10)),
                ],
            )
            st.plotly_chart(fig_tg, width="stretch")

with colB:
    # st.markdown("Categorical Outcomes Analysis")
    categorical_df = df_filtered.copy()
    categorical_df["target_clean"] = (
        categorical_df["target"].astype("string").str.strip()
    )
    standard_targets = {"low", "high", "normal", "++", "nil", "borderline"}
    categorical_targets = categorical_df[
        categorical_df["target_clean"].notna()
        & (categorical_df["target_clean"] != "")
        & ~categorical_df["target_clean"].str.lower().isin(standard_targets)
    ].copy()

    if categorical_targets.empty:
        st.info("No categorical outcomes are available under the current filters.")
    else:
        counts = (
            categorical_targets["target_clean"]
            .value_counts()
            .head(10)
            .rename_axis("target")
            .reset_index(name="count")
        )
        fig_cat = px.bar(
            counts,
            x="target",
            y="count",
            color="target",
            text="count",
            title="Top 10 Categorical Outcomes",
            labels={"target": "Outcome", "count": "Number of Records"},
        )
        fig_cat.update_traces(textposition="outside")
        fig_cat.update_layout(
            showlegend=False,
            xaxis_tickangle=-35,
            yaxis_rangemode="tozero",
        )
        st.plotly_chart(fig_cat, width="stretch")


colC, colD = st.columns(2)
with colC:
    # st.markdown("Doctor Diagnostic Yield")
    if not doctor_stats.empty:
        fig_doc_scatter = px.scatter(
            doctor_stats,
            x="total",
            y="abnormal_rate",
            hover_name="doctor_name",
            size="total",
            color="abnormal_rate",
            title="Testing Volume vs Abnormality Rate",
            labels={"total": "Total Tests", "abnormal_rate": "Abnormal Rate (%)"},
        )
        st.plotly_chart(fig_doc_scatter, width="stretch")
    else:
        st.info("Doctor data is not available.")

with colD:
    # st.markdown("**Age Group Health Profile**")
    if "age_group" in df_filtered.columns:
        age_data = df_filtered.copy()
        age_data["is_abnormal"] = (
            age_data["target"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(ABNORMAL_VALUES)
        )
        age_stats = (
            age_data.groupby("age_group", as_index=False)
            .agg(
                total=("is_abnormal", "count"),
                abnormal=("is_abnormal", "sum"),
            )
        )
        age_melt = age_stats.melt(
            "age_group",
            ["total", "abnormal"],
            var_name="Type",
            value_name="Count",
        )
        st.plotly_chart(
            px.bar(
                age_melt,
                x="age_group",
                y="Count",
                color="Type",
                barmode="group",
                title="Total vs Abnormal Results by Age Group",
            ),
            width="stretch",
        )
    else:
        st.info("Age group data is not available.")

# -------------------------
# Doctor vs Patient analysis
# -------------------------
# st.markdown("---")
st.subheader("🧑‍⚕️ Doctor vs Patient (Self-Prescribed) Analysis")
st.caption(
    "Self-prescribed records are rows whose original doctor_name was null. "
    "The rate is an observed abnormal-result rate, not clinical diagnostic accuracy."
)

try:
    df_merged = load_final_merged()
    merged_mask = pd.Series(True, index=df_merged.index)

    for column, selected_values in (
        ("test_name", selected_tests),
        ("gender", selected_genders),
        ("age_group", selected_age_groups),
    ):
        if column in df_merged.columns:
            merged_mask &= df_merged[column].isin(selected_values)

    df_vis = df_merged.loc[merged_mask].copy()

    if df_vis.empty:
        st.info("No raw records match the current filters.")
    else:
        st.markdown("1. Self-Prescribed Outcomes")
        self_df = df_vis[df_vis["cohort"] == "Self-Prescribed"]
        if not self_df.empty:
            self_counts = (
                self_df["is_abnormal"]
                .map({True: "Abnormal", False: "Normal"})
                .value_counts()
                .reindex(["Normal", "Abnormal"], fill_value=0)
                .rename_axis("Outcome")
                .reset_index(name="Count")
            )
            st.plotly_chart(
                px.bar(
                    self_counts,
                    x="Outcome",
                    y="Count",
                    color="Outcome",
                    title="Self-Prescribed: Normal vs Abnormal",
                    color_discrete_map={
                        "Normal": "#4caf50",
                        "Abnormal": "#e53935",
                    },
                ),
                width="stretch",
            )
        else:
            st.info("No self-prescribed records found.")

        st.markdown("2. Doctor Predictive Performance")
        doctor_df = df_vis[df_vis["cohort"] == "Ordered by Doctor"]
        if not doctor_df.empty:
            doctor_perf = (
                doctor_df.groupby("doctor_name_filled", as_index=False)
                .agg(
                    total_tests=("is_abnormal", "count"),
                    abnormal_tests=("is_abnormal", "sum"),
                )
            )
            doctor_perf["predictive_ratio"] = (
                doctor_perf["abnormal_tests"] / doctor_perf["total_tests"]
            )
            doctor_perf["abnormality_rate"] = doctor_perf["predictive_ratio"] * 100

            min_volume = st.number_input(
                "Minimum tests per doctor",
                min_value=1,
                value=5,
                step=1,
            )
            ranked = doctor_perf[
                doctor_perf["total_tests"] >= min_volume
            ].sort_values("predictive_ratio", ascending=False)

            if ranked.empty:
                st.info("No doctors meet the selected minimum volume.")
            else:
                top_n = st.slider(
                    "Doctors to display",
                    min_value=1,
                    max_value=min(50, len(ranked)),
                    value=min(10, len(ranked)),
                )
                top = ranked.head(top_n)
                st.plotly_chart(
                    px.bar(
                        top,
                        x="doctor_name_filled",
                        y="abnormality_rate",
                        hover_data=[
                            "total_tests",
                            "abnormal_tests",
                            "predictive_ratio",
                        ],
                        title="Top Doctors by Abnormality Rate",
                        labels={
                            "doctor_name_filled": "Doctor",
                            "abnormality_rate": "Abnormality Rate (%)",
                        },
                        color="abnormality_rate",
                        color_continuous_scale="Turbo",
                    ),
                    width="stretch",
                )
                st.dataframe(top, width="stretch", hide_index=True)
        else:
            st.info("No doctor-ordered records found.")

        st.markdown("3. Ultimate Comparison")
        cohort_stats = (
            df_vis.groupby("cohort", as_index=False)
            .agg(
                total_tests=("is_abnormal", "count"),
                abnormal_tests=("is_abnormal", "sum"),
            )
        )
        cohort_stats["abnormality_rate"] = (
            cohort_stats["abnormal_tests"] / cohort_stats["total_tests"] * 100
        )
        fig_comp = px.bar(
            cohort_stats,
            x="cohort",
            y="abnormality_rate",
            text="abnormality_rate",
            title="Abnormality Detection Rate: Doctor vs Patient",
            labels={"cohort": "Cohort", "abnormality_rate": "Abnormality Rate (%)"},
            color="cohort",
            color_discrete_map={
                "Ordered by Doctor": "#1976d2",
                "Self-Prescribed": "#ff9800",
            },
        )
        fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_comp, width="stretch")
except Exception as exc:
    st.error(f"Could not load Doctor vs Patient analysis: {exc}")

# -------------------------
# Blood group analysis
# -------------------------
# st.markdown("---")
st.subheader("🩸 Blood Group & Rh Typing Analysis")
col_b1, col_b2 = st.columns(2)

with col_b1:
    df_bg = df_filtered[df_filtered["test_name"] == "Grouping"].copy()
    if not df_bg.empty:
        df_bg["bg_clean"] = (
            df_bg["target"]
            .astype(str)
            .str.replace("`", "")
            .str.strip()
            .str.upper()
        )
        df_bg = df_bg[df_bg["bg_clean"] != "NORMAL"]
        bg_counts = (
            df_bg["bg_clean"]
            .value_counts()
            .rename_axis("Blood Group")
            .reset_index(name="Count")
        )
        st.plotly_chart(
            px.pie(
                bg_counts,
                names="Blood Group",
                values="Count",
                hole=0.4,
                title="Blood Group Distribution",
            ),
            width="stretch",
        )
    else:
        st.info("No Blood Grouping data available for current filters.")

with col_b2:
    df_rh = df_filtered[df_filtered["test_name"] == "Rh Typing"].copy()
    if not df_rh.empty:
        df_rh["rh_clean"] = df_rh["target"].astype(str).str.strip().str.upper()
        rh_counts = (
            df_rh["rh_clean"]
            .value_counts()
            .rename_axis("Rh Factor")
            .reset_index(name="Count")
        )
        st.plotly_chart(
            px.pie(
                rh_counts,
                names="Rh Factor",
                values="Count",
                hole=0.4,
                title="Rh Factor Distribution",
            ),
            width="stretch",
        )
    else:
        st.info("No Rh Typing data available for current filters.")

# -------------------------
# Individual patient view
# -------------------------
st.subheader("🔍 Individual Patient View")

if "patient_id" not in df_filtered.columns:
    st.info("Patient ID is not available in the filtered data.")
else:
    patient_ids = sorted(df_filtered["patient_id"].dropna().unique().tolist())
    selected_pid = st.selectbox(
        "Select Patient ID",
        options=patient_ids,
        key="patient_id_select",
    )

    patient_rows = df_filtered[df_filtered["patient_id"] == selected_pid].copy()

    if patient_rows.empty:
        st.info("No data for selected patient.")
    else:
        patient_tests = sorted(
            patient_rows["test_name"].dropna().astype(str).unique().tolist()
        )
        selected_test = st.selectbox(
            "Select Test",
            options=patient_tests,
            key="patient_test_select",
        )

        selected_test_rows = patient_rows[
            patient_rows["test_name"].astype(str) == selected_test
        ].copy()

        if selected_test_rows.empty:
            st.info("No result is available for the selected test.")
        else:
            row = selected_test_rows.iloc[0]
            st.write(f"Patient: {row.get('name', 'N/A')} (ID: {selected_pid})")
            st.write(
                f"Age: {row.get('age', 'N/A')} | "
                f"Gender: {row.get('gender', 'N/A')}"
            )
            st.write(f"Test: {selected_test}")
            st.write(f"Target / Status: {row.get('target', 'N/A')}")

            has_numeric = (
                pd.notna(row.get("result_num"))
                and pd.notna(row.get("ref_low"))
                and pd.notna(row.get("ref_high"))
            )

            if has_numeric:
                fig_patient = go.Figure(
                    go.Bar(
                        x=["Result", "Ref Low", "Ref High"],
                        y=[
                            row["result_num"],
                            row["ref_low"],
                            row["ref_high"],
                        ],
                        marker_color=["orange", "royalblue", "red"],
                        text=[
                            str(row["result_num"]),
                            str(row["ref_low"]),
                            str(row["ref_high"]),
                        ],
                        textposition="auto",
                    )
                )
                fig_patient.update_layout(
                    title=(
                        f"Patient {selected_pid}: Result vs Reference Range "
                        f"({selected_test})"
                    ),
                    xaxis_title="Measurement",
                    yaxis_title="Value",
                    showlegend=False,
                )
                st.plotly_chart(fig_patient, width="stretch")
            else:
                st.info("Graph only for Numeric values")

            selected_display_cols = [
                column
                for column in DISPLAY_COLS
                if column in selected_test_rows.columns
            ]
            st.write("Selected test details:")
            st.dataframe(
                selected_test_rows[selected_display_cols],
                width="stretch",
                hide_index=True,
            )
