import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "Telco-Customer-Churn-Processed.csv"
)

ORIGINAL_DATA_PATH = (
    BASE_DIR
    / "data"
    / "Telco-Customer-Churn.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "churn_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    if DATA_PATH.exists():

        df = pd.read_csv(DATA_PATH)

    elif ORIGINAL_DATA_PATH.exists():

        df = pd.read_csv(ORIGINAL_DATA_PATH)

    else:

        st.error(
            "❌ Dataset not found. "
            "Please check the data folder."
        )

        st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading dataset: {e}"
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df.columns = df.columns.str.strip()


if "TotalCharges" in df.columns:

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


if "MonthlyCharges" in df.columns:

    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    )


if "tenure" in df.columns:

    df["tenure"] = pd.to_numeric(
        df["tenure"],
        errors="coerce"
    )


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Intelligence")

st.markdown(
    """
    **Telco Customer Churn Analysis & Prediction**

    Analyze customer behavior and predict the probability
    that a customer may leave the company.
    """
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Dashboard Filters")

filtered_df = df.copy()


if "Contract" in df.columns:

    contract_values = sorted(
        df["Contract"].dropna().unique().tolist()
    )

    selected_contracts = st.sidebar.multiselect(
        "Contract Type",
        contract_values,
        default=contract_values
    )

    if selected_contracts:

        filtered_df = filtered_df[
            filtered_df["Contract"].isin(
                selected_contracts
            )
        ]


if "InternetService" in df.columns:

    service_values = sorted(
        df["InternetService"].dropna().unique().tolist()
    )

    selected_services = st.sidebar.multiselect(
        "Internet Service",
        service_values,
        default=service_values
    )

    if selected_services:

        filtered_df = filtered_df[
            filtered_df["InternetService"].isin(
                selected_services
            )
        ]


if "Churn" in df.columns:

    churn_values = sorted(
        df["Churn"].dropna().unique().tolist()
    )

    selected_churn = st.sidebar.multiselect(
        "Churn Status",
        churn_values,
        default=churn_values
    )

    if selected_churn:

        filtered_df = filtered_df[
            filtered_df["Churn"].isin(
                selected_churn
            )
        ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(filtered_df)


if "Churn" in filtered_df.columns:

    churned_customers = (
        filtered_df["Churn"]
        .astype(str)
        .str.strip()
        .eq("Yes")
        .sum()
    )

else:

    churned_customers = 0


if total_customers > 0:

    churn_rate = (
        churned_customers
        / total_customers
    ) * 100

else:

    churn_rate = 0


if "MonthlyCharges" in filtered_df.columns:

    avg_monthly_charges = (
        filtered_df["MonthlyCharges"].mean()
    )

else:

    avg_monthly_charges = 0


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )


with col3:

    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )


with col4:

    st.metric(
        "Avg Monthly Charges",
        f"${avg_monthly_charges:,.2f}"
    )


st.divider()


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

st.subheader("📊 Customer Churn Overview")

col1, col2 = st.columns(2)


with col1:

    churn_data = (
        filtered_df["Churn"]
        .value_counts()
        .reset_index()
    )

    churn_data.columns = [
        "Churn",
        "Customers"
    ]

    fig = px.pie(
        churn_data,
        names="Churn",
        values="Customers",
        title="Customer Churn Distribution",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    if "Contract" in filtered_df.columns:

        contract_data = pd.crosstab(
            filtered_df["Contract"],
            filtered_df["Churn"],
            normalize="index"
        ) * 100

        contract_data = contract_data.reset_index()

        if "Yes" in contract_data.columns:

            fig = px.bar(
                contract_data,
                x="Contract",
                y="Yes",
                title="Churn Rate by Contract Type",
                labels={
                    "Yes": "Churn Rate (%)"
                },
                text_auto=".1f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# TENURE ANALYSIS
# ============================================================

st.subheader("📈 Customer Tenure Analysis")

if "tenure" in filtered_df.columns:

    tenure_data = (
        filtered_df
        .groupby("Churn")["tenure"]
        .mean()
        .reset_index()
    )

    tenure_data.columns = [
        "Churn",
        "Average Tenure"
    ]

    fig = px.bar(
        tenure_data,
        x="Churn",
        y="Average Tenure",
        title="Average Customer Tenure by Churn Status",
        text_auto=".1f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MONTHLY CHARGES
# ============================================================

st.subheader("💰 Monthly Charges Analysis")

if "MonthlyCharges" in filtered_df.columns:

    fig = px.box(
        filtered_df,
        x="Churn",
        y="MonthlyCharges",
        title="Monthly Charges by Churn Status"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAYMENT METHOD
# ============================================================

st.subheader("💳 Payment Method Analysis")

if "PaymentMethod" in filtered_df.columns:

    payment_data = pd.crosstab(
        filtered_df["PaymentMethod"],
        filtered_df["Churn"],
        normalize="index"
    ) * 100

    payment_data = payment_data.reset_index()

    if "Yes" in payment_data.columns:

        fig = px.bar(
            payment_data,
            x="PaymentMethod",
            y="Yes",
            title="Churn Rate by Payment Method",
            labels={
                "Yes": "Churn Rate (%)"
            },
            text_auto=".1f"
        )

        fig.update_layout(
            xaxis_tickangle=-30
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PREDICTION SECTION
# ============================================================

st.divider()

st.header("🤖 Customer Churn Prediction")

st.write(
    "Enter customer information below to estimate "
    "the probability of churn."
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.warning(
        "⚠️ Trained model not found."
    )

    st.info(
        "Please run the churn prediction notebook "
        "and create models/churn_model.pkl first."
    )

else:

    try:

        model = joblib.load(MODEL_PATH)

    except Exception as e:

        st.error(
            f"❌ Could not load model: {e}"
        )

        model = None


    if model is not None:

        # ----------------------------------------------------
        # CREATE INPUT FORM
        # ----------------------------------------------------

        with st.form(
            "churn_prediction_form"
        ):

            st.subheader(
                "Customer Information"
            )

            col1, col2 = st.columns(2)


            # Gender
            if "gender" in df.columns:

                with col1:

                    gender = st.selectbox(
                        "Gender",
                        sorted(
                            df["gender"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                    )


            # Senior Citizen
            if "SeniorCitizen" in df.columns:

                with col2:

                    senior_citizen = st.selectbox(
                        "Senior Citizen",
                        [0, 1],
                        format_func=lambda x:
                            "Yes" if x == 1 else "No"
                    )


            # Partner
            if "Partner" in df.columns:

                with col1:

                    partner = st.selectbox(
                        "Partner",
                        sorted(
                            df["Partner"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                    )


            # Dependents
            if "Dependents" in df.columns:

                with col2:

                    dependents = st.selectbox(
                        "Dependents",
                        sorted(
                            df["Dependents"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                    )


            # Tenure
            if "tenure" in df.columns:

                with col1:

                    tenure = st.number_input(
                        "Tenure (Months)",
                        min_value=0,
                        max_value=100,
                        value=12
                    )


            # Monthly Charges
            if "MonthlyCharges" in df.columns:

                with col2:

                    monthly_charges = st.number_input(
                        "Monthly Charges",
                        min_value=0.0,
                        value=70.0
                    )


            # Contract
            if "Contract" in df.columns:

                contract = st.selectbox(
                    "Contract",
                    sorted(
                        df["Contract"]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                )


            # Internet Service
            if "InternetService" in df.columns:

                internet_service = st.selectbox(
                    "Internet Service",
                    sorted(
                        df["InternetService"]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                )


            # Payment Method
            if "PaymentMethod" in df.columns:

                payment_method = st.selectbox(
                    "Payment Method",
                    sorted(
                        df["PaymentMethod"]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                )


            # Total Charges
            if "TotalCharges" in df.columns:

                total_charges = st.number_input(
                    "Total Charges",
                    min_value=0.0,
                    value=1000.0
                )


            predict_button = st.form_submit_button(
                "🔮 Predict Churn"
            )


        # ====================================================
        # PREDICTION
        # ====================================================

        if predict_button:

            input_data = {}

            if "gender" in df.columns:
                input_data["gender"] = gender

            if "SeniorCitizen" in df.columns:
                input_data["SeniorCitizen"] = senior_citizen

            if "Partner" in df.columns:
                input_data["Partner"] = partner

            if "Dependents" in df.columns:
                input_data["Dependents"] = dependents

            if "tenure" in df.columns:
                input_data["tenure"] = tenure

            if "MonthlyCharges" in df.columns:
                input_data["MonthlyCharges"] = monthly_charges

            if "Contract" in df.columns:
                input_data["Contract"] = contract

            if "InternetService" in df.columns:
                input_data["InternetService"] = internet_service

            if "PaymentMethod" in df.columns:
                input_data["PaymentMethod"] = payment_method

            if "TotalCharges" in df.columns:
                input_data["TotalCharges"] = total_charges


            input_df = pd.DataFrame(
                [input_data]
            )


            # Make sure columns match
            expected_columns = list(
                model.named_steps[
                    "preprocessor"
                ].feature_names_in_
            )

            for column in expected_columns:

                if column not in input_df.columns:

                    input_df[column] = 0


            input_df = input_df[
                expected_columns
            ]


            # Prediction
            prediction = model.predict(
                input_df
            )[0]

            probability = model.predict_proba(
                input_df
            )[0][1]


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.subheader(
                "🎯 Prediction Result"
            )


            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )


            with result_col2:

                if prediction == 1:

                    st.error(
                        "⚠️ HIGH CHURN RISK"
                    )

                else:

                    st.success(
                        "✅ LOW CHURN RISK"
                    )


            # Risk interpretation

            if probability >= 0.70:

                st.error(
                    "🔴 High Risk: Customer is highly "
                    "likely to churn."
                )

            elif probability >= 0.40:

                st.warning(
                    "🟡 Medium Risk: Customer shows "
                    "some churn risk."
                )

            else:

                st.success(
                    "🟢 Low Risk: Customer has a lower "
                    "probability of churn."
                )


# ============================================================
# CUSTOMER DATA
# ============================================================

st.divider()

st.subheader("👥 Customer Data")

st.write(
    f"Showing {len(filtered_df):,} customers"
)

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Filtered CSV",
    data=csv_data,
    file_name="filtered_customer_churn.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Intelligence | "
    "Data Analyst Portfolio Project"
)
# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.divider()

st.header("💡 Business Insights & Recommendations")

st.markdown("""
### Key Recommendations

**1. 🎯 Target high-risk contract customers**  
Focus retention campaigns on customers belonging to contract types with higher churn.

**2. 🤝 Improve early customer experience**  
Customers with shorter tenure should receive stronger onboarding and engagement.

**3. 💳 Investigate payment-related churn**  
Analyze payment methods associated with higher churn and identify possible friction.

**4. 🌐 Improve high-risk service segments**  
Investigate service quality, pricing and support for internet-service segments with higher churn.

**5. 🤖 Use predictive churn scoring**  
Use the machine-learning model to identify customers with a high probability of leaving and prioritize them for retention campaigns.
""")