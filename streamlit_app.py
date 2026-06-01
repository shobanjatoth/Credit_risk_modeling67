import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)


st.markdown("""
<style>

/* ==========================
   GLOBAL APP
========================== */

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 50%,
        #334155 100%
    );
}

/* ==========================
   HEADER
========================== */

.main-title{
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:700;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:25px;
}

/* ==========================
   FORM CARD
========================== */

div[data-testid="stForm"]{
    background:rgba(255,255,255,0.05);
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.08);
}

/* ==========================
   LABELS
========================== */

label,
.stMarkdown,
p{
    color:white !important;
}

/* ==========================
   NUMBER INPUTS
========================== */

div[data-testid="stNumberInput"]{
    width:100%;
}

div[data-testid="stNumberInput"] input{
    background:#1e293b !important;
    color:white !important;
    caret-color:white !important;
    border:1px solid #475569 !important;
    border-radius:10px !important;
}

/* Plus / Minus Buttons */

div[data-testid="stNumberInput"] button{
    background:#2563eb !important;
    color:white !important;
    border:none !important;
}

div[data-testid="stNumberInput"] button:hover{
    background:#1d4ed8 !important;
    color:white !important;
}

/* ==========================
   SIDEBAR
========================== */

section[data-testid="stSidebar"]{
    background:#111827;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* ==========================
   BUTTON
========================== */

div.stButton > button,
div.stForm button{
    width:100%;
    background:linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    ) !important;
    color:white !important;
    border:none !important;
    border-radius:12px !important;
    padding:12px !important;
    font-size:18px !important;
    font-weight:700 !important;
}

/* ==========================
   RESULT BOX
========================== */

.result-box{
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
    font-size:32px;
    font-weight:700;
    margin-top:25px;
}

.low-risk{
    background:#16a34a;
}

.medium-risk{
    background:#d97706;
}

.high-risk{
    background:#ea580c;
}

.very-high-risk{
    background:#dc2626;
}

/* ==========================
   METRICS
========================== */

div[data-testid="metric-container"]{
    background:rgba(255,255,255,0.08);
    border-radius:15px;
    padding:15px;
    border:1px solid rgba(255,255,255,0.1);
}

div[data-testid="metric-container"] *{
    color:white !important;
}

/* ==========================
   REMOVE WHITE INPUT AREAS
========================== */

[data-baseweb="input"]{
    background:#1e293b !important;
}

[data-baseweb="base-input"]{
    background:#1e293b !important;
    border-radius:10px !important;
}

/* ==========================
   LAYOUT
========================== */

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

</style>
""", unsafe_allow_html=True)



st.sidebar.title("💳 Credit Risk Dashboard")

st.sidebar.markdown("""
### Risk Categories

🟢 Low Risk

🟡 Medium Risk

🟠 High Risk

🔴 Very High Risk

This AI system evaluates the applicant's credit profile and predicts the risk category.
""")



st.markdown(
    """
    <div class="main-title">
        💳 Credit Risk Prediction System
    </div>

    <div class="sub-title">
        AI Powered Credit Risk Assessment Dashboard
    </div>
    """,
    unsafe_allow_html=True
)



with st.form("credit_risk_form"):

    col1, col2 = st.columns(2)

    with col1:

        Age_Oldest_TL = st.number_input("Age of Oldest Trade Line", 0.0)
        enq_L3m = st.number_input("Number of Enquiries in Last 3 Months", 0.0)
        time_since_recent_enq = st.number_input("Months Since Recent Enquiry", 0.0)
        enq_L6m = st.number_input("Number of Enquiries in Last 6 Months", 0.0)
        num_std = st.number_input("Number of Standard Accounts", 0.0)
        num_std_12mts = st.number_input("Standard Accounts in Last 12 Months", 0.0)
        Time_With_Curr_Empr = st.number_input("Time With Current Employer", 0.0)
        time_since_recent_payment = st.number_input("Months Since Recent Payment", 0.0)
        AGE = st.number_input("Applicant Age", 18.0, value=30.0)
        enq_L12m = st.number_input("Number of Enquiries in Last 12 Months", 0.0)
        NETMONTHLYINCOME = st.number_input("Net Monthly Income", 0.0, value=25000.0)
        Age_Newest_TL = st.number_input("Age of Newest Trade Line", 0.0)
        pct_currentBal_all_TL = st.number_input("Current Balance Percentage Across All Trade Lines", 0.0)
        num_std_6mts = st.number_input("Standard Accounts in Last 6 Months", 0.0)
        time_since_recent_deliquency = st.number_input("Months Since Recent Delinquency", 0.0)

    with col2:

        Total_TL = st.number_input("Total Trade Lines", 0.0)
        tot_enq = st.number_input("Total Enquiries", 0.0)
        max_unsec_exposure_inPct = st.number_input("Maximum Unsecured Exposure (%)", 0.0)
        time_since_first_deliquency = st.number_input("Months Since First Delinquency", 0.0)
        pct_PL_enq_L6m_of_L12m = st.number_input("Personal Loan Enquiries (6M / 12M %)", 0.0)
        max_delinquency_level = st.number_input("Maximum Delinquency Level", 0.0)
        Secured_TL = st.number_input("Secured Trade Lines", 0.0)
        Tot_Closed_TL = st.number_input("Total Closed Trade Lines", 0.0)
        pct_tl_open_L12M = st.number_input("Trade Lines Opened in Last 12 Months (%)", 0.0)
        pct_PL_enq_L6m_of_ever = st.number_input("Personal Loan Enquiries (6M / Ever %)", 0.0)
        max_recent_level_of_deliq = st.number_input("Maximum Recent Delinquency Level", 0.0)
        PL_enq_L6m = st.number_input("Personal Loan Enquiries in Last 6 Months", 0.0)
        recent_level_of_deliq = st.number_input("Recent Delinquency Level", 0.0)
        Other_TL = st.number_input("Other Trade Lines", 0.0)
        pct_active_tl = st.number_input("Active Trade Lines Percentage", 0.0)

    submit = st.form_submit_button("🔍 Predict Credit Risk")



if submit:

    payload = {
        "Age_Oldest_TL": Age_Oldest_TL,
        "enq_L3m": enq_L3m,
        "time_since_recent_enq": time_since_recent_enq,
        "enq_L6m": enq_L6m,
        "num_std": num_std,
        "num_std_12mts": num_std_12mts,
        "Time_With_Curr_Empr": Time_With_Curr_Empr,
        "time_since_recent_payment": time_since_recent_payment,
        "AGE": AGE,
        "enq_L12m": enq_L12m,
        "NETMONTHLYINCOME": NETMONTHLYINCOME,
        "Age_Newest_TL": Age_Newest_TL,
        "pct_currentBal_all_TL": pct_currentBal_all_TL,
        "num_std_6mts": num_std_6mts,
        "time_since_recent_deliquency": time_since_recent_deliquency,
        "Total_TL": Total_TL,
        "tot_enq": tot_enq,
        "max_unsec_exposure_inPct": max_unsec_exposure_inPct,
        "time_since_first_deliquency": time_since_first_deliquency,
        "pct_PL_enq_L6m_of_L12m": pct_PL_enq_L6m_of_L12m,
        "max_delinquency_level": max_delinquency_level,
        "Secured_TL": Secured_TL,
        "Tot_Closed_TL": Tot_Closed_TL,
        "pct_tl_open_L12M": pct_tl_open_L12M,
        "pct_PL_enq_L6m_of_ever": pct_PL_enq_L6m_of_ever,
        "max_recent_level_of_deliq": max_recent_level_of_deliq,
        "PL_enq_L6m": PL_enq_L6m,
        "recent_level_of_deliq": recent_level_of_deliq,
        "Other_TL": Other_TL,
        "pct_active_tl": pct_active_tl
    }

    try:

        with st.spinner("Analyzing credit profile..."):

            response = requests.post(
                API_URL,
                json=payload
            )

        if response.status_code == 200:

            result = response.json()

            prediction_map = {
                0: ("🟢 LOW RISK", "low-risk"),
                1: ("🟡 MEDIUM RISK", "medium-risk"),
                2: ("🟠 HIGH RISK", "high-risk"),
                3: ("🔴 VERY HIGH RISK", "very-high-risk")
            }

            prediction_text, css_class = prediction_map.get(
                result["prediction"],
                ("UNKNOWN", "high-risk")
            )

            st.markdown(
                f'''
                <div class="result-box {css_class}">
                    {prediction_text}
                </div>
                ''',
                unsafe_allow_html=True
            )

            probs = result["probabilities"]

            st.markdown("## 📊 Risk Probability Analysis")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("🟢 Low Risk", f"{probs[0]*100:.2f}%")

            with c2:
                st.metric("🟡 Medium Risk", f"{probs[1]*100:.2f}%")

            with c3:
                st.metric("🟠 High Risk", f"{probs[2]*100:.2f}%")

            with c4:
                st.metric("🔴 Very High Risk", f"{probs[3]*100:.2f}%")

        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"Connection Error: {e}")

