
# app.py
import streamlit as st
import pandas as pd
import webbrowser
from main import load_pickles, calculate_kpis, plot_segment_distribution, plot_top_customers, plot_sales_trend

# -------------------------------
# 1️⃣ Load Data & Models
# -------------------------------
hybrid_df, final_df, rfm_df, scaler, segment_map, final_customer, df, kmeans_model = load_pickles()

# -------------------------------
# 2️⃣ Page Config
# -------------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# -------------------------------
# 3️⃣ Dashboard Header
# -------------------------------
st.title("Customer Segmentation Dashboard")
st.markdown("Interactive dashboard integrating Tableau and Python insights.")

# -------------------------------
# 4️⃣ Tableau Section
# -------------------------------

st.subheader("Tableau Dashboard")

# link
# tableau_link = "https://public.tableau.com/views/Book1_17741968575400/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
# if st.button("Click to Open Tableau Dashboard"):
#     webbrowser.open_new_tab(tableau_link)

tableau_link = "https://public.tableau.com/views/Book1_17741968575400/Dashboard1"

st.markdown(f"[👉 Click to Open Tableau Dashboard]({tableau_link})")
st.subheader("Some Images Of Tableau Dashboard")
    
# 2x2 Collage
col1, col2 = st.columns(2, gap="small")

with col1:
    st.image("img1.png", use_container_width=True)
    st.image("img2.png", use_container_width=True)

with col2:
    st.image("img3.png", use_container_width=True)


# -------------------------------
# 5️⃣ KPI Cards (Responsive)
# -------------------------------
st.subheader("Key Metrics")
total_customers, total_revenue, avg_order = calculate_kpis(df)
col1, col2, col3 = st.columns([1,1,1], gap="medium")
col1.metric("Total Customers", total_customers)
col2.metric("Total Revenue", f"${total_revenue:,.2f}")
col3.metric("Average Order Value", f"${avg_order:,.2f}")

# -------------------------------
# 6️⃣ Charts Section (Responsive)
# -------------------------------
st.subheader("Charts")
tab1, tab2, tab3 = st.tabs(["Segment Distribution", "Top Customers", "Monthly Trend"])

with tab1:
    st.plotly_chart(plot_segment_distribution(final_df), use_container_width=True)
with tab2:
    st.plotly_chart(plot_top_customers(df), use_container_width=True)
with tab3:
    st.plotly_chart(plot_sales_trend(df), use_container_width=True)

# -------------------------------
# 7️⃣ Customer Segment Prediction
# -------------------------------
st.title("Customer Segment Prediction")
st.markdown("Select a customer to see which segment they belong to based on their purchase behavior.")

# Responsive layout for selectbox + button
col1, col2 = st.columns([2,1])
with col1:
    customer_ids = final_df['CustomerID'].unique().tolist()
    selected_customer = st.selectbox("Select Customer:", customer_ids)
with col2:
    predict_btn = st.button("Predict Segment")

# -------------------------------
# Helper Function to Get RFM
# -------------------------------
def get_customer_rfm(customer_id, df):
    customer_data = df[df['CustomerID'] == customer_id].copy()
    if customer_data.empty:
        return None
    r = customer_data['Recency'].values[0]
    f = customer_data['Frequency'].values[0]
    m = customer_data['Monetary'].values[0]
    return [r, f, m]

# -------------------------------
# Predict Segment
# -------------------------------
if predict_btn:
    customer_rfm = get_customer_rfm(selected_customer, final_df)
    if customer_rfm is None:
        st.error("Customer data not found!")
    else:
        scaled_rfm = scaler.transform([customer_rfm])
        cluster = kmeans_model.predict(scaled_rfm)[0]
        segment_label = segment_map.get(cluster, "Unknown Segment")

        st.markdown(
            f"<h4>Recency: {customer_rfm[0]} | Frequency: {customer_rfm[1]} | Monetary: ${customer_rfm[2]:,.2f}</h4>",
            unsafe_allow_html=True
        )
        st.markdown(f"<h3>Predicted Segment: {segment_label}</h3>", unsafe_allow_html=True)
