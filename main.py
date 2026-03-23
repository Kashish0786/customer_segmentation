
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------
# 1️⃣ Load Pickle Files
# -------------------------------
def load_pickles():
    rfm_df = pickle.load(open("rfm_final_upload.pkl", "rb"))
    final_df = pickle.load(open("final_df.pkl", "rb"))
    hybrid_df = pickle.load(open("hybrid_df.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    segment_map = pickle.load(open("segment_map.pkl", "rb"))
    final_customer = pickle.load(open("final_customer_data.pkl", "rb"))
    df = pickle.load(open("df.pkl", "rb"))
    kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
    return hybrid_df, final_df, rfm_df, scaler, segment_map, final_customer, df, kmeans_model

# -------------------------------
# 2️⃣ KPI Calculation
# -------------------------------
def calculate_kpis(df):
    total_customers = df['CustomerID'].nunique()
    total_revenue = df['TotalPrice'].sum()
    avg_order = df['TotalPrice'].mean()
    return total_customers, total_revenue, avg_order

# -------------------------------
# 3️⃣ Charts / Plots
# -------------------------------
def plot_segment_distribution(rfm_df):
    seg_counts = rfm_df['Segment'].value_counts()
    fig = px.pie(names=seg_counts.index, values=seg_counts.values, title="Segment Distribution")
    return fig

def plot_top_customers(df):
    # Top 10 customers by total spending
    top_customers = df.groupby('CustomerID')['TotalPrice'].sum().nlargest(10)
    fig = px.bar(
        top_customers,
        x=top_customers.index,
        y=top_customers.values,
        title="Top 10 Customers by Spending",
        labels={"x": "CustomerID", "y": "Total Spending"}
    )
    return fig

def plot_sales_trend(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['TotalPrice'].sum().reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].astype(str)
    fig = px.line(monthly_sales, x='Month', y='TotalPrice', title="Monthly Sales Trend")
    return fig

# -------------------------------
# 4️⃣ Predict Customer Segment
# -------------------------------
def predict_customer_segment(final_df, scaler, kmeans_model, segment_map, customer_id):
    """
    Predict customer segment for a given CustomerID.
    Works with final_df (already contains CustomerID, Recency, Frequency, Monetary columns).
    """
    if customer_id not in final_df['CustomerID'].values:
        return "CustomerID not found"

    customer_row = final_df.loc[final_df['CustomerID'] == customer_id, ['Recency', 'Frequency', 'Monetary']]
    scaled_row = scaler.transform(customer_row)
    cluster = kmeans_model.predict(scaled_row)[0]
    segment = segment_map.get(cluster, "Unknown Segment")
    return segment