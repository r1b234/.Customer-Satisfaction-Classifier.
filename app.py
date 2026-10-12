import streamlit as st
import pandas as pd
import sqlite3
from crm_database import init_crm_db, fetch_crm_data
from classifier import classify_sentiment

st.set_page_config(
    page_title="Customer Satisfaction CRM",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for modern, attractive styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

init_crm_db()
df = fetch_crm_data()

st.title("Customer Satisfaction CRM Dashboard")
st.markdown("Advanced NLP-powered CRM platform analyzing local customer feedback, subscription tiers, and sentiment metrics.")

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
selected_tiers = st.sidebar.multiselect("Select Tiers", options=df['tier'].unique(), default=df['tier'].unique())
selected_classes = st.sidebar.multiselect("Satisfaction Classification", options=["Satisfied", "Neutral", "Dissatisfied"], default=["Satisfied", "Neutral", "Dissatisfied"])

filtered_df = df[df['tier'].isin(selected_tiers) & df['classification'].isin(selected_classes)]

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)
total_surveys = len(filtered_df)
satisfied_count = len(filtered_df[filtered_df['classification'] == 'Satisfied'])
neutral_count = len(filtered_df[filtered_df['classification'] == 'Neutral'])
dissatisfied_count = len(filtered_df[filtered_df['classification'] == 'Dissatisfied'])

sat_rate = (satisfied_count / total_surveys * 100) if total_surveys > 0 else 0

col1.metric("📊 Total Surveys", total_surveys)
col2.metric("✅ Satisfied Rate", f"{sat_rate:.1f}%")
col3.metric("⚖️ Neutral Feedback", neutral_count)
col4.metric("🚨 Dissatisfied (At Risk)", dissatisfied_count)

st.markdown("---")

# Layout Tabs for Visualizations and Data
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visual Analytics", "📝 Survey Records", "👤 Customer Tiers Breakdown", "➕ Live Feedback Classifier"])

with tab1:
    st.subheader("Visual Satisfaction Insights & Graphs")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### Satisfaction Classification Distribution")
        class_counts = filtered_df['classification'].value_counts()
        st.bar_chart(class_counts, color="#ff4b4b")
        
    with col_b:
        st.markdown("#### Feedback Volume by Subscription Tier")
        tier_counts = filtered_df['tier'].value_counts()
        st.bar_chart(tier_counts, color="#1f77b4")
        
    st.markdown("#### Sentiment Score Distribution (Polarity)")
    if not filtered_df.empty:
        st.line_chart(filtered_df.set_index('survey_id')['sentiment_score'])

with tab2:
    st.subheader("Filtered Survey Comments Database")
    st.dataframe(filtered_df, use_container_width=True)

with tab3:
    st.subheader("Cross-Tabulation: Tier vs Satisfaction")
    if not filtered_df.empty:
        crosstab_df = pd.crosstab(filtered_df['tier'], filtered_df['classification'])
        st.dataframe(crosstab_df, use_container_width=True)
        st.bar_chart(crosstab_df)

with tab4:
    st.subheader("Test New Customer Review (English / Hinglish)")
    with st.form("live_feedback_form"):
        cust_name = st.text_input("Customer Name (e.g., Rajesh Sharma)")
        tier_choice = st.selectbox("Subscription Tier", ["Enterprise", "Pro", "Standard", "Free"])
        comment_input = st.text_area("Customer Review / Comment")
        submit_btn = st.form_submit_button("Run NLP Classification & Save")
        
        if submit_btn and cust_name and comment_input:
            score, classification = classify_sentiment(comment_input)
            new_id = int(df['survey_id'].max() + 1) if not df.empty else 1
            
            conn = sqlite3.connect("crm_database.db")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO surveys VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                           (new_id, cust_name, tier_choice, comment_input, score, classification))
            conn.commit()
            conn.close()
            
            st.success(f"Successfully processed! Classification: **{classification}** (Sentiment Score: {score})")
            st.rerun()
