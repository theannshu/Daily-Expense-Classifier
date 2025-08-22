import streamlit as st
import pandas as pd
import pickle
import os
import nltk
import plotly.express as px
from datetime import datetime

# Download required NLTK resources silently
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Import helper functions
try:
    from classifier import rule_based_category, clean_text
except ImportError:
    st.error("❌ Could not import from classifier.py.")
    st.stop()

# Load model & vectorizer
if os.path.exists("expense_classifier.pkl") and os.path.exists("vectorizer.pkl"):
    with open("expense_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
else:
    st.error("❌ Model or vectorizer not found. Please run classifier.py first.")
    st.stop()

# CSV files
training_file = "training_expenses.csv"
user_file = "user_expenses.csv"

# Create CSV files if they don't exist
for file in [training_file, user_file]:
    if not os.path.exists(file):
        pd.DataFrame(columns=["Date", "Amount", "Description", "Category"]).to_csv(file, index=False)

# Load user data
df = pd.read_csv(user_file)
if not df.empty:
    df['Date'] = pd.to_datetime(df['Date'])

# Predict category function
def predict_category(description):
    category = rule_based_category(description)
    if category:
        return category
    cleaned_desc = clean_text(description)
    desc_vec = vectorizer.transform([cleaned_desc])
    return model.predict(desc_vec)[0]

# Page settings
st.set_page_config(layout="wide", page_title="Daily Expense Classifier", page_icon="💰")
st.title("💰 Daily Expense Classifier Dashboard")

# ROW 1 
col1, col2 = st.columns(2)

with col1:
    st.subheader("➕ Add New Expense")
    with st.form("expense_form"):
        date = st.date_input("Enter Date", value=datetime.now().date())
        amount = st.number_input("Enter Amount", min_value=0.0, format="%.2f")
        description = st.text_input("Enter Expense Description")
        submitted = st.form_submit_button("Add Expense")

    if submitted:
        category = predict_category(description)
        new_entry = pd.DataFrame([[date, amount, description, category]],
                                 columns=["Date", "Amount", "Description", "Category"])
        new_entry.to_csv(user_file, mode="a", header=False, index=False)
        st.success(f"✅ Expense added successfully under category: **{category}**")
        st.session_state.refresh = True
        st.rerun()

with col2:
    st.subheader("🗓️ Today's Spending Overview")
    today = datetime.now().date()
    today_df = df[df['Date'].dt.date == today] if not df.empty else pd.DataFrame()
    if not today_df.empty:
        total_spent_today = today_df['Amount'].sum()
        st.metric(label="Total Spent Today", value=f"₹{total_spent_today:.2f}")
        st.dataframe(today_df.groupby('Category')['Amount'].sum().reset_index(),
                     hide_index=True, use_container_width=True)
    else:
        st.info(f"No expenses recorded for today ({today}) yet.")

# ROW 2 
if not df.empty:
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Expense Distribution by Category")
        category_summary = df.groupby('Category')['Amount'].sum().reset_index()
        fig_pie = px.pie(category_summary, values='Amount', names='Category',
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        st.subheader("📊 Monthly Spending Trend")
        monthly_summary = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
        monthly_summary['Date'] = monthly_summary['Date'].astype(str)
        fig_bar = px.bar(monthly_summary, x='Date', y='Amount',
                         color='Amount', color_continuous_scale=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig_bar, use_container_width=True)

# ROW 3 
if not df.empty:
    st.subheader("📈 Daily Spending Over Time")
    daily_summary = df.groupby('Date')['Amount'].sum().reset_index()
    fig_line = px.line(daily_summary, x='Date', y='Amount', markers=True,
                       color_discrete_sequence=['#1f77b4'])
    fig_line.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_line, use_container_width=True)

# ROW 4 
if not df.empty:
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📅 Weekly Spending")
        weekly_summary = df.copy()
        weekly_summary['Week'] = weekly_summary['Date'].dt.to_period('W').astype(str)
        st.dataframe(weekly_summary.groupby('Week')['Amount'].sum()
                     .reset_index().sort_values(by='Week', ascending=False),
                     hide_index=True, use_container_width=True)

    with col6:
        st.subheader("📅 Monthly Spending")
        monthly_summary_table = df.copy()
        monthly_summary_table['Month'] = monthly_summary_table['Date'].dt.to_period('M').astype(str)
        st.dataframe(monthly_summary_table.groupby('Month')['Amount'].sum()
                     .reset_index().sort_values(by='Month', ascending=False),
                     hide_index=True, use_container_width=True)

#ROW 5
if not df.empty:
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("📌 Spending by Category (All Time)")
        st.dataframe(df.groupby('Category')['Amount'].sum().reset_index()
                     .sort_values(by='Amount', ascending=False),
                     hide_index=True, use_container_width=True)

    with col8:
        st.subheader("📥 Export Categorized Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download All Expenses as CSV", csv,
                           file_name="categorized_expenses.csv", mime="text/csv")
