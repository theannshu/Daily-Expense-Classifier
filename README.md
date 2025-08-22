# 💰 Daily Expense Classifier

## 📌 Project Overview
The **Daily Expense Classifier** is a hybrid **rule-based + machine learning system** that helps individuals automatically categorize their expenses based on transaction descriptions.  

Instead of manually entering categories (e.g., Food, Transport, Utilities), the system:
- Uses **keyword rules** for common expenses (e.g., "Uber → Transport", "Pizza → Food")  
- Falls back to a **machine learning model** trained on labeled expense data  
- Provides a **Streamlit dashboard** for recording, analyzing, and exporting expenses  

This project was developed as part of **Project 37: Daily Expense Classifier**.

---

## ✨ Features
- 🔹 **Automatic Categorization** – enter description → get instant category  
- 🔹 **Hybrid Classification** – combines rule-based keywords + ML model  
- 🔹 **Daily Expense Overview** – quick glance at today’s total and breakdown  
- 🔹 **Spending Analysis** – weekly, monthly, and all-time summaries  
- 🔹 **Interactive Visualizations** – Pie & Bar charts for category and trends  
- 🔹 **Real-Time Dashboard Updates** – expenses update charts instantly  
- 🔹 **Export Data** – download categorized data as CSV  

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- **Pandas, Scikit-learn, NLTK** → Data cleaning, ML training  
- **Streamlit** → Dashboard & UI  
- **Plotly Express** → Interactive charts  

---

## 📂 Project Structure

DEC/
│── app.py # Streamlit dashboard (user interface)
│── classifier.py # Model training + hybrid classification logic
│── training_expenses.csv # Training dataset for ML model
│── user_expenses.csv # User-entered expenses (live dashboard data)
│── expense_classifier.pkl # Saved trained ML model
│── vectorizer.pkl # TF-IDF vectorizer used by model
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── .venv/ # Virtual environment (not uploaded to GitHub)
│── pycache/ # Cache files (ignored)


---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/theannshu/DEC.git
cd DEC


Create & activate virtual environment
python -m venv .venv
# Activate venv
# On Windows (PowerShell):
.venv\Scripts\Activate
# On Mac/Linux:
source .venv/bin/activate

Install dependencies
pip install -r requirements.txt

▶️ Running the Project
Step 1: Train the model

The model needs training data (training_expenses.csv).
Run:

python classifier.py


✅ This will:

Clean and preprocess training data

Train ML model (Naive Bayes + TF-IDF)

Save expense_classifier.pkl & vectorizer.pkl

Step 2: Start the Streamlit Dashboard
streamlit run app.py

📤 Exporting Data

The Download CSV button allows users to export all categorized expenses (user_expenses.csv)

Useful for reporting, financial tracking, or external tools (Excel, Tableau, etc.)


📌 Notes

training_expenses.csv → Only for model training

user_expenses.csv → Stores user-entered expenses in real time

Both are separate to avoid overwriting training data

For deployment, Streamlit Cloud or Docker can be used

🏆 Project Status

✔️ Fully working prototype
✔️ Meets all functional requirements in project description:

Input → Categorize → Summarize → Visualize → Export
