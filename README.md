# 💰 Daily Expense Classifier (DEC)

Streamlit app using pyhton as core language, that auto-categorizes expenses using a hybrid rule-based + ML model and visualizes your spending.

## Features
- Add new expenses (Date, Description, Amount)
- Automatic category prediction (Food, Transport, Bills, Education, etc.)
- Daily, Weekly, Monthly summaries
- Interactive Pie, Bar, Line charts
- Export categorized data as CSV
- One-click retrain from `training_expenses.csv`

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

## If you need to retrain

- python classifier.py
