import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

TRAINING_FILE = "training_expenses.csv"


# Rule-based keyword overrides

keyword_overrides = {
    # Food / Groceries
    "food": "Food",
    "pizza": "Food",
    "restaurant": "Food",
    "groceries": "Groceries",
    "supermarket": "Groceries",
    "snack": "Food",
    "sweets": "Food",

    # Shopping
    "shirt": "Shopping",
    "clothes": "Shopping",
    "shoes": "Shopping",
    "dress": "Shopping",

    # Transport
    "uber": "Transport",
    "bus": "Transport",
    "cab": "Transport",
    "taxi": "Transport",
    "metro": "Transport",
    "train": "Transport",
    "petrol": "Transport",
    "fuel": "Transport",

    # Utilities / Bills
    "electricity": "Utilities",
    "water bill": "Utilities",
    "internet": "Utilities",
    "wifi": "Utilities",
    "recharge": "Bills",
    "phone": "Bills",
    "mobile": "Bills",

    # Health
    "hospital": "Health",
    "medicine": "Health",
    "doctor": "Health",
    "treatment": "Health",
    "consultation": "Health",

    # Donations
    "donate": "Donations",
    "charity": "Donations",
    "beggar": "Donations",

    # Entertainment
    "movie": "Entertainment",
    "cinema": "Entertainment",
    "netflix": "Entertainment",
    "spotify": "Entertainment",

    # Education
    "college": "Education",
    "school": "Education",
    "course": "Education",
    "tuition": "Education",

    # Rent
    "rent": "Rent",
    "hostel": "Rent",

    # Others
    "gift": "Others",
    "misc": "Others"
}


# Text cleaning function

def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# Rule-based category detection

def rule_based_category(description):
    desc_lower = description.lower()
    for keyword, category in keyword_overrides.items():
        if keyword in desc_lower:
            return category
    return None


# Load and preprocess dataset

if not os.path.exists(TRAINING_FILE):
    raise FileNotFoundError(f"❌ Training data file '{TRAINING_FILE}' not found! Please create it with labeled expenses.")

df = pd.read_csv(TRAINING_FILE)

if "Description" not in df.columns or "Category" not in df.columns:
    raise ValueError("❌ Training file must contain 'Description' and 'Category' columns.")

df["Cleaned_Description"] = df["Description"].apply(clean_text)


# Train ML model

X = df["Cleaned_Description"]
y = df["Category"]

# Remove categories with < 2 samples
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df["Category"].isin(valid_classes)]

X = df["Cleaned_Description"]
y = df["Category"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

clf = MultinomialNB()
clf.fit(X_train, y_train)


# Evaluate model

y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save model & vectorizer

with open("expense_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("💾 Model and vectorizer saved!")


# Hybrid prediction function

def predict_category(description):
    rb_category = rule_based_category(description)
    if rb_category:
        return rb_category

    desc_clean = clean_text(description)
    desc_vec = vectorizer.transform([desc_clean])
    return clf.predict(desc_vec)[0]
