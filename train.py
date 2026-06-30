"""
train.py — SmartBot Model Trainer
Run this once before starting chatbot.py
"""

import json
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Download required NLTK data ───────────────────────────────────────────────
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# ── Preprocessing (must match chatbot.py) ────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w]
    return ' '.join(words)

# ── Load intents ──────────────────────────────────────────────────────────────
try:
    with open("intents.json", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded intents.json — {len(data['intents'])} intents found")
except FileNotFoundError:
    print("❌ intents.json not found! Please create it first.")
    exit(1)

# ── Build dataset ─────────────────────────────────────────────────────────────
sentences, labels = [], []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(preprocess(pattern))
        labels.append(intent["tag"])

print(f"📊 Total training samples: {len(sentences)}")

if len(set(labels)) < 2:
    print("❌ Need at least 2 intent tags to train.")
    exit(1)

# ── Train/Test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42, stratify=labels
    if all(labels.count(l) >= 2 for l in set(labels)) else None
)

# ── Build pipeline ────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),       
        stop_words='english',
        max_features=8000,
        sublinear_tf=True,        
    )),
    ('clf', MultinomialNB(alpha=0.3)) 
])

# ── Train ─────────────────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

# ── Evaluate ──────────────────────────────────────────────────────────────────
if len(X_test) > 0:
    y_pred = pipeline.predict(X_test)
    accuracy = (y_pred == y_test).mean() * 100
    print(f"📈 Accuracy on test set: {accuracy:.1f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# ── Save model ────────────────────────────────────────────────────────────────
# Save the full pipeline (vectorizer + model together)
pickle.dump(pipeline, open("model_pipeline.pkl", "wb"))
print("💾 Saved: model_pipeline.pkl")

# Also save separately for backward compatibility
pickle.dump(pipeline.named_steps['tfidf'], open("vectorizer.pkl", "wb"))
pickle.dump(pipeline.named_steps['clf'], open("model.pkl", "wb"))
print("💾 Saved: vectorizer.pkl and model.pkl")
print("\n🚀 Ready! Now run: python chatbot.py")
