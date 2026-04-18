import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load intents
with open("intents.json") as file:
    data = json.load(file)

sentences = []
labels = []

# Prepare dataset
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        pattern = pattern.lower()
        pattern = re.sub(r'[^a-zA-Z ]', '', pattern)

        sentences.append(pattern)
        labels.append(intent["tag"])

# Vectorization (🔥 BIGRAM for better accuracy)
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(sentences)

# Train model (🔥 best for text)
model = MultinomialNB()
model.fit(X, labels)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")