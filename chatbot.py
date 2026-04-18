import json
import pickle
import random
import re
import datetime
from textblob import TextBlob   # ✅ Spell correction

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    data = json.load(file)

# ✅ Chat history list
chat_log = []

import datetime

# 👇 Ask user name
name = input("👤 Enter your name: ")

# 👇 Time-based greeting
hour = datetime.datetime.now().hour

if hour < 12:
    greeting = "Good Morning"
elif hour < 18:
    greeting = "Good Afternoon"
else:
    greeting = "Good Evening"

print(f"\n🤖 {greeting}, {name}! Welcome to Smart AI Chatbot")





hour = datetime.datetime.now().hour


print("""
  You can ask:
1. Study tips
2. Career guidance
3. Programming
4. Jokes
5. Motivation
""")

print("⌨️ Type 'quit' to exit")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break
    # ✅ SPELL CORRECTION
    corrected_input = str(TextBlob(user_input).correct())
    if corrected_input.lower() != user_input.lower():
        print("🔧 Corrected:", corrected_input)

    # Clean input
    cleaned_input = corrected_input.lower()
    cleaned_input = re.sub(r'[^a-zA-Z ]', '', cleaned_input)

    response = ""   # ✅ store bot response

    # 🔥 RULE-BASED RESPONSES

    if "study" in cleaned_input:
        response = "Study regularly and revise concepts."

    elif "exam" in cleaned_input:
        response = "Practice previous papers and revise important topics."
    
    elif "time" in cleaned_input:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        response = f"Current time is {current_time}"

    elif "date" in cleaned_input:
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        response = f"Today's date is {current_date}"

    elif "career" in cleaned_input or "carrer" in cleaned_input:
        response = "Focus on skills, projects, and consistency."

    elif "joke" in cleaned_input:
        response = "Why do programmers hate nature? Too many bugs 😂"

    elif "hi" in cleaned_input or "hello" in cleaned_input or "hey" in cleaned_input:
        response = random.choice(["Hello!", "Hi there!", "Hey!"])

    elif "bye" in cleaned_input or "goodbye" in cleaned_input:
        response = "Goodbye! Take care!"

    elif "thanks" in cleaned_input:
        response = "You're welcome!"

    elif "placement" in cleaned_input:
        response = "Focus on DSA, projects, and communication skills."

    elif "skill" in cleaned_input:
        response = "Learn programming, problem-solving, and communication skills."

    elif "python" in cleaned_input:
        response = "Python is widely used in AI, web development, and automation."

    elif "java" in cleaned_input:
        response = "Java is an object-oriented programming language."

    elif "html" in cleaned_input:
        response = "HTML is used to create web pages."

    elif "ai" in cleaned_input:
        response = "AI is the simulation of human intelligence in machines."

    elif "machine learning" in cleaned_input or "ml" in cleaned_input:
        response = "Machine learning allows systems to learn from data."

    elif "cricket" in cleaned_input:
        response = "Cricket is a popular sport played with bat and ball."

    elif "football" in cleaned_input:
        response = "Football is the world's most popular sport."

    elif "sport" in cleaned_input:
        response = "Sports are great for health and teamwork!"

    elif "movie" in cleaned_input:
        response = "You can watch Inception, Interstellar, or Avengers!"

    elif "technology" in cleaned_input:
        response = "Technology solves real-world problems using innovation."

    elif "computer" in cleaned_input:
        response = "A computer is an electronic device that processes data."

    elif "internet" in cleaned_input:
        response = "The internet connects millions of computers worldwide."

    elif "health" in cleaned_input:
        response = "Eat healthy, exercise daily, and sleep well."

    elif "exercise" in cleaned_input:
        response = "Exercise keeps your body fit and mind fresh."

    elif "diet" in cleaned_input:
        response = "Eat balanced meals with fruits, vegetables, and protein."

    elif "time" in cleaned_input:
        response = "I can't access real-time clock yet."

    elif "weather" in cleaned_input:
        response = "I can't fetch live weather currently."

    elif "motivate" in cleaned_input or "motivation" in cleaned_input:
        response = "Keep going! You are doing great 💪"

    else:
        # 🔥 ML Prediction (fallback)
        X_test = vectorizer.transform([cleaned_input])
        prediction = model.predict(X_test)[0]
        proba = model.predict_proba(X_test)
        confidence = max(proba[0])

        print("Predicted:", prediction)
        print("Confidence:", confidence)

        if confidence < 0.3:
           response = random.choice([
    "Hmm, I didn't get that 🤔",
    "Can you rephrase?",
    "I'm still learning 😅"
])
        else:
            for intent in data["intents"]:
                if intent["tag"] == prediction:
                    response = random.choice(intent["responses"])

    # ✅ PRINT RESPONSE
    print("Bot:", response)

    # ✅ SAVE CHAT HISTORY
    chat_log.append({
        "user": user_input,
        "bot": response
    })

    # ✅ DISPLAY CHAT HISTORY
    print("\n📜 Chat History:")
    for chat in chat_log:
        print(f"You: {chat['user']} | Bot: {chat['bot']}")
    print("------------------")
    response = f"{name}, keep going! You are doing great 💪"