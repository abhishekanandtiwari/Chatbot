"""
chatbot.py — SmartBot with ML + Groq AI fallback
Run train.py first to generate model_pipeline.pkl
"""

import json
import pickle
import random
import re
import datetime
import os
import nltk
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from groq import Groq

# ── NLTK setup ────────────────────────────────────────────────────────────────
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

# =============================================================================
# CONFIGURATION — edit these
# =============================================================================
GROQ_API_KEY  = "your_api_key_here"    # GET FREE KEY: https://console.groq.com
GROQ_MODEL    = "llama-3.1-8b-instant" # stable free model
ML_CONFIDENCE = 0.25                   # below this threshold -> use keyword/groq
BOT_NAME      = "SmartBot"
# =============================================================================

# ── Load ML pipeline ──────────────────────────────────────────────────────────
try:
    pipeline = pickle.load(open("model_pipeline.pkl", "rb"))
    print("✅ ML model loaded.")
except FileNotFoundError:
    print("❌ model_pipeline.pkl not found. Please run train.py first.")
    exit(1)

# ── Load intents ──────────────────────────────────────────────────────────────
try:
    with open("intents.json", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ intents.json not found.")
    exit(1)

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── State ─────────────────────────────────────────────────────────────────────
chat_log       = []
groq_history   = []
MAX_GROQ_TURNS = 6

# =============================================================================
# HELPERS
# =============================================================================

def preprocess(text: str) -> str:
    """Spell-correct, lowercase, lemmatize."""
    try:
        text = str(TextBlob(text).correct())
    except Exception:
        pass
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w]
    return ' '.join(words)


def rule_based(raw: str):
    """Real-time rules. Returns response string, '__EXIT__', or None."""
    t = raw.lower()

    if any(p in t for p in ["what time", "current time", "time now", "what's the time"]):
        return datetime.datetime.now().strftime("It's %I:%M %p")

    if any(p in t for p in ["what date", "today's date", "today date", "what day"]):
        return datetime.datetime.now().strftime("Today is %A, %d %B %Y")

    if t in {"exit", "bye", "goodbye", "see you", "see ya", "cya"}:
        return "__EXIT__"

    return None


def keyword_match(low: str):
    """Fast keyword lookup. Returns response string or None."""
    keyword_map = {
        ("hi", "hello", "hey", "hiya", "yo", "sup", "howdy", "greetings", "hlo", "helo"):
            ["Hello! How can I help you? 😊",
             "Hey there! What's on your mind?",
             "Hi! I'm SmartBot. Ask me anything!"],

        ("joke", "funny", "laugh", "humor", "make me laugh", "tell me a joke"):
            ["Why do programmers hate nature? Too many bugs! 😂",
             "Why did the developer go broke? He used all his cache! 😄",
             "Why do Java developers wear glasses? They don't C#! 😆",
             "What's a computer's favourite snack? Microchips! 🍟"],

        ("study", "studies", "study tips", "how to study", "concentrate", "focus"):
            ["Try the Pomodoro technique: 25 min study, 5 min break! ⏱️",
             "Revise before sleeping — your brain consolidates memory overnight 🧠",
             "Test yourself instead of re-reading. Active recall works best! 📚"],

        ("exam", "test", "marks", "grades", "score", "revision", "how to pass"):
            ["Practice previous papers and revise important topics! 📝",
             "Don't panic! Revise notes, sleep well, eat before the exam 💪"],

        ("motivate", "motivation", "feel low", "give up", "feeling sad",
         "i am tired", "stressed", "demotivated", "want to quit", "failing", "hopeless"):
            ["Keep going! Every expert was once a beginner 💪",
             "Tough times don't last, tough people do! 🔥",
             "You are capable of amazing things. Believe in yourself! ⭐"],

        ("career", "job", "placement", "interview", "internship"):
            ["Focus on skills, build projects, and network actively! 💼",
             "Practice DSA on LeetCode daily and build real projects 🚀"],

        ("python",):
            ["Python is great for AI, web dev, and automation! 🐍",
             "Python is beginner-friendly with a huge community. Start today! 💡"],

        ("java",):
            ["Java is object-oriented and used in Android and backend dev! ☕"],

        ("javascript", "nodejs", "node js", "react"):
            ["JavaScript is the language of the web — frontend and backend! 🌐"],

        ("html", "css", "web design", "make a website"):
            ["HTML is the structure, CSS is the styling — together they make websites! 🌐"],

        ("machine learning", "what is ml", "deep learning", "neural network"):
            ["ML lets computers learn from data without being explicitly programmed! 🧠"],

        ("what is ai", "artificial intelligence", "ai application"):
            ["AI simulates human intelligence in machines — voice assistants, self-driving cars, and more! 🤖"],

        ("data science", "data analyst", "become data scientist"):
            ["Data Science combines statistics, programming, and domain knowledge! 📊",
             "Learn Python, SQL, Pandas, and ML to start in data science! 💡"],

        ("health", "healthy", "stay fit", "healthy lifestyle"):
            ["Eat balanced meals, exercise daily, and sleep 7-8 hours! 💪"],

        ("exercise", "workout", "gym", "fitness", "lose weight", "build muscle"):
            ["Start with 30 min of exercise 3-4 times a week. Consistency beats intensity! 💪"],

        ("diet", "eat healthy", "nutrition", "healthy food"):
            ["Eat balanced meals: proteins, healthy fats, fruits, and veggies! 🥗"],

        ("money", "save money", "savings", "invest", "finance", "budget"):
            ["Save 20% of income and invest early in index funds! 📈",
             "Follow 50-30-20 rule: 50% needs, 30% wants, 20% savings! 💰"],

        ("time management", "productive", "productivity", "procrastinat"):
            ["Plan your day the night before and focus on your top 3 tasks! 📋"],

        ("book", "recommend a book", "what to read"):
            ["Must-reads: Atomic Habits, The Alchemist, Rich Dad Poor Dad! 📚"],

        ("movie", "film", "what to watch", "recommend a movie"):
            ["Must-watch: Inception, Interstellar, The Dark Knight, 3 Idiots! 🎬"],

        ("cricket", "football", "sport", "ipl", "world cup"):
            ["Sports build teamwork, discipline, and fitness! ⚽🏏"],

        ("thanks", "thank you", "thank", "cheers", "appreciated", "that helped"):
            ["You're welcome! 😊", "Happy to help! 🙌", "Anytime! 🤖"],

        ("goodbye", "good night", "take care", "see you later", "bye bye"):
            ["Goodbye! Have a great day! 👋", "See you later! Take care! 😊"],

        ("who are you", "what are you", "are you a bot", "your name", "introduce yourself"):
            [f"I'm {BOT_NAME} — an AI-powered chatbot that can answer almost anything! 🤖"],

        ("how are you", "how are you doing", "you good", "how's it going"):
            ["I'm doing great, thanks! 😊 How about you?",
             "All systems running smoothly! 🤖 How can I help?"],

        ("cheer me up", "say something positive", "good vibes", "uplift me"):
            ["You are stronger than you think! Keep going 💪",
             "The best is yet to come. Stay positive! ✨"],

        ("fun fact", "did you know", "trivia", "interesting fact"):
            ["Fun fact: Honey never spoils — archaeologists found 3000-year-old honey in Egyptian tombs! 🍯",
             "Did you know? Octopuses have three hearts and blue blood! 🐙"],

        ("cybersecurity", "ethical hacking", "hacker", "network security"):
            ["Cybersecurity protects systems from digital attacks! 🔐",
             "To start: learn networking, Linux, and tools like Kali Linux! 💡"],

        ("cloud", "aws", "azure", "google cloud"):
            ["Cloud Computing delivers storage and servers over the internet! ☁️",
             "Top platforms: AWS, Azure, and Google Cloud. AWS is most popular! 💡"],
    }

    for keywords, replies in keyword_map.items():
        if any(kw in low for kw in keywords):
            return random.choice(replies)

    return None


def ml_predict(cleaned: str):
    """ML pipeline prediction. Returns (tag, response, confidence) or (None, None, conf)."""
    proba      = pipeline.predict_proba([cleaned])[0]
    confidence = max(proba)
    tag        = pipeline.classes_[proba.argmax()]

    if confidence < ML_CONFIDENCE:
        return None, None, confidence

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return tag, random.choice(intent["responses"]), confidence

    return None, None, confidence


def ask_groq(user_input: str, user_name: str, system_prompt: str) -> str:
    """Groq AI call with rolling conversation history."""
    global groq_history

    groq_history.append({"role": "user", "content": user_input})
    trimmed = groq_history[-(MAX_GROQ_TURNS * 2):]

    try:
        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + trimmed,
            max_tokens=300,
            temperature=0.7,
        )
        reply = res.choices[0].message.content.strip()
        groq_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"I couldn't connect right now. Please check your internet or API key. (Error: {e})"


def show_history():
    if not chat_log:
        print("📜 No chat history yet.\n")
        return
    print("\n📜 Chat History")
    print("-" * 50)
    for i, entry in enumerate(chat_log, 1):
        print(f"  {i:>2}. You : {entry['user']}")
        print(f"       Bot : {entry['bot']}")
    print("-" * 50 + "\n")


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# =============================================================================
# STARTUP
# =============================================================================

clear_screen()
name = input("Enter your name: ").strip() or "Friend"

hour     = datetime.datetime.now().hour
greeting = ("Good Morning" if hour < 12
            else "Good Afternoon" if hour < 18
            else "Good Evening")

SYSTEM_PROMPT = (
    f"You are {BOT_NAME}, a friendly, smart, and helpful AI assistant "
    f"talking to {name}. "
    "Answer questions clearly and concisely in 2-4 lines. "
    "Use emojis occasionally to keep the tone friendly. "
    "If asked who made you, say you are SmartBot, an independent AI assistant. "
    "Never say you are made by Meta, Google, or any company."
)

print(f"\n{'=' * 50}")
print(f"  {greeting}, {name}! I'm {BOT_NAME}")
print(f"{'=' * 50}")
print("  Ask me ANYTHING!")
print("  'history' = chat log | 'clear' = clear screen | 'quit' = exit")
print(f"{'=' * 50}\n")

# =============================================================================
# MAIN LOOP
# =============================================================================

while True:
    try:
        user_input = input(f"[{name}] > ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\nGoodbye, {name}! Take care!")
        break

    if not user_input:
        continue

    low = user_input.lower()

    # ── Built-in commands ──────────────────────────────────────────────────────
    if low in {"quit", "exit"}:
        print(f"\nGoodbye, {name}! Keep learning and growing!")
        break

    if low == "history":
        show_history()
        continue

    if low == "clear":
        clear_screen()
        continue

    # ==========================================================================
    # RESPONSE PIPELINE
    #  1. Rule-based  — real-time data (time, date)
    #  2. Keyword map — instant local answers for common topics
    #  3. ML model    — trained intent classifier
    #  4. Groq AI     — answers ANYTHING via internet
    # ==========================================================================
    source   = ""
    response = ""

    # 1 — Rule-based
    rule = rule_based(user_input)
    if rule == "__EXIT__":
        print(f"\nGoodbye, {name}! Take care!")
        break
    elif rule:
        response = rule
        source   = "rule"

    # 2 — Keyword match
    if not response:
        kw_resp = keyword_match(low)
        if kw_resp:
            response = kw_resp
            source   = "keyword"

    # 3 — ML model
    if not response:
        cleaned            = preprocess(user_input)
        tag, ml_resp, conf = ml_predict(cleaned)
        if ml_resp:
            response = ml_resp
            source   = f"ml ({conf:.0%})"

    # 4 — Groq AI
    if not response:
        print("  Thinking...", end="\r", flush=True)
        response = ask_groq(user_input, name, SYSTEM_PROMPT)
        source   = "ai"
        print(" " * 20, end="\r", flush=True)

    # Safety net
    if not response:
        response = "Sorry, I couldn't understand that. Please try rephrasing!"
        source   = "fallback"

    # ── Output ─────────────────────────────────────────────────────────────────
    print(f"\n{BOT_NAME}: {response}\n")

    # ── Save to history ────────────────────────────────────────────────────────
    chat_log.append({
        "user"  : user_input,
        "bot"   : response,
        "source": source,
        "time"  : datetime.datetime.now().strftime("%H:%M:%S"),
    })
