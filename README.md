# 🤖 SmartBot — AI-Powered Chatbot

SmartBot is a hybrid chatbot that combines instant rule-based answers, a trained
machine learning intent classifier, and Groq AI (LLaMA 3) to answer almost any
question — not just a fixed list of topics.

---

## ✨ Features

- **Rule-based responses** for real-time data (time, date)
- **Keyword matching** for instant replies to common topics (greetings, study, career, jokes, etc.)
- **ML intent classifier** trained on your own `intents.json`
- **Groq AI fallback** — answers anything that doesn't match the above
- **Spell correction** using TextBlob
- **Conversation memory** — Groq remembers the last few turns for natural follow-ups
- **Chat history** viewable anytime with the `history` command

---

## 📁 Project Files

 File                        Purpose 

 train.py                     Trains the ML model from `intents.json` and saves it 
 chatbot.py                   Main chatbot — run this to chat 
 intents.json                 Training data (tags, patterns, responses)
 model_pipeline.pkl           Auto-generated trained model (after running `train.py`)



## ⚙️ Setup

### 1. Install dependencies

bash
pip install groq textblob scikit-learn nltk
python -m textblob.download_corpora


### 2. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Create an API key
4. Open chatbot.py and paste it here:

python
GROQ_API_KEY = "your_api_key_here"   # 🔑 replace with your real key


### 3. Train the model

Run this once (and again any time you edit `intents.json`):

bash
python train.py


This creates `model_pipeline.pkl`, `model.pkl`, and `vectorizer.pkl`.

### 4. Run the chatbot

bash
python chatbot.py




## 💬 How to Use

When the bot starts, type your name and start chatting.

**Special commands:**

 Command                          Action 

 history                          Shows your full conversation so far 
 clear                            Clears the terminal screen 
 quit or exit                     Ends the chat 



## 🧠 How It Works

Every message goes through 4 layers, in order, until one answers it:


User message
     │
     ▼
1) Rule-based     → time, date, exit phrases
     │ no match
     ▼
2) Keyword map    → hi, study, career, jokes, etc. (instant, offline)
     │ no match
     ▼
3) ML model       → trained on intents.json (offline)
     │ low confidence
     ▼
4) Groq AI        → answers literally anything (needs internet)


This keeps common questions fast and free (no API calls), while Groq AI
handles everything else.



## 📝 Customizing `intents.json`

To teach the bot new topics, add a new block to `intents.json:

json
{
  "tag": "your_topic_name",
  "patterns": ["things users might type", "another phrasing", "..."],
  "responses": ["First possible reply", "Second possible reply"]
}


**Tips:**
- Add 10+ patterns per intent for better accuracy
- Add 3-5 varied responses so replies don't feel repetitive
- **Always re-run `python train.py'** after editing `intents.json'


## 🔧 Configuration

All key settings are at the top of `chatbot.py`:

python
GROQ_API_KEY  = "your_api_key_here"     # Your Groq API key
GROQ_MODEL    = "llama-3.1-8b-instant"  # Groq model used for fallback answers
ML_CONFIDENCE = 0.25                    # Minimum confidence for ML model to answer
BOT_NAME      = "SmartBot"              # Bot's display name




## 🐛 Troubleshooting

**"model_pipeline.pkl not found"**
→ Run `python train.py' first.

**Bot gives no response / blank replies**
→ Check your `GROQ_API_KEY' is set correctly and you have internet access.

**"SyntaxError: break outside loop"**
→ Indentation issue — make sure all pipeline code is indented inside `while True:'.

**Bot gives wrong/irrelevant answers from ML model**
→ Add more patterns to that intent in `intents.json' and retrain.

**Spell correction is slow**
→ TextBlob's correction can lag on long sentences; this is expected behavior.


## 📦 Requirements
groq
textblob
scikit-learn
nltk




## 📄 License

Free to use and modify for personal or educational projects.
