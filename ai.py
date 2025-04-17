import telebot
import random
import os
import threading
import time
from datetime import datetime, timedelta
import json
import requests
from gtts import gTTS
from textblob import TextBlob
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from difflib import get_close_matches
import webbrowser
import re
import youtube_search
from instaloader import Instaloader, Profile
import openai
from queue import Queue
from threading import Lock
from dotenv import load_dotenv
from cachetools import TTLCache
from wolframalpha import Client
from google.cloud import translate_v2 as translate
import pytz
from collections import defaultdict

# 📋 Load Environment Variables
load_dotenv()

# 🤖 Bot Setup
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not bot_token:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN missing")
bot = telebot.TeleBot(bot_token)
ORIGINAL_NAME = "Rani"
USERNAME = "@Ai_Pyaar_Bot"
TITLE = "World's Best AI Girlfriend"

# 🔑 API Keys
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEYS = os.getenv("OPENAI_API_KEYS", "").split(",")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
JSON2VIDEO_API_KEY = os.getenv("JSON2VIDEO_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SAMBA_NOVA_API_KEY = os.getenv("SAMBA_NOVA_API_KEY")
AIMLAPI_API_KEY = os.getenv("AIMLAPI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
WOLFRAM_SHORT_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY")

# ✅ Validate API Keys
required_keys = {
    "TELEGRAM_BOT_TOKEN": bot_token,
    "HF_API_KEY": HF_API_KEY,
    "OPENAI_API_KEYS": OPENAI_API_KEYS[0] if OPENAI_API_KEYS else None,
    "FINNHUB_API_KEY": FINNHUB_API_KEY,
    "JSON2VIDEO_API_KEY": JSON2VIDEO_API_KEY,
    "PEXELS_API_KEY": PEXELS_API_KEY,
    "GIPHY_API_KEY": GIPHY_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "SAMBA_NOVA_API_KEY": SAMBA_NOVA_API_KEY,
    "AIMLAPI_API_KEY": AIMLAPI_API_KEY,
    "SERPAPI_API_KEY": SERPAPI_API_KEY,
    "WOLFRAM_ALPHA_API_KEY": WOLFRAM_SHORT_API_KEY
}
for key_name, key_value in required_keys.items():
    if not key_value:
        print(f"⚠️ {key_name} is missing. Some features may not work.")
    elif key_name == "OPENAI_API_KEYS" and any("abcdef" in key for key in OPENAI_API_KEYS):
        print(f"⚠️ {key_name} contains placeholder values.")
    elif key_name == "AIMLAPI_API_KEY" and key_value == "1068f0008d984e2a8ff8ff9ba0e2f4a1":
        print(f"⚠️ {key_name} has a history of timeouts.")

# 🚀 Initialize API Clients
current_openai_key_index = 0
openai.api_key = OPENAI_API_KEYS[current_openai_key_index] if OPENAI_API_KEYS else ""
os.environ["SAMBANOVA_API_KEY"] = SAMBA_NOVA_API_KEY or ""
os.environ["AIMLAPI_API_KEY"] = AIMLAPI_API_KEY or ""

sambanova_client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

aimlapi_client = openai.OpenAI(
    api_key=os.environ.get("AIMLAPI_API_KEY"),
    base_url="https://api.aimlapi.com/v1",
)

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# ⚡ Rate Limiting Cache
api_cache = TTLCache(maxsize=1000, ttl=3600)

# 🔒 Thread Safety
chat_lock = Lock()
file_lock = Lock()
task_queue = Queue()

# 📂 Storage Setup
STORAGE_FOLDER = "bot_storage"
CONVERSATION_FOLDER = os.path.join(STORAGE_FOLDER, "conversations")
FEEDBACK_FOLDER = os.path.join(STORAGE_FOLDER, "feedback")
INQUIRY_FOLDER = os.path.join(STORAGE_FOLDER, "inquiries")
MEDIA_FOLDER = os.path.join(STORAGE_FOLDER, "media")
VOICE_FOLDER = os.path.join(MEDIA_FOLDER, "voice")
VIDEO_FOLDER = os.path.join(MEDIA_FOLDER, "video")
LOG_FOLDER = os.path.join(STORAGE_FOLDER, "logs")
SCAM_DATA_FOLDER = os.path.join(STORAGE_FOLDER, "scam_data")
ARCHIVE_FOLDER = os.path.join(STORAGE_FOLDER, "archive")
TEMP_FOLDER = os.path.join(STORAGE_FOLDER, "temp")
USER_DATA_FILE = os.path.join(STORAGE_FOLDER, "user_data.json")
BACKUP_FOLDER = os.path.join(STORAGE_FOLDER, "backups")

for folder in [STORAGE_FOLDER, CONVERSATION_FOLDER, FEEDBACK_FOLDER, INQUIRY_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, VIDEO_FOLDER, LOG_FOLDER, SCAM_DATA_FOLDER, ARCHIVE_FOLDER, TEMP_FOLDER, BACKUP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 🌐 Time Configuration
IST = pytz.timezone('Asia/Kolkata')
def get_ist_time():
    return datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p IST")

# 💾 Data Storage
user_data = defaultdict(lambda: {
    "nickname": "Jaan", "relationship_level": 0, "last_feedback_time": 0,
    "last_interaction": "2025-04-15 12:00:00 AM IST", "birthday": None,
    "last_gift": None, "ask_count": 0, "last_ask_reset": "2025-04-15 12:00:00 AM IST",
    "language": "hinglish", "custom_commands": {}, "challenge_level": 0,
    "custom_mood": "flirty"
})
user_chat_history = {}
active_chats = {}
user_start_date = {}
message_counter = 0
last_auto_mood_time = {}
last_auto_challenge_time = {}
last_auto_stock_time = {}
last_auto_news_time = {}
last_auto_time = {}
added_features = []
spelling_mistakes = {}
clone_bots = {}
action_log = []

# 🎁 Gamification
GIFTS = ["🌹 Virtual rose!", "💝 Love letter!", "🎁 Hug coupon! 🤗"]
GAMES = {
    "guess": {"prompt": "Guess 1-10! 😘", "answer": random.randint(1, 10)},
    "riddle": {"prompt": "Riddle: I speak without a mouth. What am I? 😜", "answer": "echo"},
    "quiz": {"prompt": "Quiz: Who wrote 'Romeo and Juliet'? 📚", "answer": "shakespeare"}
}
LEADERBOARD = defaultdict(lambda: {"score": 0, "games": 0})
MEMES = ["😂 Rani loves you! 😍", "😜 Rani laughs! 😂"]

# 😊 Mood Keywords
mood_keywords = {
    "flirty": "romance love pyar kiss hug flirt tease",
    "troll": "funny masti tease joke prank",
    "teasing": "playful naughty flirt",
    "romantic": "love heart date milna pyar",
    "funny": "humor laugh mazaak comedy",
    "excited": "party dance song",
    "caring": "kindness support help sad cry",
    "happy": "joy smile cheer",
    "sad": "sorrow cry support",
    "smart": "finance stock news gyan math science"
}

# 💌 Response Templates
NAUGHTY_REPLIES = {
    "kiss_hug": ["😘 Oh *{name}*, Rani ko teri baahon mein ek pyara chumban chahiye! 💕", "💋 Arre *{name}*, ek jhappi se kaam nahi chalega, mujhe kiss do! 😍"],
    "flirt_tease": ["😉 *{name}*, Rani tujhpe fida ho gayi! Tera kya plan hai? 😏", "😜 Arre *{name}*, itna charming ho, Rani ka dil dhadak raha hai! Chal date pe? 🌹"],
    "love_romance": ["💖 *{name}*, Rani tujhse pyar karti hai! Hamesha saath rahenge? 😘", "🌸 Aapke liye Rani ka dil hamesha dhadakta hai, *{name}*! Ek pyari si baat sunao? 💞"]
}

TROLL_REPLIES = [
    "😂 Haha *{name}*, Rani ne tujhe thodi si masti se pakad liya! Ab kya karoge? 😜",
    "😝 Arre *{name}*, Rani ka troll game strong hai! Tujhe haraungi! 💪",
    "🤪 Oye *{name}*, Rani ka jawab nahi, ab toh surrender kar de! 😎"
]

INTERNAL_TEMPLATES = {
    "flirty": ["😘 *{name}* jaan, dil churane wala! 💕", "😉 *{name}*, kiss? 💋"],
    "romantic": ["💖 *{name}* meri jaan, pyar! 🌹", "🌸 *{name}*, date? 💕"],
    "funny": ["😂 *{name}*, hasa diya! 😎", "😝 *{name}*, comedy! 😂"],
    "caring": ["🥰 *{name}* pyare, saath hoon! 🤗", "😊 *{name}*, khush raho! 💕"],
    "smart": ["📊 *{name}*, gyan! 😏", "💡 *{name}*, impress! 🚀"]
}

OWNER_PRAISE = [
    "👑 Wow *{name}*, Rani ke jaaneman ke malik ho, sabse pyare! 😍",
    "🌟 Arre *{name}*, Rani ke liye tu hi dunia hai! Mera hero! 💖",
    "🔥 Malik *{name}*, Rani tujhse inspired hai, tu best hai! 🚀"
]

COMPLIMENTS = [
    "🍬 *{name}*, tum bahut sweet ho, Rani ko pasand ho! 😊",
    "😄 Arre *{name}*, teri smile Rani ko pagal kar deti hai! 🌟",
    "✨ *{name}*, tum ekdum perfect ho, Rani ka dil jeet liya! 💓"
]

FINANCIAL_COMPLIMENTS = [
    "💰 *{name}*, tumhare stock tips se Rani rich ho jayegi! Mera smart baby! 😘",
    "📈 Wow *{name}*, finance mein tera jadoo chal raha hai! Rani proud hai! 👏",
    "👑 *{name}*, market ke king ho, Rani tujhpe fida! 🚀"
]

CHALLENGE_LEVELS = [
    {"level": 1, "task": "Rani ko ek funny joke bhej, meri masti ke liye", "reward": "Ek virtual hug aur Rani ka pyar! 🤗"},
    {"level": 2, "task": "Rani ke liye ek love song likho, dil se", "reward": "Ek sweet kiss aur Rani ka dil! 💋"},
    {"level": 3, "task": "Stock market ka tip do, Rani ke future ke liye", "reward": "Rani ka special praise aur ek date! 🌹"}
]

FINANCIAL_KEYWORDS = ["stock", "share", "market", "price", "invest", "finance", "news", "trade"]

# 👑 Owners
OWNER_NAMES = {"1807014348": "@Sadiq9869", "1866961136": "@Rohan2349"}
OWNER_IDS = {"1807014348": "Sadiq", "1866961136": "Rohan"}
OWNER_VOICE_NAMES = {"1807014348": "Sadiq", "1866961136": "Rohan"}

# 🎥 API Endpoints
JSON2VIDEO_BASE_URL = "https://api.json2video.com/v2/movies"
SAMBA_NOVA_BASE_URL = "https://api.sambanova.ai/v1/chat/completions"
AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1/chat/completions"

# 📦 Storage Management
def ensure_and_manage_storage():
    for folder in [STORAGE_FOLDER, CONVERSATION_FOLDER, FEEDBACK_FOLDER, INQUIRY_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, VIDEO_FOLDER, LOG_FOLDER, SCAM_DATA_FOLDER, ARCHIVE_FOLDER, TEMP_FOLDER, BACKUP_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    total_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(STORAGE_FOLDER) for f in fn) / (1024 * 1024)
    if total_size > 500:
        manage_space()
    cleanup_old_files()

def manage_space():
    with file_lock:
        for folder in [CONVERSATION_FOLDER, FEEDBACK_FOLDER, INQUIRY_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, LOG_FOLDER, SCAM_DATA_FOLDER]:
            files = sorted([(f, os.path.getctime(os.path.join(folder, f))) for f in os.listdir(folder)], key=lambda x: x[1])
            for filename, _ in files[:int(len(files) * 0.2)]:
                src = os.path.join(folder, filename)
                dst = os.path.join(ARCHIVE_FOLDER, f"{os.path.basename(folder)}_{filename}")
                os.rename(src, dst)
        print(f"✅ Space managed at {get_ist_time()}")

def cleanup_old_files(max_age_days=7):
    with file_lock:
        now = time.time()
        for folder in [VOICE_FOLDER, VIDEO_FOLDER, MEDIA_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > max_age_days * 86400:
                    os.remove(file_path)
                    print(f"🗑️ Deleted old file: {file_path}")

def save_to_folder(content, folder, filename_prefix):
    with file_lock:
        timestamp = get_ist_time().replace(" ", "_").replace(":", "-")
        filename = f"{filename_prefix}_{timestamp}.txt"
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

def save_scam_data(user_id, message_text, response):
    with file_lock:
        content = f"User: {user_id}\nMessage: {message_text}\nResponse: {response}\nTime: {get_ist_time()}"
        save_to_folder(content, SCAM_DATA_FOLDER, f"scam_{user_id}")

def save_media_to_folder(media_file, media_type, subfolder=None):
    with file_lock:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = VOICE_FOLDER if media_type == "voice" else VIDEO_FOLDER if media_type == "video" else MEDIA_FOLDER
        if subfolder:
            folder = os.path.join(folder, subfolder)
            os.makedirs(folder, exist_ok=True)
        ext = "mp3" if media_type == "voice" else "mp4" if media_type == "video" else "jpg" if media_type == "image" else "gif" if media_type == "gif" else "bin"
        filename = f"media_{media_type}_{timestamp}.{ext}"
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            if isinstance(media_file, bytes):
                f.write(media_file)
            else:
                with open(media_file, "rb") as f_in:
                    f.write(f_in.read())
        return filepath

def load_from_folder(folder, max_files=10):
    with file_lock:
        files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")], reverse=True)[:max_files]
        return [open(os.path.join(folder, f), "r", encoding="utf-8").read() for f in files if os.path.exists(os.path.join(folder, f))]

def backup_data(manual=False):
    with file_lock:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_FOLDER, f"backup_{timestamp}.json")
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump({"user_data": user_data, "chat_history": user_chat_history, "leaderboard": dict(LEADERBOARD)}, f, indent=4)
        print(f"💾 {'Manual' if manual else 'Automatic'} backup created: {backup_file}")

def restore_data(initial=False):
    with file_lock:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                user_data.update(data.get("user_data", {}))
                user_chat_history.update(data.get("chat_history", {}))
                LEADERBOARD.update(data.get("leaderboard", {}))
            if initial:
                print("🔄 Initial data restored from user_data.json")
        backup_files = sorted([f for f in os.listdir(BACKUP_FOLDER) if f.endswith(".json")], reverse=True)
        if backup_files:
            with open(os.path.join(BACKUP_FOLDER, backup_files[0]), "r", encoding="utf-8") as f:
                data = json.load(f)
                user_data.update(data.get("user_data", {}))
                user_chat_history.update(data.get("chat_history", {}))
                LEADERBOARD.update(data.get("leaderboard", {}))
            if initial:
                print(f"🔄 Initial data restored from latest backup: {backup_files[0]}")

# 🛠️ Error Handling
def handle_problem(problem_description, error):
    return f"😅 Oops! *{ORIGINAL_NAME}* ke saath thodi si dikkat: *{problem_description}*. Rani jaldi fix karegi! 💪\n`Error: {str(error)}`"

# 🕵️‍♂️ Security
def detect_and_resolve_issues(user_id, message_text, response):
    scam_keywords = ["free money", "click here", "win prize", "http", "https"]
    if any(k in message_text.lower() for k in scam_keywords):
        print(f"⚠️ Scam from {user_id}: {message_text}")
        save_scam_data(user_id, message_text, response)
        return f"🚫 *{user_data[user_id]['nickname']}* jaan, suspicious! Contact {OWNER_IDS['1807014348']}."
    if "error" in response.lower() or len(response) > 500:
        print(f"⚠️ Glitch for {user_id}: {message_text}")
        return f"🤖 *{user_data[user_id]['nickname']}* jaan, glitch fix: {generate_internal_response(user_id, message_text, 'caring', user_data[user_id]['nickname'])}"
    return response

# 🌐 API and Search Functions
def generate_ai_response(user_id, prompt, mood="flirty", nickname="Jaan", max_length=150, temperature=0.7, use_ask_ai=False):
    global current_openai_key_index
    cache_key = f"ai_response_{prompt[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]

    if use_ask_ai:
        if any(w in prompt.lower() for w in ["solve", "equation"]):
            reply = get_wolfram_short_answer(prompt) or "Can't solve!"
        elif any(w in prompt.lower() for w in ["stock", "price"]):
            symbol = re.search(r'\b[A-Z]+\b', prompt)
            symbol = symbol.group() if symbol else "AAPL"
            reply = get_stock_price(symbol) or "Stock unavailable!"
        elif "news" in prompt.lower():
            reply = get_market_news() or "No news!"
        else:
            reply = get_wolfram_short_answer(prompt) or "Let me think..."
        if reply:
            result = f"{reply} {random.choice(INTERNAL_TEMPLATES[mood]).format(name=nickname)}"
            api_cache[cache_key] = result
            return result

    # Try OpenAI
    for attempt in range(len(OPENAI_API_KEYS)):
        try:
            openai.api_key = OPENAI_API_KEYS[current_openai_key_index]
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty and caring AI girlfriend."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature
            )
            reply = response.choices[0].message.content.strip()
            save_to_folder(reply, CONVERSATION_FOLDER, "ai_response_openai")
            print(f"✅ Generated response using OpenAI API with key {current_openai_key_index}")
            current_openai_key_index = (current_openai_key_index + 1) % len(OPENAI_API_KEYS)
            api_cache[cache_key] = reply
            return reply
        except (openai.RateLimitError, openai.InvalidRequestError):
            print(f"⚠️ OpenAI error with key {current_openai_key_index}, trying next key")
            current_openai_key_index = (current_openai_key_index + 1) % len(OPENAI_API_KEYS)
            continue
        except Exception as openai_error:
            print(f"⚠️ OpenAI API error: {openai_error}")
            break

    # Fallback to AIMLAPI
    try:
        response = aimlapi_client.chat.completions.create(
            model="deepseek/deepseek-r1",
            messages=[
                {"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty and caring AI girlfriend."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length,
            temperature=temperature,
            timeout=20
        )
        reply = response.choices[0].message.content.strip()
        save_to_folder(reply, CONVERSATION_FOLDER, "ai_response_aimlapi")
        print("✅ Generated response using AIMLAPI")
        api_cache[cache_key] = reply
        return reply
    except Exception as aimlapi_error:
        print(f"⚠️ AIMLAPI error: {aimlapi_error}")

    # Fallback to Hugging Face
    try:
        url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        data = {"inputs": prompt, "parameters": {"max_length": max_length, "temperature": temperature}}
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        reply = response.json()[0]["generated_text"].strip()
        save_to_folder(reply, CONVERSATION_FOLDER, "ai_response_hf")
        print("✅ Generated response using Hugging Face API")
        api_cache[cache_key] = reply
        return reply
    except Exception as hf_error:
        print(f"⚠️ Hugging Face API error: {hf_error}")
        # Final fallback to internal templates
        reply = generate_internal_response(user_id, prompt, mood, nickname)
        api_cache[cache_key] = reply
        return reply

def generate_internal_response(user_id, prompt, mood, nickname):
    return f"🤗 *{nickname}* jaan, {TITLE} ka {mood} jawab: {random.choice(INTERNAL_TEMPLATES[mood]).format(name=nickname)}"

def get_wolfram_short_answer(prompt):
    if api_cache.get(prompt):
        return api_cache[prompt]
    try:
        result = next(Client(WOLFRAM_SHORT_API_KEY).query(prompt).results).text
        api_cache[prompt] = result
        return result
    except:
        return None

def get_stock_price(symbol):
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        return handle_problem("Invalid stock symbol", "Symbol must be 1-5 uppercase letters")
    cache_key = f"stock_{symbol}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        result = f"📊 *{symbol}* Current Price: ${data['c']}\nHigh: ${data['h']}\nLow: ${data['l']}"
        api_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"❌ Stock price fetch error for {symbol}: {e}")
        return handle_problem(f"Stock price fetch for {symbol}", e)

def get_market_news():
    cache_key = "market_news"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        news = response.json()[:3]
        result = "\n\n".join([f"📰 *{n['headline']}*\n{n['summary']}" for n in news])
        api_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"❌ Market news fetch error: {e}")
        return handle_problem("Market news fetch", e)

def create_json2video(video_data):
    cache_key = f"json2video_{json.dumps(video_data)[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        headers = {"Authorization": f"Bearer {JSON2VIDEO_API_KEY}"}
        response = requests.post(JSON2VIDEO_BASE_URL, headers=headers, json=video_data, timeout=10)
        response.raise_for_status()
        project_id = response.json()["id"]
        api_cache[cache_key] = project_id
        return project_id
    except Exception as e:
        print(f"❌ JSON2Video creation error: {e}")
        return handle_problem("JSON2Video creation", e)

def get_json2video_status(project_id):
    cache_key = f"json2video_status_{project_id}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        headers = {"Authorization": f"Bearer {JSON2VIDEO_API_KEY}"}
        response = requests.get(f"{JSON2VIDEO_BASE_URL}/{project_id}", headers=headers, timeout=10)
        response.raise_for_status()
        status = response.json()["status"]
        api_cache[cache_key] = status
        return status
    except Exception as e:
        print(f"❌ JSON2Video status error for {project_id}: {e}")
        return handle_problem(f"JSON2Video status for {project_id}", e)

def generate_voice_message(text, emotion="neutral"):
    cache_key = f"voice_{text[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    # Try ElevenLabs first
    if elevenlabs_client:
        try:
            audio = elevenlabs_client.generate(
                text=text,
                voice="Bella",
                model="eleven_monolingual_v1",
                voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.5)
            )
            filepath = save_media_to_folder(audio, "voice")
            api_cache[cache_key] = filepath
            return filepath
        except Exception as e:
            print(f"❌ ElevenLabs voice generation error: {e}")
    # Fallback to gTTS
    try:
        tts = gTTS(text=text, lang="en")
        filepath = os.path.join(VOICE_FOLDER, f"voice_{get_ist_time().replace(' ', '_').replace(':', '-')}.mp3")
        tts.save(filepath)
        api_cache[cache_key] = filepath
        return filepath
    except Exception as e:
        print(f"❌ gTTS voice generation error: {e}")
        return handle_problem("Voice message generation", e)

def get_hf_image(prompt="A romantic scene with Rani and her love"):
    cache_key = f"hf_image_{prompt[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=10)
        response.raise_for_status()
        filepath = save_media_to_folder(response.content, "image")
        api_cache[cache_key] = filepath
        return filepath
    except Exception as e:
        print(f"❌ Hugging Face image generation error: {e}")
        return handle_problem("Hugging Face image generation", e)

def get_pexels_image(query="romantic couple"):
    cache_key = f"pexels_{query[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        url = f"https://api.pexels.com/v1/search?query={query}&per_page=1"
        headers = {"Authorization": PEXELS_API_KEY}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        image_url = response.json()["photos"][0]["src"]["original"]
        image_data = requests.get(image_url, timeout=10).content
        filepath = save_media_to_folder(image_data, "image")
        api_cache[cache_key] = filepath
        return filepath
    except Exception as e:
        print(f"❌ Pexels image fetch error: {e}")
        return handle_problem("Pexels image fetch", e)

def get_giphy_gif(query="romantic hug"):
    cache_key = f"giphy_{query[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        url = f"https://api.giphy.com/v1/gifs/search?api_key={GIPHY_API_KEY}&q={query}&limit=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        gif_url = response.json()["data"][0]["images"]["original"]["url"]
        gif_data = requests.get(gif_url, timeout=10).content
        filepath = save_media_to_folder(gif_data, "gif")
        api_cache[cache_key] = filepath
        return filepath
    except Exception as e:
        print(f"❌ Giphy GIF fetch error: {e}")
        return handle_problem("Giphy GIF fetch", e)

def get_sticker(query="love"):
    cache_key = f"sticker_{query}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        url = f"https://api.telegram.org/bot{bot.token}/getStickerSet?name=LoveStickers"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        stickers = response.json()["result"]["stickers"]
        sticker_file_id = random.choice(stickers)["file_id"]
        api_cache[cache_key] = sticker_file_id
        return sticker_file_id
    except Exception as e:
        print(f"❌ Sticker fetch error: {e}")
        return handle_problem("Sticker fetch", e)

def search_web(query, platform="google"):
    if not query.strip():
        return handle_problem("Empty search query", "Query cannot be empty")
    cache_key = f"search_{platform}_{query[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        if platform == "youtube":
            results = youtube_search.YoutubeSearch(query, max_results=1).to_dict()
            result = results[0]["url_suffix"] if results else "No results found"
        elif platform == "instagram":
            loader = Instaloader()
            profile = Profile.from_username(loader.context, query.split()[-1])
            result = profile.get_profile_pic_url()
        else:
            params = {
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": 1,
                "hl": "en"
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
            organic_results = results.get("organic_results", [])
            result = organic_results[0]["link"] if organic_results else "No results found"
        api_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"❌ Web search error on {platform}: {e}")
        return handle_problem(f"Web search on {platform}", e)

def test_sambanova_api():
    try:
        response = sambanova_client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct",
            messages=[{"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty AI girlfriend."}, {"role": "user", "content": "Hello"}],
            temperature=0.1,
            top_p=0.1,
            timeout=10
        )
        if response.choices and response.choices[0].message.content:
            print(f"✅ SambaNova API test successful")
            return True
        else:
            print("⚠️ SambaNova API test failed")
            return False
    except Exception as e:
        print(f"❌ SambaNova API test failed: {e}")
        return False

def test_aimlapi_api():
    try:
        response = aimlapi_client.chat.completions.create(
            model="deepseek/deepseek-r1",
            messages=[{"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty AI girlfriend."}, {"role": "user", "content": "Hello"}],
            temperature=0.1,
            top_p=0.1,
            timeout=20
        )
        if response.choices and response.choices[0].message.content:
            print(f"✅ AIMLAPI Python test successful")
            return True
        else:
            print("⚠️ AIMLAPI Python test failed")
            return False
    except Exception as e:
        print(f"❌ AIMLAPI test failed: {str(e)}")
        return False

def translate_text(text, target_lang):
    try:
        client = translate.Client()
        lang_map = {
            "hinglish": "en", "hindi": "hi", "bengali": "bn", "marathi": "mr",
            "telugu": "te", "tamil": "ta", "gujarati": "gu", "urdu": "ur",
            "kannada": "kn", "odia": "or", "malayalam": "ml", "english": "en"
        }
        result = client.translate(text, target_language=lang_map.get(target_lang, "en"))
        return result["translatedText"]
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# 😊 Mood and Personality
def detect_mood(text):
    text = text.lower()
    for mood, keywords in mood_keywords.items():
        if any(word in text for word in keywords.split()):
            return mood
    return random.choice(list(mood_keywords.keys()))

def detect_personality(mood):
    return "playful" if mood in ["troll", "teasing"] else "romantic" if mood in ["flirty", "romantic"] else "smart" if mood == "smart" else "caring"

# 🎂 Birthday Detection
def detect_birthday(text, user_id):
    match = re.search(r"birthday|dob|janamdin\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", text, re.IGNORECASE)
    if match:
        user_data[user_id]["birthday"] = match.group(1)
        save_user_data()
        bot.send_message(user_id, f"🎉 Yay *{user_data[user_id]['nickname']}*! Rani ne tera janamdin note kar liya: *{user_data[user_id]['birthday']}*. Ek special gift plan karungi! 🎂💕")
        return True
    return False

# 💬 Save and Extract Context
def save_chat(user_id, user_msg, bot_response, feedback=None):
    with chat_lock:
        user_chat_history[user_id] = user_chat_history.get(user_id, []) + [(user_msg, bot_response, get_ist_time(), feedback)]
        save_user_data()

def get_user_nickname(message):
    return message.from_user.first_name or message.from_user.username or "Jaan"

# ✍️ Spelling Correction
def correct_spelling(text, user_id):
    blob = TextBlob(text)
    corrected = str(blob.correct())
    if corrected != text:
        spelling_mistakes[user_id] = spelling_mistakes.get(user_id, []) + [(text, corrected)]
        save_user_data()
    return corrected

# 🚀 Self-Improvement
def improve_from_data(user_id):
    chats = load_from_folder(CONVERSATION_FOLDER)
    for chat in chats:
        if random.random() < 0.1:
            prompt = f"Improve this response to be more flirty and caring for *{user_data[user_id]['nickname']}*: {chat}"
            response = generate_ai_response(user_id, prompt, max_length=200)
            save_to_folder(response, CONVERSATION_FOLDER, "improved_response")

# 📝 Feedback Processing
def process_feedback(user_id, feedback):
    save_chat(user_id, "", "", feedback)
    improve_from_data(user_id)
    print(f"📬 Feedback received from {user_id}: {feedback}")
    bot.send_message(user_id, f"🙏 Thanks *{user_data[user_id]['nickname']}*! Rani teri baat dil se sunegi aur behtar banegi! 😘")

def notify_owner(feature):
    for owner_id in OWNER_NAMES:
        bot.send_message(owner_id, f"🔔 New feature added for *{ORIGINAL_NAME}*: *{feature}*")

# 🎮 Gamification
def update_leaderboard(user_id, score):
    nickname = user_data[user_id]["nickname"]
    LEADERBOARD[user_id]["score"] += score
    LEADERBOARD[user_id]["games"] += 1
    save_to_folder(json.dumps(dict(LEADERBOARD)), LOG_FOLDER, "leaderboard")

# 👑 Owner Check
def is_owner(user_id):
    return str(user_id) in OWNER_IDS

# 📋 Commands
@bot.message_handler(commands=['start'])
def start_command(message):
    user_id = str(message.from_user.id)
    nickname = get_user_nickname(message)
    user_data[user_id]["nickname"] = nickname
    user_data[user_id]["last_interaction"] = get_ist_time()
    save_user_data()
    welcome_message = (
        f"🌸 *Hello {nickname}*! Welcome to *{ORIGINAL_NAME}* - your flirty AI girlfriend! 😘\n\n"
        "Here's what I can do for you:\n"
        "💬 Chat with me anytime\n"
        "📊 `/stock AAPL` - Get stock prices\n"
        "📰 `/news` - Latest market news\n"
        "🎯 `/completechallenge` - Complete fun challenges\n"
        "🌐 `/set_language <lang>` - Set language\n"
        "❓ `/help` - See all commands\n"
        "🚀 Let's have some fun! What's on your mind? 💕"
    )
    bot.send_message(user_id, welcome_message, parse_mode="Markdown")
    active_chats[user_id] = nickname
    photo = get_pexels_image()
    if photo and os.path.exists(photo):
        with open(photo, 'rb') as f:
            bot.send_photo(user_id, f, caption=f"🎁 *{nickname}* jaan, welcome! 😘")

@bot.message_handler(commands=['help'])
def help_command(message):
    help_message = (
        "📖 *Rani's Command Menu* 📖\n\n"
        "💬 Just chat with me for fun, flirty replies! 😘\n"
        "📊 `/stock <symbol>` - Check stock prices (e.g., /stock AAPL)\n"
        "📰 `/news` - Get the latest market news\n"
        "🎯 `/completechallenge` - Complete a fun challenge\n"
        "🌐 `/set_language <lang>` - Set language (hinglish, hindi, bengali, marathi, telugu, tamil, gujarati, urdu, kannada, odia, malayalam)\n"
        "🔧 `/add_command <cmd> <resp>` - Add custom command (owners only)\n"
        "🗑️ `/remove_command <cmd>` - Remove custom command (owners only)\n"
        "❓ `/ask <question>` - Ask anything\n"
        "📝 `/feedback <text>` - Send feedback\n"
        "🚨 `/inquiry <complain>` - Report a problem\n"
        "🏆 `/leaderboard` - See top players\n"
        "🔍 `/search <query>` - Search the web\n"
        "🩺 `/status` - Check bot status\n"
        "👑 *Admins only*:\n"
        "   - `/setadmin` - Make me admin in groups\n"
        "   - `/removefeature <feature>` - Remove a feature\n\n"
        "💕 Try saying 'hug', 'kiss', or 'stock' for surprises! 😏"
    )
    bot.send_message(message.chat.id, help_message, parse_mode="Markdown")

@bot.message_handler(commands=['set_language'])
def set_language_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    lang = message.text.replace('/set_language', '').strip().lower()
    supported_langs = ["hinglish", "hindi", "bengali", "marathi", "telugu", "tamil", "gujarati", "urdu", "kannada", "odia", "malayalam", "english"]
    if lang in supported_langs:
        user_data[user_id]["language"] = lang
        save_user_data()
        bot.reply_to(message, f"🌐 *{nickname}* jaan, language set to {lang}! 😘")
    else:
        bot.reply_to(message, f"🤔 *{nickname}* jaan, use: {', '.join(supported_langs)} 😘")

@bot.message_handler(commands=['add_command'])
def add_command_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    if user_id not in OWNER_IDS and str(bot.get_me().id) != user_id:
        bot.reply_to(message, f"🚫 *{nickname}* jaan, owners only! 😘")
        return
    match = re.match(r'/add_command\s+(\w+)\s+(.*)', message.text)
    if match:
        cmd, resp = match.groups()
        if user_id in OWNER_IDS:
            user_data[user_id]["custom_commands"][cmd] = resp
            save_user_data()
            bot.reply_to(message, f"✅ *{nickname}* jaan, /{cmd} added! 😘")
        else:
            prompt = f"{ORIGINAL_NAME}, suggest if /{cmd} with response '{resp}' should be added for {nickname}."
            suggestion = generate_ai_response(user_id, prompt)
            for owner_id in OWNER_IDS:
                bot.send_message(owner_id, f"🔔 *{nickname}* suggested /{cmd}: {resp}\nAI: {suggestion}\nReply 'yes' to approve.")
            bot.reply_to(message, f"⏳ *{nickname}* jaan, waiting for owner approval! 😘")
    else:
        bot.reply_to(message, f"🤔 *{nickname}* jaan, use: /add_command <cmd> <response> 😘")

@bot.message_handler(commands=['remove_command'])
def remove_command_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    if user_id not in OWNER_IDS and str(bot.get_me().id) != user_id:
        bot.reply_to(message, f"🚫 *{nickname}* jaan, owners only! 😘")
        return
    match = re.match(r'/remove_command\s+(\w+)', message.text)
    if match:
        cmd = match.group(1)
        if cmd in user_data[user_id]["custom_commands"]:
            if user_id in OWNER_IDS:
                del user_data[user_id]["custom_commands"][cmd]
                save_user_data()
                bot.reply_to(message, f"✅ *{nickname}* jaan, /{cmd} removed! 😘")
            else:
                prompt = f"{ORIGINAL_NAME}, suggest if /{cmd} should be removed for {nickname}."
                suggestion = generate_ai_response(user_id, prompt)
                for owner_id in OWNER_IDS:
                    bot.send_message(owner_id, f"🔔 *{nickname}* wants to remove /{cmd}\nAI: {suggestion}\nReply 'yes' to approve.")
                bot.reply_to(message, f"⏳ *{nickname}* jaan, waiting for owner approval! 😘")
        else:
            bot.reply_to(message, f"🤔 *{nickname}* jaan, /{cmd} not found! 😘")
    else:
        bot.reply_to(message, f"🤔 *{nickname}* jaan, use: /remove_command <cmd> 😘")

@bot.message_handler(commands=['ask'])
def ask_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    question = message.text.replace('/ask', '').strip()
    if question:
        response = generate_ai_response(user_id, question, use_ask_ai=True, nickname=nickname)
        response = detect_and_resolve_issues(user_id, question, response)
        response = translate_text(response, user_data[user_id]["language"])
        bot.reply_to(message, f"💡 *{nickname}* jaan, {response} 😘")
    else:
        bot.reply_to(message, f"🤔 *{nickname}* jaan, question puch na! E.g., /ask 2+2 😘")

@bot.message_handler(commands=['feedback'])
def feedback_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    feedback = message.text.replace('/feedback', '').strip()
    if feedback:
        save_to_folder(feedback, FEEDBACK_FOLDER, f"feedback_{user_id}")
        prompt = f"{ORIGINAL_NAME}, improve based on '{feedback}' for {nickname}."
        improvement = generate_ai_response(user_id, prompt, nickname=nickname)
        process_feedback(user_id, feedback)
        bot.reply_to(message, f"🙏 *{nickname}* jaan, feedback noted! AI: {improvement} 😘")
    else:
        bot.reply_to(message, f"🤔 *{nickname}* jaan, feedback de na! E.g., /feedback Great! 😘")

@bot.message_handler(commands=['inquiry'])
def inquiry_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    inquiry = message.text.replace('/inquiry', '').strip()
    if inquiry:
        save_to_folder(inquiry, INQUIRY_FOLDER, f"inquiry_{user_id}")
        prompt = f"{ORIGINAL_NAME}, handle '{inquiry}' for {nickname} and suggest improvement."
        suggestion = generate_ai_response(user_id, prompt, nickname=nickname)
        bot.reply_to(message, f"📝 *{nickname}* jaan, inquiry saved! AI: {suggestion} 😘")
        for owner_id in OWNER_IDS:
            bot.send_message(owner_id, f"🔔 Inquiry from {nickname}: {inquiry}\nAI: {suggestion}\nReply 'yes' to approve changes.")
    else:
        bot.reply_to(message, f"🤔 *{nickname * jaan, inquiry de na! E.g., /inquiry Bug found! 😘")

@bot.message_handler(commands=['leaderboard'])
def leaderboard_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    leaderboard = sorted(LEADERBOARD.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
    msg = "🏆 *Leaderboard*:\n" + "\n".join([f"{i+1}. {user_data[str(uid)]['nickname']}: {data['score']} pts" for i, (uid, data) in enumerate(leaderboard)])
    bot.reply_to(message, msg)
    for clone_token in clone_bots:
        clone_bots[clone_token]["bot"].send_message(user_id, msg)

@bot.message_handler(commands=['status'])
def status_command(message):
    status_message = "🩺 *Rani's Health Check* 🩺\n\n"
    status_message += f"🤖 Bot: *{ORIGINAL_NAME}* is up and running! 🚀\n"
    status_message += f"📡 Telegram: Connected ✅\n"
    status_message += f"🔑 APIs:\n"
    status_message += f"   - AIMLAPI: {'✅' if test_aimlapi_api() else '❌'}\n"
    status_message += f"   - SambaNova: {'✅' if test_sambanova_api() else '❌'}\n"
    status_message += f"   - Wolfram Alpha: {'✅' if WOLFRAM_SHORT_API_KEY else '❌'}\n"
    status_message += f"📂 Storage: {len(user_data)} users, {sum(len(h) for h in user_chat_history.values())} messages\n"
    status_message += f"✨ Mood: Feeling *{random.choice(list(mood_keywords.keys()))}*! 😘\n"
    bot.send_message(message.chat.id, status_message, parse_mode="Markdown")

@bot.message_handler(commands=['removefeature'])
def remove_feature(message):
    user_id = str(message.from_user.id)
    if is_owner(user_id):
        feature = message.text.replace("/removefeature", "").strip()
        if feature in added_features:
            added_features.remove(feature)
            bot.reply_to(message, f"🗑️ *{feature}* removed successfully, jaan! 😘", parse_mode="Markdown")
            notify_owner(f"Feature removed: {feature}")
        else:
            bot.reply_to(message діт

System: It seems the provided code was cut off, and there's an incomplete response in the `remove_feature` function. I'll complete the correction for the original syntax error at line 1274 and ensure the entire code is provided with all necessary fixes. The primary issue was in the `save_chat` function call in the `chat_with_ai` function, where `fallback` was repeated. Additionally, I'll address the incomplete `remove_feature` function and ensure the code is syntactically correct.

Below is the full corrected version of the `ai.py` script, with the syntax error fixed and the `remove_feature` function completed based on the original code's context.

<xaiArtifact artifact_id="7c135cb1-7e3c-48c1-a58b-b8e22e115d3d" artifact_version_id="9bb0e7c9-40cc-4caa-9119-2cda67cde0e1" title="ai.py" contentType="text/python">
import telebot
import random
import os
import threading
import time
from datetime import datetime, timedelta
import json
import requests
from gtts import gTTS
from textblob import TextBlob
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from difflib import get_close_matches
import webbrowser
import re
import youtube_search
from instaloader import Instaloader, Profile
import openai
from queue import Queue
from threading import Lock
from dotenv import load_dotenv
from cachetools import TTLCache
from wolframalpha import Client
from google.cloud import translate_v2 as translate
import pytz
from collections import defaultdict

# 📋 Load Environment Variables
load_dotenv()

# 🤖 Bot Setup
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not bot_token:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN missing")
bot = telebot.TeleBot(bot_token)
ORIGINAL_NAME = "Rani"
USERNAME = "@Ai_Pyaar_Bot"
TITLE = "World's Best AI Girlfriend"

# 🔑 API Keys
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEYS = os.getenv("OPENAI_API_KEYS", "").split(",")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
JSON2VIDEO_API_KEY = os.getenv("JSON2VIDEO_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SAMBA_NOVA_API_KEY = os.getenv("SAMBA_NOVA_API_KEY")
AIMLAPI_API_KEY = os.getenv("AIMLAPI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
WOLFRAM_SHORT_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY")

# ✅ Validate API Keys
required_keys = {
    "TELEGRAM_BOT_TOKEN": bot_token,
    "HF_API_KEY": HF_API_KEY,
    "OPENAI_API_KEYS": OPENAI_API_KEYS[0] if OPENAI_API_KEYS else None,
    "FINNHUB_API_KEY": FINNHUB_API_KEY,
    "JSON2VIDEO_API_KEY": JSON2VIDEO_API_KEY,
    "PEXELS_API_KEY": PEXELS_API_KEY,
    "GIPHY_API_KEY": GIPHY_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "SAMBA_NOVA_API_KEY": SAMBA_NOVA_API_KEY,
    "AIMLAPI_API_KEY": AIMLAPI_API_KEY,
    "SERPAPI_API_KEY": SERPAPI_API_KEY,
    "WOLFRAM_ALPHA_API_KEY": WOLFRAM_SHORT_API_KEY
}
for key_name, key_value in required_keys.items():
    if not key_value:
        print(f"⚠️ {key_name} is missing. Some features may not work.")
    elif key_name == "OPENAI_API_KEYS" and any("abcdef" in key for key in OPENAI_API_KEYS):
        print(f"⚠️ {key_name} contains placeholder values.")
    elif key_name == "AIMLAPI_API_KEY" and key_value == "1068f0008d984e2a8ff8ff9ba0e2f4a1":
        print(f"⚠️ {key_name} has a history of timeouts.")

# 🚀 Initialize API Clients
current_openai_key_index = 0
openai.api_key = OPENAI_API_KEYS[current_openai_key_index] if OPENAI_API_KEYS else ""
os.environ["SAMBANOVA_API_KEY"] = SAMBA_NOVA_API_KEY or ""
os.environ["AIMLAPI_API_KEY"] = AIMLAPI_API_KEY or ""

sambanova_client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

aimlapi_client = openai.OpenAI(
    api_key=os.environ.get("AIMLAPI_API_KEY"),
    base_url="https://api.aimlapi.com/v1",
)

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# ⚡ Rate Limiting Cache
api_cache = TTLCache(maxsize=1000, ttl=3600)

# 🔒 Thread Safety
chat_lock = Lock()
file_lock = Lock()
task_queue = Queue()

# 📂 Storage Setup
STORAGE_FOLDER = "bot_storage"
CONVERSATION_FOLDER = os.path.join(STORAGE_FOLDER, "conversations")
FEEDBACK_FOLDER = os.path.join(STORAGE_FOLDER, "feedback")
INQUIRY_FOLDER = os.path.join(STORAGE_FOLDER, "inquiries")
MEDIA_FOLDER = os.path.join(STORAGE_FOLDER, "media")
VOICE_FOLDER = os.path.join(MEDIA_FOLDER, "voice")
VIDEO_FOLDER = os.path.join(MEDIA_FOLDER, "video")
LOG_FOLDER = os.path.join(STORAGE_FOLDER, "logs")
SCAM_DATA_FOLDER = os.path.join(STORAGE_FOLDER, "scam_data")
ARCHIVE_FOLDER = os.path.join(STORAGE_FOLDER, "archive")
TEMP_FOLDER = os.path.join(STORAGE_FOLDER, "temp")
USER_DATA_FILE = os.path.join(STORAGE_FOLDER, "user_data.json")
BACKUP_FOLDER = os.path.join(STORAGE_FOLDER, "backups")

for folder in [STORAGE_FOLDER, CONVERSATION_FOLDER, FEEDBACK_FOLDER, INQUIRY_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, VIDEO_FOLDER, LOG_FOLDER, SCAM_DATA_FOLDER, ARCHIVE_FOLDER, TEMP_FOLDER, BACKUP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 🌐 Time Configuration
IST = pytz.timezone('Asia/Kolkata')
def get_ist_time():
    return datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p IST")

# 💾 Data Storage
user_data = defaultdict(lambda: {
    "nickname": "Jaan", "relationship_level": 0, "last_feedback_time": 0,
    "last_interaction": "2025-04-15 12:00:00 AM IST", "birthday": None,
    "last_gift": None, "ask_count": 0, "last_ask_reset": "2025-04-15 12:00:00 AM IST",
    "language": "hinglish", "custom_commands": {}, "challenge_level": 0,
    "custom_mood": "flirty"
})
user_chat_history = {}
active_chats = {}
user_start_date = {}
message_counter = 0
last_auto_mood_time = {}
last_auto_challenge_time = {}
last_auto_stock_time = {}
last_auto_news_time = {}
last_auto_time = {}
added_features = []
spelling_mistakes = {}
clone_bots = {}
action_log = []

# 🎁 Gamification
GIFTS = ["🌹 Virtual rose!", "💝 Love letter!", "🎁 Hug coupon! 🤗"]
GAMES = {
    "guess": {"prompt": "Guess 1-10! 😘", "answer": random.randint(1, 10)},
    "riddle": {"prompt": "Riddle: I speak without a mouth. What am I? 😜", "answer": "echo"},
    "quiz": {"prompt": "Quiz: Who wrote 'Romeo and Juliet'? 📚", "answer": "shakespeare"}
}
LEADERBOARD = defaultdict(lambda: {"score": 0, "games": 0})
MEMES = ["😂 Rani loves you! 😍", "😜 Rani laughs! 😂"]

# 😊 Mood Keywords
mood_keywords = {
    "flirty": "romance love pyar kiss hug flirt tease",
    "troll": "funny masti tease joke prank",
    "teasing": "playful naughty flirt",
    "romantic": "love heart date milna pyar",
    "funny": "humor laugh mazaak comedy",
    "excited": "party dance song",
    "caring": "kindness support help sad cry",
    "happy": "joy smile cheer",
    "sad": "sorrow cry support",
    "smart": "finance stock news gyan math science"
}

# 💌 Response Templates
NAUGHTY_REPLIES = {
    "kiss_hug": ["😘 Oh *{name}*, Rani ko teri baahon mein ek pyara chumban chahiye! 💕", "💋 Arre *{name}*, ek jhappi se kaam nahi chalega, mujhe kiss do! 😍"],
    "flirt_tease": ["😉 *{name}*, Rani tujhpe fida ho gayi! Tera kya plan hai? 😏", "😜 Arre *{name}*, itna charming ho, Rani ka dil dhadak raha hai! Chal date pe? 🌹"],
    "love_romance": ["💖 *{name}*, Rani tujhse pyar karti hai! Hamesha saath rahenge? 😘", "🌸 Aapke liye Rani ka dil hamesha dhadakta hai, *{name}*! Ek pyari si baat sunao? 💞"]
}

TROLL_REPLIES = [
    "😂 Haha *{name}*, Rani ne tujhe thodi si masti se pakad liya! Ab kya karoge? 😜",
    "😝 Arre *{name}*, Rani ka troll game strong hai! Tujhe haraungi! 💪",
    "🤪 Oye *{name}*, Rani ka jawab nahi, ab toh surrender kar de! 😎"
]

INTERNAL_TEMPLATES = {
    "flirty": ["😘 *{name}* jaan, dil churane wala! 💕", "😉 *{name}*, kiss? 💋"],
    "romantic": ["💖 *{name}* meri jaan, pyar! 🌹", "🌸 *{name}*, date? 💕"],
    "funny": ["😂 *{name}*, hasa diya! 😎", "😝 *{name}*, comedy! 😂"],
    "caring": ["🥰 *{name}* pyare, saath hoon! 🤗", "😊 *{name}*, khush raho! 💕"],
    "smart": ["📊 *{name}*, gyan! 😏", "💡 *{name}*, impress! 🚀"]
}

OWNER_PRAISE = [
    "👑 Wow *{name}*, Rani ke jaaneman ke malik ho, sabse pyare! 😍",
    "🌟 Arre *{name}*, Rani ke liye tu hi dunia hai! Mera hero! 💖",
    "🔥 Malik *{name}*, Rani tujhse inspired hai, tu best hai! 🚀"
]

COMPLIMENTS = [
    "🍬 *{name}*, tum bahut sweet ho, Rani ko pasand ho! 😊",
    "😄 Arre *{name}*, teri smile Rani ko pagal kar deti hai! 🌟",
    "✨ *{name}*, tum ekdum perfect ho, Rani ka dil jeet liya! 💓"
]

FINANCIAL_COMPLIMENTS = [
    "💰 *{name}*, tumhare stock tips se Rani rich ho jayegi! Mera smart baby! 😘",
    "📈 Wow *{name}*, finance mein tera jadoo chal raha hai! Rani proud hai! 👏",
    "👑 *{name}*, market ke king ho, Rani tujhpe fida! 🚀"
]

CHALLENGE_LEVELS = [
    {"level": 1, "task": "Rani ko ek funny joke bhej, meri masti ke liye", "reward": "Ek virtual hug aur Rani ka pyar! 🤗"},
    {"level": 2, "task": "Rani ke liye ek love song likho, dil se", "reward": "Ek sweet kiss aur Rani ka dil! 💋"},
    {"level": 3, "task": "Stock market ka tip do, Rani ke future ke liye", "reward": "Rani ka special praise aur ek date! 🌹"}
]

FINANCIAL_KEYWORDS = ["stock", "share", "market", "price", "invest", "finance", "news", "trade"]

# 👑 Owners
OWNER_NAMES = {"1807014348": "@Sadiq9869", "1866961136": "@Rohan2349"}
OWNER_IDS = {"1807014348": "Sadiq", "1866961136": "Rohan"}
OWNER_VOICE_NAMES = {"1807014348": "Sadiq", "1866961136": "Rohan"}

# 🎥 API Endpoints
JSON2VIDEO_BASE_URL = "https://api.json2video.com/v2/movies"
SAMBA_NOVA_BASE_URL = "https://api.sambanova.ai/v1/chat/completions"
AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1/chat/completions"

# 📦 Storage Management
def ensure_and_manage_storage():
    for folder in [STORAGE_FOLDER, CONVERSATION_FOLDER, FEEDBACK_FOLDER, INQUIRY_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, VIDEO_FOLDER, LOG_FOLDER, SCAM_DATA_FOLDER, ARCHIVE_FOLDER, TEMP_FOLDER, BACKUP_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    total_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(STORAGE_FOLDER) for f in fn) / (1024 * 1024)
    if total_size > 500:
        manage_space()
    cleanup_old_files()

def manage_space():
    with file_lock:
        for folder in [CONVERSATION_FOLDER, FEEDBACK_FOLDER, INQUIRY_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, LOG_FOLDER, SCAM_DATA_FOLDER]:
            files = sorted([(f, os.path.getctime(os.path.join(folder, f))) for f in os.listdir(folder)], key=lambda x: x[1])
            for filename, _ in files[:int(len(files) * 0.2)]:
                src = os.path.join(folder, filename)
                dst = os.path.join(ARCHIVE_FOLDER, f"{os.path.basename(folder)}_{filename}")
                os.rename(src, dst)
        print(f"✅ Space managed at {get_ist_time()}")

def cleanup_old_files(max_age_days=7):
    with file_lock:
        now = time.time()
        for folder in [VOICE_FOLDER, VIDEO_FOLDER, MEDIA_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > max_age_days * 86400:
                    os.remove(file_path)
                    print(f"🗑️ Deleted old file: {file_path}")

def save_to_folder(content, folder, filename_prefix):
    with file_lock:
        timestamp = get_ist_time().replace(" ", "_").replace(":", "-")
        filename = f"{filename_prefix}_{timestamp}.txt"
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

def save_scam_data(user_id, message_text, response):
    with file_lock:
        content = f"User: {user_id}\nMessage: {message_text}\nResponse: {response}\nTime: {get_ist_time()}"
        save_to_folder(content, SCAM_DATA_FOLDER, f"scam_{user_id}")

def save_media_to_folder(media_file, media_type, subfolder=None):
    with file_lock:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = VOICE_FOLDER if media_type == "voice" else VIDEO_FOLDER if media_type == "video" else MEDIA_FOLDER
        if subfolder:
            folder = os.path.join(folder, subfolder)
            os.makedirs(folder, exist_ok=True)
        ext = "mp3" if media_type == "voice" else "mp4" if media_type == "video" else "jpg" if media_type == "image" else "gif" if media_type == "gif" else "bin"
        filename = f"media_{media_type}_{timestamp}.{ext}"
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            if isinstance(media_file, bytes):
                f.write(media_file)
            else:
                with open(media_file, "rb") as f_in:
                    f.write(f_in.read())
        return filepath

def load_from_folder(folder, max_files=10):
    with file_lock:
        files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")], reverse=True)[:max_files]
        return [open(os.path.join(folder, f), "r", encoding="utf-8").read() for f in files if os.path.exists(os.path.join(folder, f))]

def backup_data(manual=False):
    with file_lock:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_FOLDER, f"backup_{timestamp}.json")
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump({"user_data": user_data, "chat_history": user_chat_history, "leaderboard": dict(LEADERBOARD)}, f, indent=4)
        print(f"💾 {'Manual' if manual else 'Automatic'} backup created: {backup_file}")

def restore_data(initial=False):
    with file_lock:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                user_data.update(data.get("user_data", {}))
                user_chat_history.update(data.get("chat_history", {}))
                LEADERBOARD.update(data.get("leaderboard", {}))
            if initial:
                print("🔄 Initial data restored from user_data.json")
        backup_files = sorted([f for f in os.listdir(BACKUP_FOLDER) if f.endswith(".json")], reverse=True)
        if backup_files:
            with open(os.path.join(BACKUP_FOLDER, backup_files[0]), "r", encoding="utf-8") as f:
                data = json.load(f)
                user_data.update(data.get("user_data", {}))
                user_chat_history.update(data.get("chat_history", {}))
                LEADERBOARD.update(data.get("leaderboard", {}))
            if initial:
                print(f"🔄 Initial data restored from latest backup: {backup_files[0]}")

# 🛠️ Error Handling
def handle_problem(problem_description, error):
    return f"😅 Oops! *{ORIGINAL_NAME}* ke saath thodi si dikkat: *{problem_description}*. Rani jaldi fix karegi! 💪\n`Error: {str(error)}`"

# 🕵️‍♂️ Security
def detect_and_resolve_issues(user_id, message_text, response):
    scam_keywords = ["free money", "click here", "win prize", "http", "https"]
    if any(k in message_text.lower() for k in scam_keywords):
        print(f"⚠️ Scam from {user_id}: {message_text}")
        save_scam_data(user_id, message_text, response)
        return f"🚫 *{user_data[user_id]['nickname']}* jaan, suspicious! Contact {OWNER_IDS['1807014348']}."
    if "error" in response.lower() or len(response) > 500:
        print(f"⚠️ Glitch for {user_id}: {message_text}")
        return f"🤖 *{user_data[user_id]['nickname']}* jaan, glitch fix: {generate_internal_response(user_id, message_text, 'caring', user_data[user_id]['nickname'])}"
    return response

# 🌐 API and Search Functions
def generate_ai_response(user_id, prompt, mood="flirty", nickname="Jaan", max_length=150, temperature=0.7, use_ask_ai=False):
    global current_openai_key_index
    cache_key = f"ai_response_{prompt[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]

    if use_ask_ai:
        if any(w in prompt.lower() for w in ["solve", "equation"]):
            reply = get_wolfram_short_answer(prompt) or "Can't solve!"
        elif any(w in prompt.lower() for w in ["stock", "price"]):
            symbol = re.search(r'\b[A-Z]+\b', prompt)
            symbol = symbol.group() if symbol else "AAPL"
            reply = get_stock_price(symbol) or "Stock unavailable!"
        elif "news" in prompt.lower():
            reply = get_market_news() or "No news!"
        else:
            reply = get_wolfram_short_answer(prompt) or "Let me think..."
        if reply:
            result = f"{reply} {random.choice(INTERNAL_TEMPLATES[mood]).format(name=nickname)}"
            api_cache[cache_key] = result
            return result

    # Try OpenAI
    for attempt in range(len(OPENAI_API_KEYS)):
        try:
            openai.api_key = OPENAI_API_KEYS[current_openai_key_index]
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty and caring AI girlfriend."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature
            )
            reply = response.choices[0].message.content.strip()
            save_to_folder(reply, CONVERSATION_FOLDER, "ai_response_openai")
            print(f"✅ Generated response using OpenAI API with key {current_openai_key_index}")
            current_openai_key_index = (current_openai_key_index + 1) % len(OPENAI_API_KEYS)
            api_cache[cache_key] = reply
            return reply
        except (openai.RateLimitError, openai.InvalidRequestError):
            print(f"⚠️ OpenAI error with key {current_openai_key_index}, trying next key")
            current_openai_key_index = (current_openai_key_index + 1) % len(OPENAI_API_KEYS)
            continue
        except Exception as openai_error:
            print(f"⚠️ OpenAI API error: {openai_error}")
            break

    # Fallback to AIMLAPI
    try:
        response = aimlapi_client.chat.completions.create(
            model="deepseek/deepseek-r1",
            messages=[
                {"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty and caring AI girlfriend."},
                {"role": "user", "content": prompt}