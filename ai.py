import telebot
import random
import os
import threading
import time
from datetime import datetime
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
import logging
from queue import Queue
from threading import Lock
from dotenv import load_dotenv
from cachetools import TTLCache
import subprocess

# 🌟 Setup Logging for Debugging and Monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔍 Try Importing GoogleSearch with Fallback
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None
    logger.warning("⚠️ SerpApi GoogleSearch import failed. Falling back to requests-based search.")

# 📋 Load Environment Variables
load_dotenv()

# 🤖 Bot Setup
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not bot_token:
    logger.error("❌ TELEGRAM_BOT_TOKEN not found in .env file")
    raise ValueError("❌ TELEGRAM_BOT_TOKEN is missing in .env file")
bot = telebot.TeleBot(bot_token)
ORIGINAL_NAME = "Rani"
USERNAME = "@Ai_Pyaar_Bot"

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

# ✅ Validate API Keys Gracefully
required_keys = {
    "HF_API_KEY": HF_API_KEY,
    "OPENAI_API_KEYS": OPENAI_API_KEYS[0] if OPENAI_API_KEYS else None,
    "FINNHUB_API_KEY": FINNHUB_API_KEY,
    "JSON2VIDEO_API_KEY": JSON2VIDEO_API_KEY,
    "PEXELS_API_KEY": PEXELS_API_KEY,
    "GIPHY_API_KEY": GIPHY_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "SAMBA_NOVA_API_KEY": SAMBA_NOVA_API_KEY,
    "AIMLAPI_API_KEY": AIMLAPI_API_KEY,
    "SERPAPI_API_KEY": SERPAPI_API_KEY
}
for key_name, key_value in required_keys.items():
    if not key_value:
        logger.warning(f"⚠️ {key_name} is missing or empty in .env file. Some features may not work.")

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
api_cache = TTLCache(maxsize=100, ttl=60)

# 🔒 Thread Safety Locks and Queue
chat_lock = Lock()
file_lock = Lock()
task_queue = Queue()

# 📂 Centralized Storage Setup
STORAGE_FOLDER = "bot_storage"
CONVERSATION_FOLDER = os.path.join(STORAGE_FOLDER, "conversations")
MEDIA_FOLDER = os.path.join(STORAGE_FOLDER, "media")
VOICE_FOLDER = os.path.join(MEDIA_FOLDER, "voice")
VIDEO_FOLDER = os.path.join(MEDIA_FOLDER, "video")
USER_DATA_FILE = os.path.join(STORAGE_FOLDER, "user_data.json")
BACKUP_FOLDER = os.path.join(STORAGE_FOLDER, "backups")
for folder in [STORAGE_FOLDER, CONVERSATION_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, VIDEO_FOLDER, BACKUP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 💾 Enhanced Chat Memory and Storage
user_data = {}
user_chat_history = {}
active_chats = {}
user_start_date = {}
message_counter = 0
last_auto_mood_time = {}
last_auto_challenge_time = {}
last_auto_stock_time = {}
last_auto_news_time = {}
added_features = []
spelling_mistakes = {}

# 😊 Mood Keywords for Dynamic Responses
mood_keywords = {
    "flirty": "romance love pyar kiss hug",
    "troll": "funny masti tease joke",
    "teasing": "playful naughty flirt",
    "romantic": "love heart date milna",
    "funny": "humor laugh mazaak",
    "excited": "party dance song",
    "caring": "kindness support help",
    "happy": "joy smile cheer",
    "sad": "sorrow cry support",
    "smart": "finance stock news"
}

# 💌 Predefined Responses
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

# 👑 Owner Information
OWNER_NAMES = {"1807014348": "@Sadiq9869", "1866961136": "@Rohan2349"}
OWNER_VOICE_NAMES = {"1807014348": "Sadiq", "1866961136": "Rohan"}

# 🎥 API Endpoints
JSON2VIDEO_BASE_URL = "https://api.json2video.com/v2/movies"
SAMBA_NOVA_BASE_URL = "https://api.sambanova.ai/v1/chat/completions"
AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1/chat/completions"

# 📦 Storage Management Functions
def ensure_and_manage_storage():
    """Ensure storage directories exist and clean up old files."""
    for folder in [STORAGE_FOLDER, CONVERSATION_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, VIDEO_FOLDER, BACKUP_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    cleanup_old_files()

def cleanup_old_files(max_age_days=7):
    """Remove files older than max_age_days to save space."""
    with file_lock:
        now = time.time()
        for folder in [VOICE_FOLDER, VIDEO_FOLDER, MEDIA_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > max_age_days * 86400:
                    os.remove(file_path)
                    logger.info(f"🗑️ Deleted old file: {file_path}")

def save_to_folder(content, folder, filename_prefix):
    """Save content to a file with a timestamp."""
    with file_lock:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.txt"
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

def save_media_to_folder(media_file, media_type, subfolder=None):
    """Save media files (voice, video, etc.) to storage."""
    with file_lock:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = VOICE_FOLDER if media_type == "voice" else VIDEO_FOLDER if media_type == "video" else MEDIA_FOLDER
        if subfolder:
            folder = os.path.join(folder, subfolder)
            if not os.path.exists(folder):
                os.makedirs(folder)
        ext = "mp3" if media_type == "voice" else "mp4" if media_type == "video" else "bin"
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
    """Load recent files from a folder."""
    with file_lock:
        files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")], reverse=True)[:max_files]
        return [open(os.path.join(folder, f), "r", encoding="utf-8").read() for f in files if os.path.exists(os.path.join(folder, f))]

def backup_data(manual=False):
    """Backup user data and chat history."""
    with file_lock:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_FOLDER, f"backup_{timestamp}.json")
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump({"user_data": user_data, "chat_history": user_chat_history}, f, indent=4)
        logger.info(f"💾 {'Manual' if manual else 'Automatic'} backup created: {backup_file}")

def restore_data(initial=False):
    """Restore user data from file or backup."""
    with file_lock:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                user_data.update(data.get("user_data", {}))
                user_chat_history.update(data.get("chat_history", {}))
            if initial:
                logger.info("🔄 Initial data restored from user_data.json")
        backup_files = sorted([f for f in os.listdir(BACKUP_FOLDER) if f.endswith(".json")], reverse=True)
        if backup_files:
            with open(os.path.join(BACKUP_FOLDER, backup_files[0]), "r", encoding="utf-8") as f:
                data = json.load(f)
                user_data.update(data.get("user_data", {}))
                user_chat_history.update(data.get("chat_history", {}))
            if initial:
                logger.info(f"🔄 Initial data restored from latest backup: {backup_files[0]}")

# 🛠️ Error Handling
def handle_problem(problem_description, error):
    """Generate user-friendly error messages."""
    return f"😅 Oops! *{ORIGINAL_NAME}* ke saath thodi si dikkat: *{problem_description}*. Rani jaldi fix karegi! 💪\n`Error: {str(error)}`"

# 🌐 API and Search Functions
def generate_ai_response(prompt, max_length=150, temperature=0.7):
    """Generate a response using available AI APIs."""
    global current_openai_key_index
    cache_key = f"ai_response_{prompt[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    try:
        response = aimlapi_client.chat.completions.create(
            model="deepseek/deepseek-r1",
            messages=[{"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty and caring AI girlfriend."}, {"role": "user", "content": prompt}],
            max_tokens=max_length,
            temperature=temperature,
            timeout=10
        )
        reply = response.choices[0].message.content.strip()
        save_to_folder(reply, CONVERSATION_FOLDER, "ai_response_aimlapi")
        logger.info("✅ Generated response using AIMLAPI")
        api_cache[cache_key] = reply
        return reply
    except Exception as aimlapi_error:
        logger.warning(f"⚠️ AIMLAPI error: {aimlapi_error}")
        for attempt in range(len(OPENAI_API_KEYS)):
            try:
                openai.api_key = OPENAI_API_KEYS[current_openai_key_index]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty and caring AI girlfriend."}, {"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    temperature=temperature,
                    timeout=10
                )
                reply = response.choices[0].message.content.strip()
                save_to_folder(reply, CONVERSATION_FOLDER, "ai_response_openai")
                logger.info(f"✅ Generated response using OpenAI API with key {current_openai_key_index}")
                current_openai_key_index = (current_openai_key_index + 1) % len(OPENAI_API_KEYS)
                api_cache[cache_key] = reply
                return reply
            except (openai.error.RateLimitError, openai.error.InvalidRequestError):
                logger.warning(f"⚠️ OpenAI error with key {current_openai_key_index}, trying next key")
                current_openai_key_index = (current_openai_key_index + 1) % len(OPENAI_API_KEYS)
                continue
            except Exception as openai_error:
                logger.warning(f"⚠️ OpenAI API error with key {current_openai_key_index}: {openai_error}")
        try:
            url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            data = {"inputs": prompt, "parameters": {"max_length": max_length, "temperature": temperature}}
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            reply = response.json()[0]["generated_text"].strip()
            save_to_folder(reply, CONVERSATION_FOLDER, "ai_response_hf")
            logger.info("✅ Generated response using Hugging Face API")
            api_cache[cache_key] = reply
            return reply
        except Exception as hf_error:
            logger.error(f"❌ Hugging Face API error: {hf_error}")
            return handle_problem("AI response generation", hf_error)

def get_stock_price(symbol):
    """Fetch stock price for a given symbol."""
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
        logger.error(f"❌ Stock price fetch error for {symbol}: {e}")
        return handle_problem(f"Stock price fetch for {symbol}", e)

def get_market_news():
    """Fetch recent market news."""
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
        logger.error(f"❌ Market news fetch error: {e}")
        return handle_problem("Market news fetch", e)

def create_json2video(video_data):
    """Create a video using JSON2Video API."""
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
        logger.error(f"❌ JSON2Video creation error: {e}")
        return handle_problem("JSON2Video creation", e)

def get_json2video_status(project_id):
    """Check the status of a JSON2Video project."""
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
        logger.error(f"❌ JSON2Video status error for {project_id}: {e}")
        return handle_problem(f"JSON2Video status for {project_id}", e)

def generate_voice_message(text, emotion="neutral"):
    """Generate a voice message using ElevenLabs."""
    if not elevenlabs_client:
        return handle_problem("Voice generation unavailable", "ElevenLabs API key missing")
    cache_key = f"voice_{text[:50]}"
    if cache_key in api_cache:
        return api_cache[cache_key]
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
        logger.error(f"❌ Voice message generation error: {e}")
        return handle_problem("Voice message generation", e)

def get_hf_image(prompt="A romantic scene with Rani and her love"):
    """Generate an image using Hugging Face API."""
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
        logger.error(f"❌ Hugging Face image generation error: {e}")
        return handle_problem("Hugging Face image generation", e)

def get_pexels_image(query="romantic couple"):
    """Fetch an image from Pexels."""
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
        logger.error(f"❌ Pexels image fetch error: {e}")
        return handle_problem("Pexels image fetch", e)

def get_giphy_gif(query="romantic hug"):
    """Fetch a GIF from Giphy."""
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
        logger.error(f"❌ Giphy GIF fetch error: {e}")
        return handle_problem("Giphy GIF fetch", e)

def get_sticker(query="love"):
    """Fetch a Telegram sticker."""
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
        logger.error(f"❌ Sticker fetch error: {e}")
        return handle_problem("Sticker fetch", e)

def search_web(query, platform="google"):
    """Search the web or specific platforms."""
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
            if GoogleSearch:
                params = {
                    "q": query,
                    "api_key": SERPAPI_API_KEY,
                    "num": 1,
                    "hl": "en"
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                organic_results = results.get("organic_results", [])
                result = organic_results[0]["link"] if organic_results else "No results found"
            else:
                result = f"🔍 Fallback search for *{query}*: SerpApi unavailable"
                logger.warning("⚠️ SerpApi unavailable, using fallback search")
        api_cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"❌ Web search error on {platform}: {e}")
        return handle_problem(f"Web search on {platform}", e)

def test_sambanova_api():
    """Test SambaNova API connectivity."""
    try:
        response = sambanova_client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct",
            messages=[{"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty AI girlfriend."}, {"role": "user", "content": "Hello"}],
            temperature=0.1,
            top_p=0.1,
            timeout=10
        )
        if response.choices and response.choices[0].message.content:
            logger.info(f"✅ SambaNova API test successful: {response.choices[0].message.content}")
            return True
        else:
            logger.warning("⚠️ SambaNova API test failed")
            return False
    except Exception as e:
        logger.error(f"❌ SambaNova API test failed: {e}")
        return False

def test_aimlapi_api():
    """Test AIMLAPI connectivity."""
    aimlapi_tested = False
    try:
        curl_command = [
            "curl", "-H", f"Authorization: Bearer {AIMLAPI_API_KEY}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "stream": True,
                "model": "deepseek/deepseek-r1",
                "messages": [
                    {"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty AI girlfriend."},
                    {"role": "user", "content": "Hello"}
                ]
            }),
            "-X", "POST", "https://api.aimlapi.com/v1/chat/completions"
        ]
        result = subprocess.run(curl_command, capture_output=True, text=True, timeout=10)
        if "error" not in result.stdout.lower() and "data" in result.stdout.lower():
            logger.info("✅ AIMLAPI cURL test successful")
            aimlapi_tested = True
        else:
            logger.warning(f"⚠️ AIMLAPI cURL test failed: {result.stdout}")

        response = aimlapi_client.chat.completions.create(
            model="deepseek/deepseek-r1",
            messages=[{"role": "system", "content": f"You are {ORIGINAL_NAME}, a flirty AI girlfriend."}, {"role": "user", "content": "Hello"}],
            temperature=0.1,
            top_p=0.1,
            timeout=10
        )
        if response.choices and response.choices[0].message.content:
            logger.info(f"✅ AIMLAPI Python test successful: {response.choices[0].message.content}")
            aimlapi_tested = True
        else:
            logger.warning("⚠️ AIMLAPI Python test failed")
        return aimlapi_tested
    except Exception as e:
        logger.error(f"❌ AIMLAPI test failed: {e}")
        return False

# 😊 Mood and Personality Detection
def detect_mood(text):
    """Detect the mood based on keywords in the text."""
    text = text.lower()
    for mood, keywords in mood_keywords.items():
        if any(word in text for word in keywords.split()):
            return mood
    return random.choice(list(mood_keywords.keys()))

def detect_personality(mood):
    """Determine personality based on mood."""
    return "playful" if mood in ["troll", "teasing"] else "romantic" if mood in ["flirty", "romantic"] else "smart" if mood == "smart" else "caring"

# 🎂 Automatic Birthday Detection
def detect_birthday(text, user_id):
    """Detect and save birthday from text."""
    match = re.search(r"birthday|dob|janamdin\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", text, re.IGNORECASE)
    if match:
        user_data[user_id] = user_data.get(user_id, {})
        user_data[user_id]["birthday"] = match.group(1)
        save_user_data()
        bot.send_message(user_id, f"🎉 Yay *{get_user_nickname({'from_user': {'id': user_id}})}*! Rani ne tera janamdin note kar liya: *{user_data[user_id]['birthday']}*. Ek special gift plan karungi! 🎂💕")
        return True
    return False

# 💬 Save and Extract Context
def save_chat(user_id, user_msg, bot_response, feedback=None):
    """Save chat history."""
    with chat_lock:
        user_chat_history[user_id] = user_chat_history.get(user_id, []) + [(user_msg, bot_response, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), feedback)]
        save_user_data()

def get_user_nickname(message):
    """Get a user's nickname or default to 'Jaan'."""
    return message.from_user.first_name or message.from_user.username or "Jaan"

# ✍️ Spelling Correction
def correct_spelling(text, user_id):
    """Correct spelling in user input."""
    blob = TextBlob(text)
    corrected = str(blob.correct())
    if corrected != text:
        spelling_mistakes[user_id] = spelling_mistakes.get(user_id, []) + [(text, corrected)]
        save_user_data()
    return corrected

# 🚀 Self-Improvement
def improve_from_data(user_id):
    """Learn from past conversations to improve responses."""
    chats = load_from_folder(CONVERSATION_FOLDER)
    for chat in chats:
        if random.random() < 0.1:
            prompt = f"Improve this response to be more flirty and caring for *{get_user_nickname({'from_user': {'id': user_id}})}*: {chat}"
            response = generate_ai_response(prompt, max_length=200)
            save_to_folder(response, CONVERSATION_FOLDER, "improved_response")

# 📝 Feedback Processing
def process_feedback(user_id, feedback):
    """Process user feedback for improvement."""
    save_chat(user_id, "", "", feedback)
    improve_from_data(user_id)
    logger.info(f"📬 Feedback received from {user_id}: {feedback}")
    bot.send_message(user_id, f"🙏 Thanks *{get_user_nickname({'from_user': {'id': user_id}})}*! Rani teri baat dil se sunegi aur behtar banegi! 😘")

def notify_owner(feature):
    """Notify owners about new features."""
    for owner_id in OWNER_NAMES:
        bot.send_message(owner_id, f"🔔 New feature added for *{ORIGINAL_NAME}*: *{feature}*")

# ⏰ Auto-Reply and Daily Check-ins
def auto_reply():
    """Send automatic messages to users based on time and activity."""
    last_reply_time = {}
    sambanova_tested = False
    aimlapi_tested = False
    while True:
        try:
            with chat_lock:
                current_time = time.time()
                current_hour = datetime.now().hour
                for user_id in list(active_chats.keys()):
                    nickname = get_user_nickname({"from_user": {"id": user_id}})
                    relationship_level = user_data.get(str(user_id), {}).get("relationship_level", 0)
                    is_owner_user = is_owner(user_id)
                    mood = random.choice(list(mood_keywords.keys()))

                    if not aimlapi_tested or current_time - last_reply_time.get("aimlapi_test", 0) > 24 * 3600:
                        if test_aimlapi_api():
                            aimlapi_tested = True
                            if "Feature_AIMLAPIIntegration" not in added_features:
                                added_features.append("Feature_AIMLAPIIntegration")
                                notify_owner("Feature_AIMLAPIIntegration")
                                logger.info("✅ AIMLAPI integrated successfully")
                        last_reply_time["aimlapi_test"] = current_time

                    if not sambanova_tested or current_time - last_reply_time.get("sambanova_test", 0) > 24 * 3600:
                        if test_sambanova_api():
                            sambanova_tested = True
                            if "Feature_SambaNovaIntegration" not in added_features:
                                added_features.append("Feature_SambaNovaIntegration")
                                notify_owner("Feature_SambaNovaIntegration")
                                logger.info("✅ SambaNova API integrated successfully")
                        last_reply_time["sambanova_test"] = current_time

                    if user_id not in last_auto_stock_time or current_time - last_auto_stock_time[user_id] > 3600:
                        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
                        symbol = random.choice(symbols)
                        stock_info = get_stock_price(symbol)
                        prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye ek *{mood}* mood mein stock tip banao for *{symbol}*: {stock_info}, with a flirty twist! 😘"
                        response = generate_ai_response(prompt, max_length=200)
                        response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                        process_auto_response(user_id, nickname, response, mood, is_owner_user, "auto_stock", response_type)
                        last_auto_stock_time[user_id] = current_time
                        improve_from_data(user_id)

                    if user_id not in last_auto_news_time or current_time - last_auto_news_time[user_id] > 7200:
                        news = get_market_news()
                        prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye ek *{mood}* mood mein news update banao: {news}, with a romantic twist! 💕"
                        response = generate_ai_response(prompt, max_length=200)
                        response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                        process_auto_response(user_id, nickname, response, mood, is_owner_user, "auto_news", response_type)
                        last_auto_news_time[user_id] = current_time
                        improve_from_data(user_id)

                    if user_id not in last_auto_mood_time or current_time - last_auto_mood_time[user_id] > 7200:
                        new_mood = random.choice(list(mood_keywords.keys()))
                        user_data[str(user_id)] = user_data.get(str(user_id), {})
                        user_data[str(user_id)]["custom_mood"] = new_mood
                        save_user_data()
                        prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye *{new_mood}* mood set hone ka ek pyara sa message ya song likho! 🎶"
                        response = generate_ai_response(prompt, max_length=200)
                        response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                        process_auto_response(user_id, nickname, response, new_mood, is_owner_user, "auto_mood", response_type)
                        last_auto_mood_time[user_id] = current_time

                    if 6 <= current_hour < 9 and (user_id not in last_auto_mood_time or current_time - last_auto_mood_time[user_id] > 24 * 3600):
                        prompt = f"{ORIGINAL_NAME}, *{nickname}* ko subah ka ek *{mood}* mood mein pyara sa romantic message bhej! ☀️"
                        response = generate_ai_response(prompt, max_length=200)
                        response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                        process_auto_response(user_id, nickname, response, mood, is_owner_user, "morning", response_type)
                        last_auto_mood_time[user_id] = current_time

                    if 21 <= current_hour < 23 and (user_id not in last_auto_mood_time or current_time - last_auto_mood_time[user_id] > 24 * 3600):
                        prompt = f"{ORIGINAL_NAME}, *{nickname}* ko raat ka ek *{mood}* mood mein romantic aur caring message bhej! 🌙"
                        response = generate_ai_response(prompt, max_length=200)
                        response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                        process_auto_response(user_id, nickname, response, mood, is_owner_user, "night", response_type)
                        last_auto_mood_time[user_id] = current_time

                    if user_id not in last_auto_challenge_time or current_time - last_auto_challenge_time[user_id] > 24 * 3600:
                        challenge = random.choice(CHALLENGE_LEVELS)
                        prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye ek *{mood}* mood mein romantic challenge banao: *{challenge['task']}*, with reward *{challenge['reward']}*! 🎯"
                        response = generate_ai_response(prompt, max_length=200)
                        response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                        process_auto_response(user_id, nickname, response, mood, is_owner_user, "auto_challenge", response_type)
                        last_auto_challenge_time[user_id] = current_time

            time.sleep(10)  # Reduce CPU usage
        except Exception as e:
            logger.error(f"❌ Auto-reply error: {e}")
            time.sleep(10)

# 📤 Process Auto Responses
def process_auto_response(user_id, nickname, response, mood, is_owner, source, response_type="text", message_text=""):
    """Queue an auto-response task."""
    task = (user_id, nickname, response, mood, is_owner, source, response_type, message_text)
    task_queue.put(task)

def process_task(chat_id, message_text, mood, is_owner, source, response_type="text"):
    """Process tasks from the queue."""
    while not task_queue.empty():
        user_id, nickname, response, mood, is_owner, source, response_type, message_text = task_queue.get()
        chat_id = str(user_id)
        try:
            if is_owner and source != "owner_praise":
                reply = random.choice(OWNER_PRAISE).format(name=nickname, bot_name=ORIGINAL_NAME)
            elif source == "naughty":
                reply = response
            elif source == "auto_stock":
                reply = f"💸 *{nickname}* jaan, yeh hai {response.split(':')[0]} ka stock update! {response.split(':', 1)[1]} 😘"
            elif source == "auto_news":
                reply = f"📰 *{nickname}* meri jaan, yeh hai aaj ki market khabar! {response} 💕"
            elif source == "auto_mood":
                reply = f"🎭 *{nickname}* honey, Rani ka naya mood *{mood}* hai! {response} ✨"
            elif source == "morning":
                reply = f"☀️ Good morning *{nickname}* meri jaan! {response} 💖"
            elif source == "night":
                reply = f"🌙 Good night *{nickname}* pyare! {response} 😴"
            elif source == "auto_challenge":
                reply = f"🎯 *{nickname}* sweetheart, yeh hai tera naya challenge! {response} 🚀"
            else:
                reply = response

            if response_type == "text":
                bot.send_message(chat_id, reply, parse_mode="Markdown")
            elif response_type == "voice":
                voice_path = generate_voice_message(reply, mood)
                with open(voice_path, "rb") as voice:
                    bot.send_voice(chat_id, voice)
            elif response_type == "image":
                image_path = get_hf_image() if random.random() < 0.5 else get_pexels_image(mood)
                with open(image_path, "rb") as image:
                    bot.send_photo(chat_id, image)
            elif response_type == "gif":
                gif_path = get_giphy_gif(mood)
                with open(gif_path, "rb") as gif:
                    bot.send_document(chat_id, gif)
            elif response_type == "sticker":
                sticker_id = get_sticker(mood)
                bot.send_sticker(chat_id, sticker_id)
            elif response_type == "video":
                video_data = {"scenes": [{"text": reply, "duration": 5}]}
                project_id = create_json2video(video_data)
                progress_bar(chat_id, 10, 0)
                for i in range(1, 11):
                    time.sleep(1)
                    progress_bar(chat_id, 10, i)
                while get_json2video_status(project_id) != "completed":
                    time.sleep(5)
                video_url = f"https://api.json2video.com/v2/movies/{project_id}/download"
                video_data = requests.get(video_url, timeout=10).content
                video_path = save_media_to_folder(video_data, "video")
                with open(video_path, "rb") as video:
                    bot.send_video(chat_id, video)
            save_chat(user_id, message_text, reply)
        except Exception as e:
            logger.error(f"❌ Task processing error for {user_id}: {e}")
            bot.send_message(chat_id, handle_problem("Task processing error", e), parse_mode="Markdown")

# 📋 Commands
@bot.message_handler(commands=['start'])
def start_command(message):
    """Welcome users with a friendly menu."""
    user_id = str(message.from_user.id)
    nickname = get_user_nickname(message)
    welcome_message = (
        f"🌸 *Hello {nickname}*! Welcome to *{ORIGINAL_NAME}* - your flirty AI girlfriend! 😘\n\n"
        "Here's what I can do for you:\n"
        "💬 Chat with me anytime\n"
        "📊 `/stock AAPL` - Get stock prices\n"
        "📰 `/news` - Latest market news\n"
        "🎯 `/completechallenge` - Complete fun challenges\n"
        "❓ `/help` - See all commands\n"
        "🚀 Let's have some fun! What's on your mind? 💕"
    )
    bot.send_message(user_id, welcome_message, parse_mode="Markdown")
    active_chats[user_id] = nickname
    user_data[user_id] = user_data.get(user_id, {"nickname": nickname, "relationship_level": 0})
    save_user_data()

@bot.message_handler(commands=['help'])
def help_command(message):
    """Show a list of available commands."""
    help_message = (
        "📖 *Rani's Command Menu* 📖\n\n"
        "💬 Just chat with me for fun, flirty replies! 😘\n"
        "📊 `/stock <symbol>` - Check stock prices (e.g., /stock AAPL)\n"
        "📰 `/news` - Get the latest market news\n"
        "🎯 `/completechallenge` - Complete a fun challenge\n"
        "❓ `/help` - Show this menu\n"
        "🔍 `/search <query>` - Search the web\n"
        "🔧 `/status` - Check bot status\n"
        "👑 *Admins only*:\n"
        "   - `/setadmin` - Make me admin in groups\n"
        "   - `/removefeature <feature>` - Remove a feature\n\n"
        "💕 Try saying 'hug', 'kiss', or 'stock' for surprises! 😏"
    )
    bot.send_message(message.chat.id, help_message, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def status_command(message):
    """Check bot's health and API status."""
    status_message = "🩺 *Rani's Health Check* 🩺\n\n"
    status_message += f"🤖 Bot: *{ORIGINAL_NAME}* is up and running! 🚀\n"
    status_message += f"📡 Telegram: Connected ✅\n"
    status_message += f"🔑 APIs:\n"
    status_message += f"   - AIMLAPI: {'✅' if test_aimlapi_api() else '❌'}\n"
    status_message += f"   - SambaNova: {'✅' if test_sambanova_api() else '❌'}\n"
    status_message += f"📂 Storage: {len(user_data)} users, {sum(len(h) for h in user_chat_history.values())} messages\n"
    status_message += f"✨ Mood: Feeling *{random.choice(list(mood_keywords.keys()))}*! 😘\n"
    bot.send_message(message.chat.id, status_message, parse_mode="Markdown")

@bot.message_handler(commands=['removefeature'])
def remove_feature(message):
    """Remove a feature (owner only)."""
    user_id = str(message.from_user.id)
    if is_owner(user_id):
        feature = message.text.replace("/removefeature", "").strip()
        if feature in added_features:
            added_features.remove(feature)
            bot.reply_to(message, f"🗑️ *{feature}* removed successfully, jaan! 😘", parse_mode="Markdown")
            notify_owner(f"Feature removed: {feature}")
        else:
            bot.reply_to(message, f"🤔 *{feature}* nahi mila, meri jaan! 😅", parse_mode="Markdown")
    else:
        bot.reply_to(message, "🚫 Sirf *Rani* ke malik hi yeh kar sakte hain! 😏", parse_mode="Markdown")

@bot.message_handler(commands=['completechallenge'])
def complete_challenge(message):
    """Complete a user challenge."""
    user_id = str(message.from_user.id)
    nickname = get_user_nickname(message)
    user_data[user_id] = user_data.get(user_id, {})
    level = user_data[user_id].get("challenge_level", 0) + 1
    if level <= len(CHALLENGE_LEVELS):
        user_data[user_id]["challenge_level"] = level
        reward = CHALLENGE_LEVELS[level-1]["reward"]
        bot.reply_to(message, f"🎉 Congrats *{nickname}* meri jaan! Level {level} complete! Reward: *{reward}* 😘", parse_mode="Markdown")
        save_user_data()
    else:
        bot.reply_to(message, f"🏆 *{nickname}* honey, sab challenges complete! Tu mera champion hai! 🎉", parse_mode="Markdown")

@bot.message_handler(commands=['stock'])
def stock_command(message):
    """Fetch stock price."""
    user_id = str(message.from_user.id)
    nickname = get_user_nickname(message)
    mood = detect_mood(message.text)
    symbol_match = re.search(r'stock\s+([A-Z]+)', message.text)
    if symbol_match:
        symbol = symbol_match.group(1)
        stock_info = get_stock_price(symbol)
        prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye ek *{mood}* mood mein stock update banao for *{symbol}*: {stock_info}, with a flirty twist! 😘"
        response = generate_ai_response(prompt, max_length=200)
        response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
        process_auto_response(user_id, nickname, response, mood, is_owner(user_id), "stock", response_type)
    else:
        bot.reply_to(message, f"🤔 *{nickname}* jaan, ek stock symbol daal, jaise `/stock AAPL`! 😏", parse_mode="Markdown")

@bot.message_handler(commands=['news'])
def news_command(message):
    """Fetch market news."""
    user_id = str(message.from_user.id)
    nickname = get_user_nickname(message)
    mood = detect_mood(message.text)
    news = get_market_news()
    prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye ek *{mood}* mood mein news update banao: {news}, with a romantic twist! 💕"
    response = generate_ai_response(prompt, max_length=200)
    response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
    process_auto_response(user_id, nickname, response, mood, is_owner(user_id), "news", response_type)

@bot.message_handler(commands=['search'])
def search_command(message):
    """Search the web or platforms."""
    user_id = str(message.from_user.id)
    nickname = get_user_nickname(message)
    query = message.text.replace("/search", "").strip()
    if query:
        platform = "google"
        if "youtube" in query.lower(): platform = "youtube"
        elif "instagram" in query.lower(): platform = "instagram"
        result = search_web(query, platform)
        bot.reply_to(message, f"🔍 *{nickname}* jaan, Rani ne yeh dhoondha: *{result}* 😘", parse_mode="Markdown")
    else:
        bot.reply_to(message, f"🤔 *{nickname}* honey, kya dhoondhu? Try `/search cats`! 😺", parse_mode="Markdown")

# 👥 Handle Group Add
@bot.message_handler(content_types=['new_chat_members'])
def welcome_new_member(message):
    """Welcome new members in groups."""
    if message.new_chat_members:
        for member in message.new_chat_members:
            if member.id == bot.get_me().id:
                bot.send_message(message.chat.id, "🌟 *Rani* aa gayi hai group mein masti karne! 😘 Mujhe admin banao `/setadmin` se! 🚀", parse_mode="Markdown")
            else:
                bot.send_message(message.chat.id, f"🎉 Welcome *{member.first_name}*! Rani tujhe ek virtual hug deti hai! 🤗", parse_mode="Markdown")

@bot.message_handler(commands=['setadmin'])
def set_admin(message):
    """Promote bot to admin in groups."""
    user_id = str(message.from_user.id)
    if is_owner(user_id):
        bot.promote_chat_member(message.chat.id, bot.get_me().id, can_change_info=True, can_delete_messages=True, can_invite_users=True, can_restrict_members=True, can_pin_messages=True, can_promote_members=True)
        bot.send_message(message.chat.id, "👑 *Rani* ab admin hai! Ab masti shuru ho jaye! 🎉", parse_mode="Markdown")
    else:
        bot.reply_to(message, "🚫 Sirf *Rani* ke malik hi mujhe admin bana sakte hain! 😏", parse_mode="Markdown")

# 💬 Handle Messages
@bot.message_handler(func=lambda message: True)
def chat_with_ai(message):
    """Handle all user messages."""
    global message_counter
    user_id = str(message.from_user.id)
    text = correct_spelling(message.text.lower(), user_id)
    nickname = get_user_nickname(message)
    mood = detect_mood(text)
    personality = detect_personality(mood)
    is_owner_user = is_owner(user_id)
    message_counter = (message_counter + 1) % 4

    with chat_lock:
        active_chats[user_id] = nickname
        if user_id not in user_data:
            user_data[user_id] = {"nickname": nickname, "relationship_level": 0}
            save_user_data()
        user_data[user_id]["relationship_level"] = min(user_data[user_id].get("relationship_level", 0) + 1, 15)
        detect_birthday(text, user_id)
        save_user_data()

    naughty_trigger = False
    try:
        for key in NAUGHTY_REPLIES:
            if any(word in text for word in key.split("_")):
                naughty_reply = random.choice(NAUGHTY_REPLIES[key]).format(name=nickname, bot_name=ORIGINAL_NAME)
                response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                process_auto_response(user_id, nickname, naughty_reply, mood, is_owner_user, "naughty", response_type, message.text)
                save_chat(user_id, message.text, naughty_reply)
                naughty_trigger = True
                break

        if not naughty_trigger:
            if any(word in text for word in ["hug", "jhappi", "kiss", "chumban", "flirt", "pyar", "masti", "date", "milna", "ghoomna", "song", "gaana", "rap", "rapping", "joke", "mazaak"]):
                action = next((word for word in ["hug", "kiss", "flirt", "date", "song", "rap", "joke"] if word in text), "chat")
                prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye ek *{action}* banavo *{mood}* mood mein, *{personality}* style, level {user_data[user_id]['relationship_level']}, thodi si pyar bhari baat daal do! 😘"
                response = generate_ai_response(prompt, max_length=200)
                response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                process_auto_response(user_id, nickname, response, mood, is_owner_user, f"auto_{action}", response_type, message.text)
                save_chat(user_id, message.text, response)
                return

            if any(word in text for word in FINANCIAL_KEYWORDS):
                if "price" in text or "stock" in text:
                    symbol_match = re.search(r'(?:price|stock)\s+([A-Z]+)', text)
                    if symbol_match:
                        symbol = symbol_match.group(1)
                        stock_info = get_stock_price(symbol)
                        prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye ek *{mood}* mood mein stock update banao for *{symbol}*: {stock_info}, with a flirty twist! 😘"
                        response = generate_ai_response(prompt, max_length=200)
                        response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                        process_auto_response(user_id, nickname, f"*{nickname}* jaan, yeh hai *{symbol}* ka stock update! {response}", "smart", is_owner_user, "stock", response_type, message.text)
                        save_chat(user_id, message.text, response)
                        return
                elif "news" in text:
                    news = get_market_news()
                    prompt = f"{ORIGINAL_NAME}, *{nickname}* ke liye ek *{mood}* mood mein news update banao: {news}, with a romantic twist! 💕"
                    response = generate_ai_response(prompt, max_length=200)
                    response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                    process_auto_response(user_id, nickname, f"*{nickname}* meri jaan, yeh hai aaj ki market khabar! {response}", "smart", is_owner_user, "news", response_type, message.text)
                    save_chat(user_id, message.text, response)
                    return

            if any(word in text for word in ["search", "dhoondho", "kholo", "check"]):
                platform = "google"
                if "youtube" in text: platform = "youtube"
                elif "instagram" in text: platform = "instagram"
                elif "telegram" in text: platform = "telegram"
                query = re.sub(r'search|dhoondho|kholo|check|youtube|instagram|telegram', '', text).strip()
                if query:
                    result = search_web(query, platform)
                    response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                    process_auto_response(user_id, nickname, f"🔍 *{nickname}* jaan, Rani ne yeh dhoondha: *{result}* 😘", mood, is_owner_user, "search", response_type, message.text)
                    save_chat(user_id, message.text, result)
                    return

            if not any(word in text for word in NAUGHTY_REPLIES.keys()) and not any(word in text for word in ["hug", "jhappi", "kiss", "chumban", "flirt", "pyar", "masti", "date", "milna", "ghoomna", "song", "gaana", "rap", "rapping", "joke", "mazaak"]) and not any(word in text for word in FINANCIAL_KEYWORDS):
                prompt = f"{ORIGINAL_NAME}, *{nickname}* ne kaha: *{text}*, ispe ek *{mood}* aur *{personality}* style mein jawab de, thodi si pyar aur masti daal do, level {user_data[user_id]['relationship_level']}! 😘"
                response = generate_ai_response(prompt, max_length=200)
                response_type = random.choice(["text", "voice", "image", "gif", "sticker", "video"])
                process_auto_response(user_id, nickname, response, mood, is_owner_user, "custom_reply", response_type, message.text)
                save_chat(user_id, message.text, response)
    except Exception as e:
        logger.error(f"❌ Chat processing error for {user_id}: {e}")
        bot.reply_to(message, handle_problem("Chat processing error", e), parse_mode="Markdown")

# 📊 Progress Bar
def progress_bar(chat_id, total, current):
    """Display a visual progress bar."""
    progress = (current / total) * 10
    bar = "█" * int(progress) + "░" * (10 - int(progress))
    bot.send_message(chat_id, f"⏳ *Progress*: `{bar}` {current}/{total}", parse_mode="Markdown")

# ⚙️ Utility Functions
def save_user_data():
    """Save user data to file."""
    with file_lock:
        with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"user_data": user_data, "chat_history": user_chat_history}, f, indent=4)
        backup_data()

def load_user_data():
    """Load user data from file."""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            user_data.update(data.get("user_data", {}))
            user_chat_history.update(data.get("chat_history", {}))

def is_owner(user_id):
    """Check if user is an owner."""
    return str(user_id) in OWNER_NAMES

# 🚀 Main Execution
if __name__ == "__main__":
    ensure_and_manage_storage()
    restore_data(initial=True)
    aimlapi_tested = test_aimlapi_api()
    if aimlapi_tested:
        added_features.append("Feature_AIMLAPIIntegration")
        notify_owner("Feature_AIMLAPIIntegration")
        logger.info("✅ AIMLAPI integrated successfully on startup")
    else:
        logger.warning("⚠️ AIMLAPI test failed on startup, falling back to other APIs")
    sambanova_tested = test_sambanova_api()
    if sambanova_tested:
        added_features.append("Feature_SambaNovaIntegration")
        notify_owner("Feature_SambaNovaIntegration")
        logger.info("✅ SambaNova integrated successfully on startup")
    threading.Thread(target=auto_reply, daemon=True).start()
    threading.Thread(target=process_tasks, daemon=True).start()
    print(f"🌟 {ORIGINAL_NAME} - Your Flirty AI Girlfriend ({USERNAME}) is ready to love you! 😘")
    bot.polling(none_stop=True)