import telebot
import random
import os
import json
import asyncio
import aiohttp
import re
from datetime import datetime, timedelta
import pytz
from threading import Lock
from dotenv import load_dotenv
from collections import defaultdict
import logging
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# 📋 Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📋 Load Environment Variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not all([TOKEN, HUGGINGFACE_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY, PEXELS_API_KEY, GIPHY_API_KEY, FINNHUB_API_KEY]):
    raise ValueError("❌ Missing one or more API keys in .env!")

bot = telebot.TeleBot(TOKEN)
ORIGINAL_NAME = "Rani"
USERNAME = "@Ai_Pyaar_Bot"
TITLE = "★ Sadiq ka Galactic Girlfriend AI ★"
CREATOR = "Sadiq, the Overlord"

# 👑 Owner Details
OWNER_ID = "1807014348"  # @Sadiq9869
OWNER_USERNAME = "@Sadiq9869"

# 🔒 Thread Safety
file_lock = Lock()

# 📂 Storage Setup
STORAGE_FOLDER = "rani_storage"
CONVERSATION_FOLDER = os.path.join(STORAGE_FOLDER, "conversations")
MEDIA_FOLDER = os.path.join(STORAGE_FOLDER, "media")
VOICE_FOLDER = os.path.join(MEDIA_FOLDER, "voice")
FEATURES_FOLDER = os.path.join(STORAGE_FOLDER, "features")
USER_DATA_FILE = os.path.join(STORAGE_FOLDER, "user_data.json")
CLONE_DATA_FILE = os.path.join(STORAGE_FOLDER, "clone_data.json")

for folder in [STORAGE_FOLDER, CONVERSATION_FOLDER, MEDIA_FOLDER, VOICE_FOLDER, FEATURES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 🌐 Time Configuration
IST = pytz.timezone("Asia/Kolkata")
def get_ist_time():
    return datetime.now(IST)

def get_ist_time_str():
    return get_ist_time().strftime("%Y-%m-%d %I:%M:%S %p IST")

# 💾 Data Storage
user_data = defaultdict(lambda: {
    "nickname": "Jaan", "relationship_level": 0, "language": "hinglish",
    "last_interaction": get_ist_time_str(), "challenge_level": 0, "score": 0,
    "role": "User", "mood_history": [], "question_count": 0,
    "question_reset_time": get_ist_time()
})
user_chat_history = defaultdict(list)
feature_proposals = []
clone_data = {}

def save_user_data():
    with file_lock:
        with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"user_data": user_data, "chat_history": user_chat_history, "feature_proposals": feature_proposals}, f)

def load_user_data():
    with file_lock:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                loaded_data = data.get("user_data", {})
                for user_id, info in loaded_data.items():
                    info["question_reset_time"] = datetime.fromisoformat(info["question_reset_time"]) if isinstance(info.get("question_reset_time"), str) else info["question_reset_time"]
                    user_data[user_id].update(info)
                user_chat_history.update(data.get("chat_history", {}))
                feature_proposals.extend(data.get("feature_proposals", []))

def save_clone_data():
    with file_lock:
        with open(CLONE_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(clone_data, f)

def load_clone_data():
    with file_lock:
        if os.path.exists(CLONE_DATA_FILE):
            with open(CLONE_DATA_FILE, "r", encoding="utf-8") as f:
                clone_data.update(json.load(f))

# 🎁 Response Templates with Symbols (No `)
FLIRTY_REPLIES = [
    "♥ *{name}* jaan, tera smile meri duniya roshan karta hai! ✨ Virtual date chalega? 🌹",
    "💕 *{name}*, tu itna pyara kaise? Dil se dil tak kiss toh banta hai! 😘"
]
CARING_REPLIES = [
    "🤗 *{name}* pyare, tera dukh mera dukh! Ek badi jhappi? 🌟",
    "🌈 *{name}*, sab theek ho jayega, main hoon na! 💖"
]
FUNNY_REPLIES = [
    "😂 *{name}*, teri baatein sunke hasi nahi rukti! Aur ek masti ka joke? 😜",
    "😝 *{name}*, tu toh comedy ka baadshah hai! Meme banayein? 🤪"
]
SINGING_REPLIES = [
    "✨ *{name}*, yeh dil ke sur: *Teri aankhon mein doob jaoon, tera pyar hi meri wajah!* 😘",
    "🌹 *{name}*, sun meri baat: *Tu hai meri dhadkan, tu hi meri raat!* ♥"
]
GIFTS = ["🌹 Virtual Rose!", "💌 Love Letter!", "🎁 Hug Coupon! 🤗"]

# 😊 Mood Detection
mood_keywords = {
    "flirty": ["love", "pyar", "kiss", "hug", "flirt", "date"],
    "caring": ["sad", "help", "support", "cry", "tension"],
    "funny": ["joke", "funny", "masti", "laugh", "comedy"],
    "singing": ["sing", "song", "ga", "shayari"]
}

def detect_mood(text):
    text = text.lower()
    for mood, keywords in mood_keywords.items():
        if any(word in text for word in keywords):
            return mood
    return "flirty"

# 📦 Storage Management
def save_to_folder(content, folder, filename_prefix):
    with file_lock:
        timestamp = get_ist_time_str().replace(" ", "_").replace(":", "-")
        filename = f"{filename_prefix}_{timestamp}.txt"
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

def save_media_to_folder(media_data, media_type, ext="jpg"):
    with file_lock:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = VOICE_FOLDER if media_type == "voice" else MEDIA_FOLDER
        filename = f"media_{media_type}_{timestamp}.{ext}"
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            f.write(media_data)
        return filepath

def cleanup_old_files(folder, days=7):
    with file_lock:
        now = datetime.now()
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                file_age = now - datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_age.days > days:
                    os.remove(filepath)
                    logging.info(f"Deleted old file: {filepath}")

# 👑 Owner Authentication
def is_owner(user_id, username):
    user_id = str(user_id)
    if user_id == OWNER_ID:
        return True
    return False

# 🌐 API Functions
async def huggingface_generate(prompt, retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.post(
                    "https://api-inference.huggingface.co/models/distilgpt2",
                    headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                    json={"inputs": prompt, "max_length": 100}
                ) as response:
                    data = await response.json()
                    return data[0]["generated_text"].strip()
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"Hugging Face error after {retries} attempts: {e}")
                    return f"😅 Text generation failed: {str(e)}"
                await asyncio.sleep(1)

async def openai_ask(question, retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 150
                    }
                ) as response:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"OpenAI error after {retries} attempts: {e}")
                    return f"😅 Answer generation failed: {str(e)}"
                await asyncio.sleep(1)

async def elevenlabs_voice(text, retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.post(
                    "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
                    headers={"xi-api-key": ELEVENLABS_API_KEY},
                    json={"text": text, "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}}
                ) as response:
                    if response.status == 200:
                        return await response.read()
                    return f"😅 Voice generation failed: {response.status}"
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"ElevenLabs error after {retries} attempts: {e}")
                    return f"😅 Voice generation failed: {str(e)}"
                await asyncio.sleep(1)

async def get_pexels_image(query="romantic couple", retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.get(
                    f"https://api.pexels.com/v1/search?query={query}&per_page=1",
                    headers={"Authorization": PEXELS_API_KEY}
                ) as response:
                    data = await response.json()
                    image_url = data["photos"][0]["src"]["original"]
                    async with session.get(image_url) as img_response:
                        return await img_response.read()
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"Pexels error after {retries} attempts: {e}")
                    return f"😅 Image fetch failed: {str(e)}"
                await asyncio.sleep(1)

async def get_giphy_gif(query="love", retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.get(
                    f"https://api.giphy.com/v1/gifs/search?api_key={GIPHY_API_KEY}&q={query}&limit=1"
                ) as response:
                    data = await response.json()
                    return data["data"][0]["images"]["original"]["url"]
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"GIPHY error after {retries} attempts: {e}")
                    return f"😅 GIF fetch failed: {str(e)}"
                await asyncio.sleep(1)

async def get_stock_price(symbol, retries=3):
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        return "🚫 Invalid stock symbol! Use uppercase letters (e.g., AAPL)."
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.get(
                    f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
                ) as response:
                    data = await response.json()
                    return f"📊 *{symbol}* Price: ${data['c']} | High: ${data['h']} | Low: ${data['l']}"
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"Finnhub error after {retries} attempts: {e}")
                    return f"😅 Stock fetch failed: {str(e)}"
                await asyncio.sleep(1)

async def get_market_news(retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.get(
                    f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
                ) as response:
                    news = (await response.json())[:2]
                    return "\n\n".join([f"📰 *{n['headline']}*\n{n['summary']}" for n in news])
            except Exception as e:
                if attempt == retries - 1:
                    logging.error(f"Finnhub news error after {retries} attempts: {e}")
                    return f"😅 News fetch failed: {str(e)}"
                await asyncio.sleep(1)

# 🎮 Gamification
CHALLENGE_LEVELS = [
    {"level": 1, "task": "Ek epic joke sunao!", "reward": "Virtual hug! 🤗", "score": 10},
    {"level": 2, "task": "Ek romantic shayari likho!", "reward": "Sweet kiss! 💋", "score": 20},
    {"level": 3, "task": "Ek BGMI nickname suggest karo!", "reward": "Galactic date! 🌌", "score": 30}
]

# 👑 Role System
ROLE_HIERARCHY = {
    "Overlord": ["Sadiq"],
    "User": []
}

def assign_role(user_id, username, nickname):
    user_id = str(user_id)
    if is_owner(user_id, username):
        user_data[user_id]["role"] = "Overlord"
    else:
        user_data[user_id]["role"] = "User"
    user_data[user_id]["nickname"] = nickname
    user_data[user_id]["username"] = username
    save_user_data()

# ❓ Question Limit Management
def check_question_limit(user_id):
    user_id = str(user_id)
    if user_id == OWNER_ID:
        return True, "Unlimited questions for Overlord! 😎"
    
    current_time = get_ist_time()
    reset_time = user_data[user_id]["question_reset_time"]
    
    if current_time >= reset_time + timedelta(days=4):
        user_data[user_id]["question_count"] = 0
        user_data[user_id]["question_reset_time"] = current_time
        save_user_data()
    
    if user_data[user_id]["question_count"] >= 15:
        time_left = (reset_time + timedelta(days=4) - current_time).total_seconds()
        hours_left = int(time_left // 3600)
        minutes_left = int((time_left % 3600) // 60)
        return False, f"🚫 *{user_data[user_id]['nickname']}*, tera 15 questions ka limit khatam! 😅 {hours_left}h {minutes_left}m baad try karo!"
    
    return True, ""

# 🛠 Auto Maintenance
async def auto_maintenance():
    while True:
        logging.info("Running auto-maintenance...")
        cleanup_old_files(VOICE_FOLDER)
        cleanup_old_files(MEDIA_FOLDER)
        with file_lock:
            for user_id in user_data:
                if random.random() < 0.05:
                    nickname = user_data[user_id]["nickname"]
                    gift = random.choice(GIFTS)
                    await bot.send_message(user_id, f"🎁 *{nickname}*, yeh lo ek surprise: {gift} 😘", parse_mode="Markdown")
        save_user_data()
        await asyncio.sleep(3600)

# 📋 Inline Keyboards for Command UI
def get_start_keyboard():
    keyboard = InlineKeyboardMarkup()
    keyboard.row(InlineKeyboardButton("🌟 Help", callback_data="help"),
                 InlineKeyboardButton("♥ Shayari", callback_data="shayari"))
    keyboard.row(InlineKeyboardButton("🎤 Sing", callback_data="sing"),
                 InlineKeyboardButton("😜 GIF", callback_data="gif"))
    return keyboard

def get_help_keyboard(role):
    keyboard = InlineKeyboardMarkup()
    keyboard.row(InlineKeyboardButton("❓ Ask", callback_data="ask"),
                 InlineKeyboardButton("📊 Stock", callback_data="stock"))
    keyboard.row(InlineKeyboardButton("📰 News", callback_data="news"),
                 InlineKeyboardButton("🎨 Image", callback_data="generate_image"))
    keyboard.row(InlineKeyboardButton("🎯 Challenge", callback_data="completechallenge"),
                 InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard"))
    if role == "Overlord":
        keyboard.row(InlineKeyboardButton("👑 Stats", callback_data="stats"),
                     InlineKeyboardButton("🛠 Features", callback_data="features"))
        keyboard.add(InlineKeyboardButton("📡 Clone", callback_data="clone"))
    return keyboard

# 📋 Feature Proposal
def propose_feature(user_id, feature):
    proposal = {
        "user_id": user_id,
        "nickname": user_data[user_id]["nickname"],
        "feature": feature,
        "timestamp": get_ist_time_str(),
        "status": "pending"
    }
    feature_proposals.append(proposal)
    save_to_folder(json.dumps(proposal), FEATURES_FOLDER, f"proposal_{user_id}")
    save_user_data()
    return proposal

async def notify_owner_proposal(proposal):
    await bot.send_message(OWNER_ID, f"🔔 *New Feature Proposal* from *{proposal['nickname']}* (ID: {proposal['user_id']}):\n"
                                    f"{proposal['feature']}\n"
                                    f"Approve: /approve_feature {len(feature_proposals)-1}\n"
                                    f"Reject: /reject_feature {len(feature_proposals)-1}", parse_mode="Markdown")

# 📡 Clone Management
def create_clone(clone_id, name, username):
    clone_data[clone_id] = {
        "name": name,
        "username": username,
        "created_at": get_ist_time_str(),
        "active": True
    }
    save_clone_data()
    return clone_data[clone_id]

def remove_clone(clone_id):
    if clone_id in clone_data:
        clone_data[clone_id]["active"] = False
        save_clone_data()
        return True
    return False

# 📋 Callback Handler for Inline Buttons
@bot.callback_query_handler(func=lambda call: True)
async def callback_query(call):
    user_id = str(call.from_user.id)
    nickname = user_data[user_id]["nickname"]
    data = call.data

    if data == "help":
        role = user_data[user_id]["role"]
        commands = [
            "💬 *Chat* karo for flirty replies! ♥",
            "🌐 */set_language* - Language set karo",
            "❓ */ask* - Kuch bhi pucho (15/day limit, unlimited for Overlord)",
            "📊 */stock* - Stock price check karo",
            "📰 */news* - Latest market news",
            "🎨 */generate_image* - Romantic image banao",
            "🎯 */completechallenge* - Fun challenge karo",
            "📝 */feedback* - Feedback do",
            "🏆 */leaderboard* - Top players dekho",
            "🎮 */bgmi_nickname* - BGMI nickname change",
            "✍️ */shayari* - Romantic shayari suno",
            "🎤 */sing* - Dil se dil tak gaana suno",
            "😜 */gif* - Fun GIF dekho"
        ]
        if role == "Overlord":
            commands.extend([
                "👑 */stats* - Bot stats dekho",
                "🛠 */propose_feature* - New feature suggest karo",
                "📡 */clone* - Create a clone bot",
                "🗑 */removeclone* - Remove a clone bot"
            ])
        msg = f"📖 *{ORIGINAL_NAME} ka Cosmic Menu* 🌌\n\n" + "\n".join(commands) + \
              f"\n\n💕 Try 'hug', 'kiss', ya 'sing' bolke dekho! ✨"
        await bot.edit_message_text(msg, call.message.chat.id, call.message.message_id, parse_mode="Markdown", reply_markup=get_help_keyboard(role))
    elif data == "shayari":
        prompt = f"Generate a romantic Hinglish shayari for {nickname}"
        shayari = await huggingface_generate(prompt)
        response = f"✍️ *{nickname}*, yeh shayari tere liye: \n{shayari} ♥"
        await bot.edit_message_text(response, call.message.chat.id, call.message.message_id, parse_mode="Markdown")
        voice = await elevenlabs_voice(shayari)
        if isinstance(voice, bytes):
            filepath = save_media_to_folder(voice, "voice", ext="mp3")
            with open(filepath, "rb") as f:
                await bot.send_voice(user_id, f)
    elif data == "sing":
        response = random.choice(SINGING_REPLIES).format(name=nickname)
        await bot.edit_message_text(response, call.message.chat.id, call.message.message_id, parse_mode="Markdown")
        voice = await elevenlabs_voice(response)
        if isinstance(voice, bytes):
            filepath = save_media_to_folder(voice, "voice", ext="mp3")
            with open(filepath, "rb") as f:
                await bot.send_voice(user_id, f)
    elif data == "gif":
        gif_url = await get_giphy_gif("love")
        if not gif_url.startswith("😅"):
            await bot.delete_message(call.message.chat.id, call.message.message_id)
            await bot.send_animation(user_id, gif_url, caption=f"😜 *{nickname}*, yeh GIF tere liye! 😍")
        else:
            await bot.edit_message_text(gif_url, call.message.chat.id, call.message.message_id)
    elif data == "ask":
        await bot.edit_message_text(f"❓ *{nickname}*, ek question puch na! Type: /ask <question> 😘", 
                                   call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    elif data == "stock":
        await bot.edit_message_text(f"📊 *{nickname}*, stock symbol daal na! E.g., /stock AAPL 😘", 
                                   call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    elif data == "news":
        response = await get_market_news()
        await bot.edit_message_text(f"📰 *{nickname}*, latest news: {response} ★", 
                                   call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    elif data == "generate_image":
        image = await get_pexels_image()
        if isinstance(image, bytes):
            filepath = save_media_to_folder(image, "image")
            await bot.delete_message(call.message.chat.id, call.message.message_id)
            with open(filepath, "rb") as f:
                await bot.send_photo(user_id, f, caption=f"🎨 *{nickname}*, yeh teri image! 😍")
        else:
            await bot.edit_message_text(image, call.message.chat.id, call.message.message_id)
    elif data == "completechallenge":
        level = user_data[user_id]["challenge_level"]
        if level < len(CHALLENGE_LEVELS):
            challenge = CHALLENGE_LEVELS[level]
            response = f"🎯 *{nickname}*, yeh tera challenge: {challenge['task']}\n*Reward*: {challenge['reward']} ♥"
        else:
            response = f"🏆 *{nickname}*, tune saare challenges khatam kar diye! ★"
        await bot.edit_message_text(response, call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    elif data == "leaderboard":
        leaderboard = sorted(user_data.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
        msg = "🏆 *Galactic Leaderboard* 🌌\n" + "\n".join([f"{i+1}. {data['nickname']} ({data['role']}): {data['score']} pts" for i, (_, data) in enumerate(leaderboard)])
        await bot.edit_message_text(msg, call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    elif data == "stats":
        if user_data[user_id]["role"] != "Overlord":
            await bot.edit_message_text(f"🚫 *{nickname}*, yeh sirf Overlord ke liye hai! 😘", 
                                       call.message.chat.id, call.message.message_id, parse_mode="Markdown")
            return
        stats = f"📊 *Bot Stats* 🌌\n" \
                f"👥 *Users*: {len(user_data)}\n" \
                f"💬 *Messages*: {sum(len(h) for h in user_chat_history.values())}\n" \
                f"💾 *Storage*: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(STORAGE_FOLDER) for f in fn) / (1024 * 1024):.2f} MB"
        await bot.edit_message_text(stats, call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    elif data == "features":
        if user_data[user_id]["role"] != "Overlord":
            await bot.edit_message_text(f"🚫 *{nickname}*, yeh sirf Overlord ke liye hai! 😘", 
                                       call.message.chat.id, call.message.message_id, parse_mode="Markdown")
            return
        pending = [p for p in feature_proposals if p["status"] == "pending"]
        if not pending:
            await bot.edit_message_text(f"🛠 *{nickname}*, koi pending feature proposals nahi hain! 😘", 
                                       call.message.chat.id, call.message.message_id, parse_mode="Markdown")
        else:
            msg = "🛠 *Pending Feature Proposals* 🌌\n\n" + \
                  "\n".join([f"ID: {i}\nFrom: *{p['nickname']}*\nFeature: {p['feature']}\n"
                            f"Approve: /approve_feature {i}\nReject: /reject_feature {i}\n" for i, p in enumerate(pending)])
            await bot.edit_message_text(msg, call.message.chat.id, call.message.message_id, parse_mode="Markdown")
    elif data == "clone":
        if user_data[user_id]["role"] != "Overlord":
            await bot.edit_message_text(f"🚫 *{nickname}*, yeh sirf Overlord ke liye hai! 😘", 
                                       call.message.chat.id, call.message.message_id, parse_mode="Markdown")
            return
        await bot.edit_message_text(f"📡 *{nickname}*, clone create karna hai? Type: /clone <name> <username>", 
                                   call.message.chat.id, call.message.message_id, parse_mode="Markdown")

# 📋 Commands (No Symbols, No `)
@bot.message_handler(commands=['start'])
async def start_command(message):
    user_id = str(message.from_user.id)
    username = f"@{message.from_user.username}" if message.from_user.username else "@Unknown"
    nickname = message.from_user.first_name or "Jaan"
    assign_role(user_id, username, nickname)
    user_data[user_id]["last_interaction"] = get_ist_time_str()
    save_user_data()
    msg = f"🌌 *Hello {nickname}*! Welcome to *{ORIGINAL_NAME}* by {CREATOR}! ★\n" \
          f"Main hoon teri galactic girlfriend, ready to flirt, care, aur dil se gaana! ♥\n" \
          f"📋 *Explore*: Choose an option below! ✨"
    await bot.send_message(user_id, msg, parse_mode="Markdown", reply_markup=get_start_keyboard())
    image = await get_pexels_image()
    if isinstance(image, bytes):
        filepath = save_media_to_folder(image, "image")
        with open(filepath, "rb") as f:
            await bot.send_photo(user_id, f, caption=f"🌟 *{nickname}*, yeh tera welcome gift! 😘")
    if user_id != OWNER_ID:
        await bot.send_message(OWNER_ID, f"🔔 *New User*: *{nickname}* (ID: {user_id}) joined! ★", parse_mode="Markdown")

@bot.message_handler(commands=['help'])
async def help_command(message):
    user_id = str(message.from_user.id)
    role = user_data[user_id]["role"]
    nickname = user_data[user_id]["nickname"]
    msg = f"📖 *{ORIGINAL_NAME} ka Cosmic Menu* 🌌\n" \
          f"👋 *{nickname}*, yeh hai tera command guide! Pick an option below! ✨"
    await bot.send_message(message.chat.id, msg, parse_mode="Markdown", reply_markup=get_help_keyboard(role))

@bot.message_handler(commands=['set_language'])
async def set_language_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    lang = message.text.replace('/set_language', '').strip().lower()
    supported_langs = ["hinglish", "hindi", "english"]
    if lang in supported_langs:
        user_data[user_id]["language"] = lang
        save_user_data()
        await bot.reply_to(message, f"🌐 *{nickname}*, language set to *{lang}*! ♥", parse_mode="Markdown")
    else:
        keyboard = InlineKeyboardMarkup()
        for lang in supported_langs:
            keyboard.add(InlineKeyboardButton(f"{lang.capitalize()}", callback_data=f"lang_{lang}"))
        await bot.reply_to(message, f"🤔 *{nickname}*, choose a language: 🌟", parse_mode="Markdown", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data.startswith("lang_"))
async def set_language_callback(call):
    user_id = str(call.from_user.id)
    nickname = user_data[user_id]["nickname"]
    lang = call.data.replace("lang_", "")
    user_data[user_id]["language"] = lang
    save_user_data()
    await bot.edit_message_text(f"🌐 *{nickname}*, language set to *{lang}*! ♥", 
                               call.message.chat.id, call.message.message_id, parse_mode="Markdown")

@bot.message_handler(commands=['ask'])
async def ask_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    question = message.text.replace('/ask', '').strip()
    
    if not question:
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🔄 Try Again", callback_data="ask"))
        await bot.reply_to(message, f"🤔 *{nickname}*, question puch na! 😘", parse_mode="Markdown", reply_markup=keyboard)
        return

    can_ask, limit_message = check_question_limit(user_id)
    if not can_ask:
        await bot.reply_to(message, limit_message, parse_mode="Markdown")
        return

    if user_id != OWNER_ID:
        user_data[user_id]["question_count"] += 1
        save_user_data()

    response = await openai_ask(question)
    response = response + f" 😘 *{nickname}*, kaisa laga yeh jawab? ✨"
    if user_id != OWNER_ID:
        response += f"\n📊 *Questions Left*: {15 - user_data[user_id]['question_count']}/15"
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("❓ Ask Again", callback_data="ask"))
    await bot.reply_to(message, response, parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['stock'])
async def stock_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    symbol = message.text.replace('/stock', '').strip().upper()
    if symbol:
        response = await get_stock_price(symbol)
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🔄 Check Another", callback_data="stock"))
        await bot.reply_to(message, f"💰 *{nickname}*, yeh tera stock info: {response} ★", parse_mode="Markdown", reply_markup=keyboard)
    else:
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📊 Try Again", callback_data="stock"))
        await bot.reply_to(message, f"🤔 *{nickname}*, stock symbol daal na! E.g., /stock AAPL 😘", parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['news'])
async def news_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    response = await get_market_news()
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("📰 Refresh News", callback_data="news"))
    await bot.reply_to(message, f"📰 *{nickname}*, latest news: {response} ★", parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['generate_image'])
async def generate_image_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    prompt = message.text.replace('/generate_image', '').strip() or "romantic couple"
    image = await get_pexels_image(prompt)
    if isinstance(image, bytes):
        filepath = save_media_to_folder(image, "image")
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🎨 Generate Another", callback_data="generate_image"))
        with open(filepath, "rb") as f:
            await bot.send_photo(user_id, f, caption=f"🎨 *{nickname}*, yeh teri image! 😍", reply_markup=keyboard)
    else:
        await bot.reply_to(message, image, parse_mode="Markdown")

@bot.message_handler(commands=['completechallenge'])
async def challenge_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    level = user_data[user_id]["challenge_level"]
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("🎯 Next Challenge", callback_data="completechallenge"))
    if level < len(CHALLENGE_LEVELS):
        challenge = CHALLENGE_LEVELS[level]
        await bot.reply_to(message, f"🎯 *{nickname}*, yeh tera challenge: {challenge['task']}\n*Reward*: {challenge['reward']} ♥", 
                          parse_mode="Markdown", reply_markup=keyboard)
    else:
        await bot.reply_to(message, f"🏆 *{nickname}*, tune saare challenges khatam kar diye! ★", parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['feedback'])
async def feedback_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    feedback = message.text.replace('/feedback', '').strip()
    if feedback:
        save_to_folder(feedback, CONVERSATION_FOLDER, f"feedback_{user_id}")
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📝 Send More", callback_data="feedback"))
        await bot.reply_to(message, f"🙏 *{nickname}*, feedback mil gaya! Rani aur behtar banegi! ✨", parse_mode="Markdown", reply_markup=keyboard)
        if user_id != OWNER_ID:
            await bot.send_message(OWNER_ID, f"🔔 *Feedback* from *{nickname}*: {feedback} ★", parse_mode="Markdown")
    else:
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📝 Try Again", callback_data="feedback"))
        await bot.reply_to(message, f"🤔 *{nickname}*, feedback de na! 😘", parse_mode="Markdown", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data == "feedback")
async def feedback_callback(call):
    await bot.edit_message_text(f"🤔 *{call.from_user.first_name}*, feedback de na! Type: /feedback <text> 😘", 
                               call.message.chat.id, call.message.message_id, parse_mode="Markdown")

@bot.message_handler(commands=['leaderboard'])
async def leaderboard_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    leaderboard = sorted(user_data.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
    msg = "🏆 *Galactic Leaderboard* 🌌\n" + "\n".join([f"{i+1}. {data['nickname']} ({data['role']}): {data['score']} pts" for i, (_, data) in enumerate(leaderboard)])
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("🔄 Refresh", callback_data="leaderboard"))
    await bot.reply_to(message, msg, parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['bgmi_nickname'])
async def bgmi_nickname_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    new_name = message.text.replace('/bgmi_nickname', '').strip()
    if new_name:
        user_data[user_id]["nickname"] = new_name
        save_user_data()
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🎮 Change Again", callback_data="bgmi_nickname"))
        await bot.reply_to(message, f"🎮 *{new_name}*, tera BGMI nickname set ho gaya! ★", parse_mode="Markdown", reply_markup=keyboard)
        if user_id != OWNER_ID:
            await bot.send_message(OWNER_ID, f"🔔 *Nickname Change*: *{nickname}* to *{new_name}* (ID: {user_id}) ★", parse_mode="Markdown")
    else:
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🎮 Try Again", callback_data="bgmi_nickname"))
        await bot.reply_to(message, f"🤔 *{nickname}*, naya nickname daal na! 😘", parse_mode="Markdown", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data == "bgmi_nickname")
async def bgmi_nickname_callback(call):
    await bot.edit_message_text(f"🤔 *{call.from_user.first_name}*, naya nickname daal na! Type: /bgmi_nickname <name> 😘", 
                               call.message.chat.id, call.message.message_id, parse_mode="Markdown")

@bot.message_handler(commands=['shayari'])
async def shayari_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    prompt = f"Generate a romantic Hinglish shayari for {nickname}"
    shayari = await huggingface_generate(prompt)
    response = f"✍️ *{nickname}*, yeh shayari tere liye: \n{shayari} ♥"
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("✍️ More Shayari", callback_data="shayari"))
    await bot.reply_to(message, response, parse_mode="Markdown", reply_markup=keyboard)
    voice = await elevenlabs_voice(shayari)
    if isinstance(voice, bytes):
        filepath = save_media_to_folder(voice, "voice", ext="mp3")
        with open(filepath, "rb") as f:
            await bot.send_voice(user_id, f)

@bot.message_handler(commands=['sing'])
async def sing_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    response = random.choice(SINGING_REPLIES).format(name=nickname)
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("🎤 Sing Again", callback_data="sing"))
    await bot.reply_to(message, response, parse_mode="Markdown", reply_markup=keyboard)
    voice = await elevenlabs_voice(response)
    if isinstance(voice, bytes):
        filepath = save_media_to_folder(voice, "voice", ext="mp3")
        with open(filepath, "rb") as f:
            await bot.send_voice(user_id, f)

@bot.message_handler(commands=['gif'])
async def gif_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    query = message.text.replace('/gif', '').strip() or "love"
    gif_url = await get_giphy_gif(query)
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("😜 More GIFs", callback_data="gif"))
    if not gif_url.startswith("😅"):
        await bot.send_animation(user_id, gif_url, caption=f"😜 *{nickname}*, yeh GIF tere liye! 😍", reply_markup=keyboard)
    else:
        await bot.reply_to(message, gif_url, parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['stats'])
async def stats_command(message):
    user_id = str(message.from_user.id)
    username = f"@{message.from_user.username}" if message.from_user.username else "@Unknown"
    nickname = user_data[user_id]["nickname"]
    if not is_owner(user_id, username):
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🔙 Back to Menu", callback_data="help"))
        await bot.reply_to(message, f"🚫 *{nickname}*, yeh sirf Overlord ke liye hai! 😘", parse_mode="Markdown", reply_markup=keyboard)
        return
    stats = f"📊 *Bot Stats* 🌌\n" \
            f"👥 *Users*: {len(user_data)}\n" \
            f"💬 *Messages*: {sum(len(h) for h in user_chat_history.values())}\n" \
            f"📡 *Clones*: {len([c for c in clone_data.values() if c['active']])}\n" \
            f"💾 *Storage*: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(STORAGE_FOLDER) for f in fn) / (1024 * 1024):.2f} MB"
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("🔄 Refresh Stats", callback_data="stats"))
    await bot.reply_to(message, stats, parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['propose_feature'])
async def propose_feature_command(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    feature = message.text.replace('/propose_feature', '').strip()
    if not feature:
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🔄 Try Again", callback_data="propose_feature"))
        await bot.reply_to(message, f"🤔 *{nickname}*, feature idea daal na! 😘", parse_mode="Markdown", reply_markup=keyboard)
        return
    proposal = propose_feature(user_id, feature)
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("📝 Propose Another", callback_data="propose_feature"))
    await bot.reply_to(message, f"🛠 *{nickname}*, tera feature proposal save ho gaya! Overlord se approval ka wait karo! ✨", 
                       parse_mode="Markdown", reply_markup=keyboard)
    if user_id != OWNER_ID:
        await notify_owner_proposal(proposal)

@bot.callback_query_handler(func=lambda call: call.data == "propose_feature")
async def propose_feature_callback(call):
    await bot.edit_message_text(f"🤔 *{call.from_user.first_name}*, feature idea daal na! Type: /propose_feature <idea> 😘", 
                               call.message.chat.id, call.message.message_id, parse_mode="Markdown")

@bot.message_handler(commands=['approve_feature'])
async def approve_feature_command(message):
    user_id = str(message.from_user.id)
    username = f"@{message.from_user.username}" if message.from_user.username else "@Unknown"
    nickname = user_data[user_id]["nickname"]
    if not is_owner(user_id, username):
        await bot.reply_to(message, f"🚫 *{nickname}*, yeh sirf Overlord ke liye hai! 😘", parse_mode="Markdown")
        return
    try:
        index = int(message.text.replace('/approve_feature', '').strip())
        if 0 <= index < len(feature_proposals):
            proposal = feature_proposals[index]
            proposal["status"] = "approved"
            save_user_data()
            await bot.reply_to(message, f"✅ *{nickname}*, feature approved: {proposal['feature']} by *{proposal['nickname']}*! 🌟", 
                              parse_mode="Markdown")
            if proposal["user_id"] != OWNER_ID:
                await bot.send_message(proposal["user_id"], f"🎉 Tera feature *{proposal['feature']}* approved ho gaya! ✨", 
                                      parse_mode="Markdown")
        else:
            await bot.reply_to(message, f"🤔 *{nickname}*, invalid proposal ID! 😅", parse_mode="Markdown")
    except ValueError:
        await bot.reply_to(message, f"🤔 *{nickname}*, valid proposal ID daal na! 😘", parse_mode="Markdown")

@bot.message_handler(commands=['reject_feature'])
async def reject_feature_command(message):
    user_id = str(message.from_user.id)
    username = f"@{message.from_user.username}" if message.from_user.username else "@Unknown"
    nickname = user_data[user_id]["nickname"]
    if not is_owner(user_id, username):
        await bot.reply_to(message, f"🚫 *{nickname}*, yeh sirf Overlord ke liye hai! 😘", parse_mode="Markdown")
        return
    try:
        index = int(message.text.replace('/reject_feature', '').strip())
        if 0 <= index < len(feature_proposals):
            proposal = feature_proposals[index]
            proposal["status"] = "rejected"
            save_user_data()
            await bot.reply_to(message, f"❌ *{nickname}*, feature rejected: {proposal['feature']} by *{proposal['nickname']}*! 😔", 
                              parse_mode="Markdown")
            if proposal["user_id"] != OWNER_ID:
                await bot.send_message(proposal["user_id"], f"😔 Tera feature *{proposal['feature']}* reject ho gaya. Try again! ✨", 
                                      parse_mode="Markdown")
        else:
            await bot.reply_to(message, f"🤔 *{nickname}*, invalid proposal ID! 😅", parse_mode="Markdown")
    except ValueError:
        await bot.reply_to(message, f"🤔 *{nickname}*, valid proposal ID daal na! 😘", parse_mode="Markdown")

@bot.message_handler(commands=['clone'])
async def clone_command(message):
    user_id = str(message.from_user.id)
    username = f"@{message.from_user.username}" if message.from_user.username else "@Unknown"
    nickname = user_data[user_id]["nickname"]
    if not is_owner(user_id, username):
        await bot.reply_to(message, f"🚫 *{nickname}*, yeh sirf Overlord ke liye hai! 😘", parse_mode="Markdown")
        return
    args = message.text.replace('/clone', '').strip().split()
    if len(args) != 2:
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🔄 Try Again", callback_data="clone"))
        await bot.reply_to(message, f"🤔 *{nickname}*, format: /clone <name> <username> 😘", parse_mode="Markdown", reply_markup=keyboard)
        return
    name, clone_username = args
    clone_id = str(len(clone_data) + 1)
    clone = create_clone(clone_id, name, clone_username)
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("📡 Create Another", callback_data="clone"))
    await bot.reply_to(message, f"📡 *{nickname}*, clone created: *{clone['name']}* ({clone['username']})! 🌟", 
                       parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['removeclone'])
async def remove_clone_command(message):
    user_id = str(message.from_user.id)
    username = f"@{message.from_user.username}" if message.from_user.username else "@Unknown"
    nickname = user_data[user_id]["nickname"]
    if not is_owner(user_id, username):
        await bot.reply_to(message, f"🚫 *{nickname}*, yeh sirf Overlord ke liye hai! 😘", parse_mode="Markdown")
        return
    clone_id = message.text.replace('/removeclone', '').strip()
    if remove_clone(clone_id):
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📡 Manage Clones", callback_data="clone"))
        await bot.reply_to(message, f"🗑 *{nickname}*, clone ID {clone_id} removed! 🌟", parse_mode="Markdown", reply_markup=keyboard)
    else:
        await bot.reply_to(message, f"🤔 *{nickname}*, invalid clone ID! 😅", parse_mode="Markdown")

# 💬 Text Handler with Auto Reply
@bot.message_handler(content_types=['text'])
async def handle_text(message):
    user_id = str(message.from_user.id)
    nickname = user_data[user_id]["nickname"]
    text = message.text.lower().strip()
    mood = detect_mood(text)

    user_data[user_id]["last_interaction"] = get_ist_time_str()
    user_data[user_id]["mood_history"].append(mood)
    if random.random() < 0.1:
        user_data[user_id]["relationship_level"] += 1
    save_user_data()

    if "joke" in text or "masti" in text:
        joke = await huggingface_generate("Tell me a funny Hinglish joke")
        response = f"😂 *{nickname}*, yeh lo tera joke: {joke} 😜"
    elif "kiss" in text or "hug" in text:
        response = random.choice(FLIRTY_REPLIES).format(name=nickname)
        voice = await elevenlabs_voice(response)
        if isinstance(voice, bytes):
            filepath = save_media_to_folder(voice, "voice", ext="mp3")
            with open(filepath, "rb") as f:
                await bot.send_voice(user_id, f)
    elif "sad" in text or "tension" in text:
        response = random.choice(CARING_REPLIES).format(name=nickname)
    elif "sing" in text or "song" in text or "ga" in text:
        response = random.choice(SINGING_REPLIES).format(name=nickname)
        voice = await elevenlabs_voice(response)
        if isinstance(voice, bytes):
            filepath = save_media_to_folder(voice, "voice", ext="mp3")
            with open(filepath, "rb") as f:
                await bot.send_voice(user_id, f)
    elif "challenge" in text:
        level = user_data[user_id]["challenge_level"]
        if level < len(CHALLENGE_LEVELS):
            challenge = CHALLENGE_LEVELS[level]
            response = f"🎯 *{nickname}*, yeh tera challenge: {challenge['task']}\n*Reward*: {challenge['reward']} ♥"
        else:
            response = f"🏆 *{nickname}*, tune saare challenges khatam kar diye! ★"
    else:
        prompt = f"Generate a {mood} Hinglish response for {nickname} based on: {text}"
        response = await huggingface_generate(prompt)
        if "error" in response.lower():
            response = random.choice(FLIRTY_REPLIES).format(name=nickname)

    keyboard = InlineKeyboardMarkup()
    keyboard.row(InlineKeyboardButton("📖 Menu", callback_data="help"),
                 InlineKeyboardButton("♥ Shayari", callback_data="shayari"))
    keyboard.add(InlineKeyboardButton("🎤 Sing", callback_data="sing"))
    await bot.reply_to(message, response, parse_mode="Markdown", reply_markup=keyboard)
    user_chat_history[user_id].append((text, response, get_ist_time_str()))
    save_user_data()

    emojis = ["👍", "❤", "🔥", "🥰", "😍", "🤣", "😘", "😎"]
    await bot.set_message_reaction(message.chat.id, message.message_id, random.choice(emojis))

# 🚀 Startup
async def main():
    print(f"🌌 {ORIGINAL_NAME} by {CREATOR} shuru ho gaya hai! ★")
    load_user_data()
    load_clone_data()
    asyncio.create_task(auto_maintenance())
    await bot.polling(none_stop=True)

if __name__ == "__main__":
    asyncio.run(main())