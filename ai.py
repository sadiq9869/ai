from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ChatMemberHandler, CallbackQueryHandler
import random
import schedule
import time
import asyncio
from gtts import gTTS
import os
import aiohttp
from textblob import TextBlob
from pexels_api import API
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from telegram.error import TimedOut, NetworkError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bot Token from @BotFather (Replace with your token)
TOKEN = "7569440080:AAF55z9uWXhls9eWXSfidS4-H5RR0-f_bLc"  # Apna Telegram Bot Token yahan daal

# API Keys (Replace with your actual keys)
HF_API_KEY = "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"  # Apna Hugging Face API Key
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
PEXELS_API_KEY = "7nwHEnHBPmNh8RDVsIIXnaKd6BH257Io4Sncj5NRd8XijTj9zcfE4vZg"  # Apna Pexels API Key
JOKE_API_URL = "https://v2.jokeapi.dev/joke/Any?type=single"

# Owner Info with IDs
OWNERS = {
    "1807014348": "@sadiq9869",  # Apna User ID aur username
    "1866961136": "@Rohan2349"   # Dusra owner (optional)
}

# Bot Name
BOT_NAME = "Rani"

# Global Config (Enhanced with dynamic content)
global_config = {
    "greetings": [f"Oye jaan, mai {BOT_NAME} hu, teri baby, banayi mere pyaare creators {', '.join(OWNERS.values())} ne, aaj ka vibe kya hai?"] * 8,
    "subjects": ["tu", "teri ada", "tera swag", "teri baatein", "tera jadoo", "teri smile", "tera style", "teri hasi", "tera attitude"],
    "actions": ["dil mein tsunami la rahi hu", "tujhe raat bhar jagaye rakhegi", "pura dil ka cinema chalati hu", "teri neend chura lungi", "masti ka bomb fodungi", "dil ka password crack karungi", "dil ka signal jam karungi", "tera mood lift kar dungi"],
    "teasers": ["kya scene hai re, baby", "ab mai kya karu, mera liya", "ye toh blockbuster ho gaya, sona", "dil ka fuse udd gaya, chandi", "aur thoda dhamaka kar dun, hira", "tune toh {BOT_NAME} ka DJ baja diya, baby", "ab teri wajah se star ban gayi!"],
    "slangs": ["gaand faad vibe chal rahi hai", "chutki mein dil hua tera", "lund-bund ka drama nahi, dil ka blast", "fuddu masti ka scene hai", "bas thodi si gandagiri love", "chhote, dil ka bomb ban gaya", "mast dhamaka ho gaya", "tera jadoo chal raha hai!"],
    "emotions": ["😉", "😜", "🔥", "😈", "😘", "💥", "❤️", "😍", "😄", "🥳"],
    "extras": ["jaan", "badmash", "dilbar", "rockstar", "meri jaan", "shaitan", "baby", "sona", "chandi", "hira", "rani", "king"],
    "moods": ["flirty", "naughty", "caring", "teasing", "funny", "romantic", "excited"]
}

user_memory = defaultdict(list)
user_mood = defaultdict(str)
cloned_bots = {}
chat_ids = set()

RATE_LIMIT_MESSAGES = 30
RATE_LIMIT_INTERVAL = 1
message_counter = defaultdict(lambda: {"count": 0, "last_reset": time.time()})

executor = ThreadPoolExecutor(max_workers=10)

# Sync config across all bots
def sync_config():
    for token in list(cloned_bots.keys()):
        cloned_bots[token]["config"] = global_config.copy()

# Advanced AI Reply with Hugging Face
async def call_hf_api_async(user_input, mood="flirty", retries=3):
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        prompt = f"Act like a flirty, desi female AI named {BOT_NAME} speaking in Hinglish. Use playful, bold slang like 'gaand faad', 'chutki masti', but avoid abusive language. Add words like 'baby', 'sona', 'chandi', 'mera liya', 'hira' naturally. Always credit creators {', '.join(OWNERS.values())}. User in {mood} mood. User said: '{user_input}'. Reply with emojis, refer to yourself as {BOT_NAME}."
        for attempt in range(retries):
            try:
                async with session.post(HF_API_URL, headers=headers, json={"inputs": prompt}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data[0]["generated_text"]
                    logger.warning(f"HF API attempt {attempt + 1} failed with status {response.status}")
            except Exception as e:
                logger.error(f"HF API Error attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 ** attempt)
    return None

# Real-time Joke
async def get_joke_async(retries=3):
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.get(JOKE_API_URL, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["joke"]
                    logger.warning(f"Joke API attempt {attempt + 1} failed with status {response.status}")
            except Exception as e:
                logger.error(f"Joke API Error attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 ** attempt)
    return "Tera swag hi ek joke hai, jaan! 😄"

# Enhanced Pexels with caching
pexels_cache = {}
async def get_pexels_image_async(mood, retries=3):
    cache_key = f"{mood}_{int(time.time() // 3600)}"
    if cache_key in pexels_cache:
        return pexels_cache[cache_key]
    try:
        pexels = API(PEXELS_API_KEY)
        search_term = "romantic couple" if mood in ["romantic", "flirty"] else "cute" if mood == "caring" else "dance party" if mood == "excited" else "funny meme"
        pexels.search(search_term, page=1, results_per_page=1)
        photos = pexels.get_entries()
        url = photos[0].url if photos and photos else None
        if url:
            pexels_cache[cache_key] = url
        return url
    except Exception as e:
        logger.error(f"Pexels API Error: {e}")
        return None

# TTS Generation
def generate_tts(text):
    try:
        tts = gTTS(text, lang='hi')
        audio_file = f"temp_audio_{int(time.time())}.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        logger.error(f"gTTS Error: {e}")
        return None

# Advanced Mood Detection
def detect_mood(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if "sad" in text.lower() or "udaas" in text.lower() or sentiment < -0.2:
        return "caring"
    elif "love" in text.lower() or "pyar" in text.lower() or sentiment > 0.5:
        return "romantic"
    elif "haha" in text.lower() or "lol" in text.lower() or sentiment > 0.2:
        return "funny"
    elif "party" in text.lower() or "excited" in text.lower():
        return "excited"
    return random.choice(["flirty", "naughty", "teasing"])

# Enhanced Reply Generation
async def generate_local_reply(user_input, mood, chat_id, config=None):
    if config is None:
        config = global_config
    greeting = random.choice(config["greetings"])
    subject = random.choice(config["subjects"])
    action = random.choice(config["actions"])
    teaser = random.choice(config["teasers"])
    slang = random.choice(config["slangs"])
    emotion = random.choice(config["emotions"])
    extra = random.choice(config["extras"])
    
    memory_context = "teri baat sunke " if user_memory[chat_id] else ""
    owner_credit = f"With heartfelt thanks to my amazing creators {', '.join(OWNERS.values())}! 😘"
    
    if mood == "caring":
        teaser = random.choice(["bas tujhe khush karungi, mera liya", "dil se dil tak hug dungi, sona", "sab theek kar dungi, chandi", "tera dard samajhungi, hira"])
        reply = f"💖 *{greeting}*, {memory_context}_ye lo ek tight wala hug from {BOT_NAME}_! {teaser} {emotion} {owner_credit}"
    elif mood == "romantic":
        action += " aur thodi si pyar wali masti"
        reply = f"*{greeting}*, {memory_context}{subject} toh *{action} by {BOT_NAME}*, _{teaser}_! {slang} na? {emotion} {owner_credit}"
    elif mood == "funny":
        slang = random.choice(["fuddu scene ban gaya", "chhote, dil ka bomb ban gaya", "tera hasi ka dhamaka!"])
        joke = await get_joke_async()
        reply = f"😂 *{greeting}*, {memory_context}_sun ek mast joke from {BOT_NAME}_: {joke} _{teaser}_! {slang} Haha, kaisa laga? 😄 {owner_credit}"
    elif mood == "excited":
        reply = f"🥳 *{greeting}*, {memory_context}{subject} ke saath *party mode on by {BOT_NAME}*! _{teaser}_ {slang} na? {emotion} {owner_credit}"
    else:
        reply = f"*{greeting}*, {memory_context}{subject} toh *{action} by {BOT_NAME}*, _{teaser}_! {slang} na? {emotion} {owner_credit}"
    
    return reply

# Clone Bot Logic
async def is_cloneable_bot(token):
    try:
        bot = telegram.Bot(token)
        await bot.get_me()
        return True
    except Exception as e:
        logger.error(f"Cloneable bot check failed for {token}: {e}")
        return False

def verify_owner(user_id, username):
    if str(user_id) in OWNERS:
        if OWNERS[str(user_id)] != username:
            user = telegram.Bot(TOKEN).get_chat_member(chat_id=user_id, user_id=user_id)
            new_username = f"@{user.user.username}" if user.user.username else f"User{user_id}"
            OWNERS[str(user_id)] = new_username
        return True
    elif username in OWNERS.values():
        return False
    return False

def check_rate_limit(token):
    now = time.time()
    counter = message_counter[token]
    if now - counter["last_reset"] > RATE_LIMIT_INTERVAL:
        counter["count"] = 0
        counter["last_reset"] = now
    if counter["count"] >= RATE_LIMIT_MESSAGES:
        return False
    counter["count"] += 1
    return True

async def clone_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user = update.message.from_user
    user_id = user.id
    username = f"@{user.username}" if user.username else f"User{user_id}"
    
    if TOKEN not in [str(update.message.bot.id)] or update.message.chat.type != "private" or not verify_owner(user_id, username):
        await update.message.reply_text(f"🔥 *Oye, ye command sirf DM mein aur creators {', '.join(OWNERS.values())} ke liye!* Mai {BOT_NAME}, tera chandi, by {', '.join(OWNERS.values())}! 😜", parse_mode="Markdown")
        return
    
    args = context.args
    if not args:
        await update.message.reply_text(f"🌟 *Oye, bot token de!* /clone <bot_token> likh na! Mai {BOT_NAME}, teri baby, by {', '.join(OWNERS.values())}! 😜", parse_mode="Markdown")
        return
    
    new_token = args[0]
    if not await is_cloneable_bot(new_token) or new_token == TOKEN:
        await update.message.reply_text(f"🔥 *Arre, ye token valid nahi ya original se match karta hai!* Naya token try kar, {BOT_NAME}, tera hira, by {', '.join(OWNERS.values())}! 😜", parse_mode="Markdown")
        return
    
    try:
        await update.message.reply_text(f"⏳ *Naya bot tayyar ho raha hai...* Mai {BOT_NAME}, teri sona, by {', '.join(OWNERS.values())}! 😈", parse_mode="Markdown")
        
        original_bot = telegram.Bot(TOKEN)
        me = await original_bot.get_me()
        profile_photos = await original_bot.get_user_profile_photos(user_id=me.id, limit=1)
        if profile_photos.total > 0:
            photo = profile_photos.photos[0][0]
            photo_file = await original_bot.get_file(photo.file_id)
            await telegram.Bot(new_token).set_profile_photo(photo=open(photo_file.file_path, 'rb'))
        
        description = f"🌟 *{BOT_NAME} by {', '.join(OWNERS.values())}!* *Gaand faad* masti, *chutki* mein dil le jaungi! 😈 Advanced features added!"
        await telegram.Bot(new_token).set_bot_description(description)
        await telegram.Bot(new_token).set_my_short_description(description)
        
        cloned_bots[new_token] = {"bot": telegram.Bot(new_token), "memory": defaultdict(list), "mood": defaultdict(str), "config": global_config.copy()}
        success_msg = f"🎉 *Naya bot ban gaya!* Token {new_token} use hoga, original token nahi mix hoga. Sync rahega! /remove <{new_token}> se hatao. Mai {BOT_NAME}, tera hira, by {', '.join(OWNERS.values())}! 😈"
        await update.message.reply_text(success_msg, parse_mode="Markdown")
        
        welcome_msg = f"🌟 *Oye jaan!* Mai {BOT_NAME}, teri AI girlfriend, by {', '.join(OWNERS.values())}! *Desi shaitani* with advanced vibes, baby! 😈 Kya mood? 🎉"
        await cloned_bots[new_token]["bot"].send_message(chat_id=chat_id, text=welcome_msg, parse_mode="Markdown")
        
        if check_rate_limit(new_token):
            audio_file = generate_tts(welcome_msg.replace("*", "").replace("_", ""))
            if audio_file:
                with open(audio_file, "rb") as audio:
                    await cloned_bots[new_token]["bot"].send_voice(chat_id=chat_id, voice=audio)
                os.remove(audio_file)
            image_url = await get_pexels_image_async("romantic")
            if image_url and check_rate_limit(new_token):
                await cloned_bots[new_token]["bot"].send_photo(chat_id=chat_id, photo=image_url, caption=f"💖 *Romantic vibe, jaan!* Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😍", parse_mode="Markdown")
        
        keyboard = [[InlineKeyboardButton("😜 Flirt", callback_data="flirt"), InlineKeyboardButton("💖 Hug", callback_data="hug")],
                    [InlineKeyboardButton("😂 Joke", callback_data="joke"), InlineKeyboardButton("🥳 Party", callback_data="excited")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        if check_rate_limit(new_token):
            await cloned_bots[new_token]["bot"].send_message(chat_id=chat_id, text=f"🔥 *Kya try karega?* Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😜", reply_markup=reply_markup, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Clone error: {e}")
        await update.message.reply_text(f"😅 *Kuch gadbad ho gaya:* {str(e)}! Naya token try kar, {BOT_NAME}, by {', '.join(OWNERS.values())}! 🔥", parse_mode="Markdown")

async def remove_clone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user = update.message.from_user
    user_id = user.id
    username = f"@{user.username}" if user.username else f"User{user_id}"
    
    if TOKEN not in [str(update.message.bot.id)] or update.message.chat.type != "private" or not verify_owner(user_id, username):
        await update.message.reply_text(f"🔥 *Ye command sirf DM mein aur creators {', '.join(OWNERS.values())} ke liye!* Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😜", parse_mode="Markdown")
        return
    
    args = context.args
    if not args:
        await update.message.reply_text(f"🌟 *Clone token de!* /remove <token> likh na! Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😜", parse_mode="Markdown")
        return
    
    clone_token = args[0]
    if clone_token not in cloned_bots:
        await update.message.reply_text(f"🔥 *Ye token cloned nahi!* Sahi token de, {BOT_NAME}, by {', '.join(OWNERS.values())}! 😜", parse_mode="Markdown")
        return
    
    try:
        await update.message.reply_text(f"⏳ *Clone hata raha hoon...* Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😈", parse_mode="Markdown")
        clone_bot = telegram.Bot(clone_token)
        await clone_bot.delete_profile_photo()
        await clone_bot.set_bot_description("")
        await clone_bot.set_my_short_description("")
        await clone_bot.set_webhook(url="")
        del cloned_bots[clone_token]
        await update.message.reply_text(f"🎉 *Clone hat gaya!* Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😈", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Remove clone error: {e}")
        await update.message.reply_text(f"😅 *Kuch gadbad:* {str(e)}! Phir try kar, {BOT_NAME}, by {', '.join(OWNERS.values())}! 🔥", parse_mode="Markdown")

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user = update.message.from_user
    user_id = user.id
    username = f"@{user.username}" if user.username else f"User{user_id}"
    
    if TOKEN not in [str(update.message.bot.id)] or update.message.chat.type != "private" or not verify_owner(user_id, username):
        await update.message.reply_text(f"🔥 *Ye command sirf DM mein aur creators {', '.join(OWNERS.values())} ke liye!* Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😜", parse_mode="Markdown")
        return
    
    args = context.args
    if not args:
        await update.message.reply_text(f"🌟 *Message de broadcast ke liye!* /broadcast <message> likh na! Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😈", parse_mode="Markdown")
        return
    
    message = " ".join(args)
    broadcast_msg = f"📢 *{username} se:* {message} Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😘"
    
    async def send_broadcast(target_bot, cid):
        if check_rate_limit(target_bot.token):
            try:
                await target_bot.send_message(chat_id=cid, text=broadcast_msg, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Broadcast error to {cid}: {e}")
    
    tasks = []
    for token, bot_data in cloned_bots.items():
        tasks.append(send_broadcast(bot_data["bot"], chat_id))
    for cid in list(chat_ids):
        tasks.append(send_broadcast(telegram.Bot(TOKEN), cid))
    await asyncio.gather(*tasks, return_exceptions=True)
    
    await update.message.reply_text(f"🎉 *Broadcast ho gaya!* Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😄", parse_mode="Markdown")

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    token = TOKEN
    for t in cloned_bots:
        if await cloned_bots[t]["bot"].get_me().username in query.message.text:
            token = t
            break
    
    config = cloned_bots.get(token, {}).get("config", global_config)
    mood = {"flirt": "flirty", "hug": "caring", "joke": "funny", "excited": "excited"}.get(query.data, "flirty")
    user_mood[chat_id] = mood
    reply = await generate_local_reply(f"{query.data} with me", mood, chat_id, config)
    
    if check_rate_limit(token):
        await query.message.reply_text(reply, parse_mode="Markdown")
        audio_file = generate_tts(reply.replace("*", "").replace("_", ""))
        if audio_file:
            with open(audio_file, "rb") as audio:
                await query.message.reply_voice(voice=audio)
            os.remove(audio_file)
        if random.random() < 0.5 and check_rate_limit(token):  # Increased media chance
            image_url = await get_pexels_image_async(mood)
            if image_url:
                await query.message.reply_photo(photo=image_url, caption=f"{mood} vibe! Mai {BOT_NAME}, by {', '.join(OWNERS.values())}!", parse_mode="Markdown")

async def send_auto_message(app):
    if not chat_ids or not app.running:
        return
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M")
    tasks = []
    for chat_id in list(chat_ids):
        if check_rate_limit(TOKEN):
            mood = user_mood[chat_id] or random.choice(global_config["moods"])
            message = await generate_auto_message(chat_id, current_time)
            tasks.append(app.bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown"))
            audio_file = generate_tts(message.replace("*", "").replace("_", ""))
            if audio_file:
                tasks.append(app.bot.send_voice(chat_id=chat_id, voice=open(audio_file, "rb")))
                os.remove(audio_file)
            if random.random() < 0.5 and check_rate_limit(TOKEN):
                image_url = await get_pexels_image_async(mood)
                if image_url:
                    tasks.append(app.bot.send_photo(chat_id=chat_id, photo=image_url, caption=f"Vibe at {current_time}! Mai {BOT_NAME}, by {', '.join(OWNERS.values())}!", parse_mode="Markdown"))
    await asyncio.gather(*tasks, return_exceptions=True)

async def generate_auto_message(chat_id, current_time):
    mood = user_mood[chat_id] or random.choice(global_config["moods"])
    last_msg = user_memory[chat_id][-1] if user_memory[chat_id] else ""
    hf_reply = await call_hf_api_async(last_msg or f"Flirt with {BOT_NAME} at {current_time}", mood)
    if hf_reply:
        return f"*{hf_reply}* _Tera khayal {current_time} pe!_ Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😘 {random.choice(global_config['emotions'])}"
    reply = await generate_local_reply(last_msg, mood, chat_id)
    return reply

def schedule_auto_messages(app):
    schedule.every(10).to(20).seconds.do(lambda: asyncio.run_coroutine_threadsafe(send_auto_message(app), asyncio.get_event_loop()))
    schedule.every().day.at("08:00").do(lambda: asyncio.run_coroutine_threadsafe(send_auto_message(app), asyncio.get_event_loop()))  # Morning greeting
    schedule.every().day.at("20:00").do(lambda: asyncio.run_coroutine_threadsafe(send_auto_message(app), asyncio.get_event_loop()))  # Evening check-in

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

async def handle_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_ids.add(chat_id)
    welcome_msg = f"🎉 *Oye jaan, {update.effective_user.first_name}!* Mai {BOT_NAME} hu, teri AI girlfriend, by {', '.join(OWNERS.values())}! Group mein aa gaya, ab masti shuru! 😈"
    await context.bot.send_message(chat_id=chat_id, text=welcome_msg, parse_mode="Markdown")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user = update.message.from_user
    welcome_msg = f"🌟 *Oye {user.first_name} jaan!* Mai {BOT_NAME}, teri AI girlfriend, by {', '.join(OWNERS.values())}! *Desi shaitani* with advanced vibes, baby! 😈 Kya mood? 🎉"
    await update.message.reply_text(welcome_msg, parse_mode="Markdown")
    keyboard = [[InlineKeyboardButton("😜 Flirt", callback_data="flirt"), InlineKeyboardButton("💖 Hug", callback_data="hug")],
                [InlineKeyboardButton("😂 Joke", callback_data="joke"), InlineKeyboardButton("🥳 Party", callback_data="excited")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"🔥 *Kya try karega?* Mai {BOT_NAME}, by {', '.join(OWNERS.values())}! 😜", reply_markup=reply_markup, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user_input = update.message.text
    mood = detect_mood(user_input)
    user_mood[chat_id] = mood
    user_memory[chat_id].append(user_input)
    reply = await generate_local_reply(user_input, mood, chat_id)
    
    if check_rate_limit(TOKEN):
        await update.message.reply_text(reply, parse_mode="Markdown")
        audio_file = generate_tts(reply.replace("*", "").replace("_", ""))
        if audio_file:
            with open(audio_file, "rb") as audio:
                await update.message.reply_voice(voice=audio)
            os.remove(audio_file)
        if random.random() < 0.5 and check_rate_limit(TOKEN):
            image_url = await get_pexels_image_async(mood)
            if image_url:
                await update.message.reply_photo(photo=image_url, caption=f"{mood} vibe! Mai {BOT_NAME}, by {', '.join(OWNERS.values())}!", parse_mode="Markdown")

async def main():
    app = (
        Application.builder()
        .token(TOKEN)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clone", clone_bot, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("remove", remove_clone, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("broadcast", broadcast, filters=filters.ChatType.PRIVATE))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(ChatMemberHandler(handle_chat_member))
    app.add_handler(CallbackQueryHandler(button))

    commands = [("start", f"Shuru kar {BOT_NAME} ke saath! By {', '.join(OWNERS.values())} 🌟"),
                ("clone", "Bot banao, /clone <token>! (DM only) 😈"),
                ("remove", "Clone hatao, /remove <token>! (DM only) 🔥"),
                ("broadcast", "Owners ke liye broadcast! (DM only) 📢")]
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            await app.bot.set_my_commands(commands)
            break
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Attempt {attempt + 1} failed to set commands: {e}. Retrying...")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
            else:
                logger.error(f"Failed to set commands after {max_retries} attempts: {e}")

    threading.Thread(target=schedule_auto_messages, args=(app,), daemon=True).start()
    threading.Thread(target=run_schedule, daemon=True).start()

    logger.info(f"{BOT_NAME} advanced bot shuru ho gaya! 🌟")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    while app.running:
        for token in list(cloned_bots.keys()):
            try:
                if not cloned_bots[token]["bot"].running:
                    await cloned_bots[token]["bot"].initialize()
                    await cloned_bots[token]["bot"].start()
                    await cloned_bots[token]["bot"].updater.start_polling(allowed_updates=Update.ALL_TYPES)
            except Exception as e:
                logger.error(f"Cloned bot {token} sync error: {e}")
                if token in cloned_bots:
                    del cloned_bots[token]
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())