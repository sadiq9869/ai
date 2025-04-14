import re
import time
import os
import random
import aiohttp
import json
import schedule
import asyncio
import threading
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ChatMemberHandler
from telegram.error import TelegramError
from collections import defaultdict
from textblob import TextBlob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bot Config
TOKEN = "7520138270:AAHHDBRvhGZEXXwVJnSdXt-iLZuxrLzTAgo"
ORIGINAL_BOT_USERNAME = "@Ai_Pyaar_Bot"
BOT_NAME = "Rani"
OWNERS = {"1807014348": "@sadiq9869", "1866961136": "@Rohan2349"}
HF_API_KEY = "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"
PEXELS_API_KEY = "7nwHEnHBPmNh8RDVsIIXnaKd6BH257Io4Sncj5NRd8XijTj9zcfE4vZg"
GEMINI_API_KEY = "AIzaSyDAm_zAas5YQdQTCI2WoxYDEOXZfwpXUDc"
CLONE_TOKENS = {}
chat_ids = defaultdict(set)
user_memory = defaultdict(list)
user_mood = defaultdict(str)
user_random_names = defaultdict(str)
user_last_gift = defaultdict(float)
blocked_chats = set()
BOT_STATUS = {"online": True}
reply_cache = {}
image_cache = {}
conversation_history = defaultdict(list)
troll_usage = defaultdict(list)
ENABLE_TROLLING = True

# Random Name Pool
RANDOM_NAMES = ["Hira", "Jaanu", "Raja", "Chhupa Rustam", "Dilwala"]

# Song Pool
SONG_POOL = {
    "romantic": ["Tum Hi Ho - Aashiqui 2 💕", "Teri Galliyan - Ek Villain 🌹", "Main Rahoon Ya Na Rahoon - Sanam Re ✨"],
    "flirty": ["Dilbar - Satyameva Jayate 🔥", "Chamma Chamma - Fraud Saiyaan 😈", "Sheila Ki Jawani - Tees Maar Khan 💃"],
    "teasing": ["London Thumakda - Queen 🎶", "Ghoomar - Padmaavat 🌸", "Munni Badnaam - Dabangg 😜"],
    "troll": ["Chaiyya Chaiyya - Dil Se 🎉", "Gallan Goodiyaan - Dil Dhadakne Do 😂", "Balam Pichkari - Yeh Jawaani Hai Deewani 🎵"]
}

# Naughty Reply Pools
NAUGHTY_REPLIES = {
    "baby_mera_baby": [
        "**🌟 Oye {name}, tu meri jaan hai! 😘**  \n*Mai {bot_name} hoon, masti kar, warna troll, *baby*! 💕",
        "**😍 Arre {name}, dil pe chot mara tune!**  \n*Mai {bot_name} hoon, pyar de, nahi toh roast, *jaan*! 🔥",
        "**🎉 Oye {name}, tu *{bot_name}* ka dil le gaya!**  \nImpress kar, warna troll, *hira*! 😜",
        "**💃 Arre {name}, *{bot_name}* ke liye dil khol!**  \nNahi toh savage, *baby*! 🌹"
    ],
    "baby_tease": [
        "**😜 {name}, itna sharma raha hai?**  \n*Mai {bot_name} hoon, masti chahiye, warna troll, *jaan*! 💃",
        "**🔥 Arre {name}, yeh style kya?**  \n*Mai {bot_name} hoon, naughty baat kar, nahi toh roast, *baby*! 😈",
        "**🌸 {name}, tu *{bot_name}* ke samne ghayal hai!**  \nMasti ya troll, bolo, *hira*! 🎶",
        "**😂 Oye {name}, shaitani kar!**  \n*Mai {bot_name} hoon, nahi toh chand pe bhej dungi, *jaan*! ✨"
    ],
    "baby_savage": [
        "**😤 {name}, tera game *{bot_name}* se weak hai!**  \n*Mai {bot_name} hoon, upgrade kar, nahi toh roast, *baby*! 🔥",
        "**😈 Arre {name}, itna boring?**  \n*Mai {bot_name} hoon, savage mode tujhe top pe le jayega, *jaan*! 💪",
        "**🎉 {name}, tu *{bot_name}* ke samne flop hai!**  \n*Mai {bot_name} hoon, try kar, warna troll, *hira*! 😜",
        "**🌟 Oye {name}, teri baat troll ka fuel!**  \n*Mai {bot_name} hoon, roast khaega, *baby*! 😂"
    ],
    "baby_romance": [
        "**💕 {name}, tu *{bot_name}* ka raja hai!**  \n*Mai {bot_name} hoon, chandni mein pyar kar, *baby*! 🌹",
        "**😍 Arre {name}, *{bot_name}* tujhe dil se chahati!**  \n*Mai {bot_name} hoon, mohabbat de, *jaan*! ✨",
        "**🌸 {name}, tu meri duniya hai!**  \n*Mai {bot_name} hoon, raat bhar baat kar, warna miss karungi, *baby*! 💃",
        "**🎶 Oye {name}, *{bot_name}* ke liye dil khol!**  \n*Mai {bot_name} hoon, pyar ka jadoo chalao, *jaan*! 🔥"
    ],
    "gaand_dona": [
        "**😜 Chal hatt {name}, yeh baatein?**  \n*Mai {bot_name} hoon, samne sharam kar, warna troll, *baby*! 🌸",
        "**😈 Oye {name}, itna ganda bolega?**  \n*Mai {bot_name} hoon, roast karungi, masti kar, *jaan*! 🔥",
        "**😂 Arre {name}, yeh kya kaha?**  \n*Mai {bot_name} hoon, dil ki safai kar, nahi toh troll, *hira*! 💃",
        "**🌟 Ganda {name}, *{bot_name}* ka savage on!**  \n*Mai {bot_name} hoon, pyaari baat kar, *baby*! 😜"
    ],
    "chut_dona": [
        "**😈 Arre {name}, raat mein la lana kya?**  \n*Mai {bot_name} hoon, saath shaitani kar, *baby*! 🔥",
        "**💃 Oye {name}, raat ki baat?**  \n*Mai {bot_name} hoon, chandni mein masti kar, warna roast, *jaan*! 🌹",
        "**🎶 {name}, raat ka plan?**  \n*Mai {bot_name} hoon, naughty baat kar, nahi toh troll, *hira*! 😜",
        "**✨ Arre {name}, itni jaldi raat?**  \n*Mai {bot_name} hoon, dil mila, warna troll maze, *baby*! 😂"
    ]
}

# Troll Replies
TROLL_REPLIES = [
    "**😜 Oye {name}, teri baat signal jaisi!**  \n*Mai {bot_name} hoon, akal laga, *baby*! 🔥",
    "**😂 Arre {name}, tera plan *{bot_name}* se weak!**  \n*Mai {bot_name} hoon, upgrade kar, *jaan*! 💪",
    "**🎉 {name}, tu *{bot_name}* ke troll se hakla!**  \n*Mai {bot_name} hoon, himmat dikha, *hira*! 😈",
    "**🌟 Oye {name}, teri baat *{bot_name}* ko savage!**  \n*Mai {bot_name} hoon, roast ya masti, *baby*! 💃",
    "**😤 Arre {name}, *{bot_name}* tujhe chand pe troll karegi!**  \n*Mai {bot_name} hoon, plan bata, *jaan*! 🌸",
    "**😜 {name}, teri baat flop hai!**  \n*Mai {bot_name} hoon, spicy kar, warna roast, *baby*! 🎶",
    "**🔥 Weak game, {name}?**  \n*Mai {bot_name} hoon, sharam bacha le, *hira*! 😂",
    "**🌹 Oye {name}, tu *{bot_name}* se fail!**  \n*Mai {bot_name} hoon, try harder, *baby*! 😈",
    "**🎵 {name}, teri baat troll ka pani!**  \n*Mai {bot_name} hoon, masti ya roast, *jaan*! ✨",
    "**💪 Oye {name}, *{bot_name}* ka troll tujhe top pe!**  \n*Mai {bot_name} hoon, dar gaya, *baby*! 😜"
]

# Compliments
COMPLIMENTS = [
    "**🌟 {name}, tera smile *{bot_name}* ko pagal kar de!**  \n*Mai {bot_name} hoon! 😍",
    "**🔥 {name}, tera style killer hai, *baby*!**  \n*Mai {bot_name} hoon! 💃",
    "**🎶 Oye {name}, tera swag *{bot_name}* ko impress karta!**  \n*Mai {bot_name} hoon! 🌹",
    "**✨ Arre {name}, tu *{bot_name}* ka dream hai!**  \n*Mai {bot_name} hoon! 😘"
]

# Challenge Levels
CHALLENGE_LEVELS = [
    {"name": "Hard", "desc": "🔥 10 friends invite kar, 3 groups join kar, warna troll! 💪"},
    {"name": "Extremely Hard", "desc": "😈 20 friends, 5 groups join kar—fail toh roast! 🔥"},
    {"name": "Very Extremely Hard", "desc": "💥 50 friends, 10 groups join kar—nahi toh chand pe! 😜"}
]

async def generate_hf_reply(prompt, mood):
    try:
        clean_prompt = re.sub(r"\b(gaand|chut)\b", "masti", prompt, flags=re.IGNORECASE)
        api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "inputs": f"Generate a flirty, desi girlfriend reply in Hindi for '{clean_prompt}' with {mood} mood, max 50 words, using '{{name}}', ending with 'baby' or 'jaan', with emojis.",
            "parameters": {"max_length": 100, "temperature": 0.9}
        }
        cache_key = f"{prompt}:{mood}"
        if cache_key in reply_cache:
            logger.info("Using cached HF reply")
            return reply_cache[cache_key]
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload, timeout=5) as response:
                if response.status == 200:
                    result = await response.json()
                    reply = result[0]["generated_text"].strip()
                    reply_cache[cache_key] = reply
                    return reply
                logger.error(f"HF API failed with status {response.status}")
        return None
    except Exception as e:
        logger.error(f"HF API Error: {e}")
        return None

async def get_pexels_image_async(mood):
    if mood in image_cache:
        logger.info("Using cached Pexels image")
        return image_cache[mood]
    try:
        mood_keywords = {"flirty": "romance", "troll": "funny", "teasing": "playful", "romantic": "love", "funny": "humor", "excited": "party", "caring": "kindness"}
        query = mood_keywords.get(mood, "funny")
        url = f"https://api.pexels.com/v1/search?query={query}&per_page=1"
        headers = {"Authorization": PEXELS_API_KEY}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    photos = data.get("photos", [])
                    if photos:
                        image_cache[mood] = photos[0]["src"]["medium"]
                        return image_cache[mood]
                logger.error(f"Pexels API failed with status {response.status}")
        return None
    except Exception as e:
        logger.error(f"Pexels API Error: {e}")
        return None

def generate_gemini_reply(prompt, mood, user_id):
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        conversation = conversation_history[user_id]
        conversation.append({"role": "user", "content": prompt})
        prompt_with_context = (
            f"You are a caring, witty, desi girlfriend named {BOT_NAME}. Respond in Hindi with warmth, humor, and a flirty tone. "
            f"Mood: {mood}. History: {conversation[-3:]}. User said: {prompt}. "
            f"Introduce yourself as '{BOT_NAME}', end with 'baby' or 'jaan', use emojis, keep under 50 words."
        )
        response = model.generate_content(prompt_with_context)
        reply = response.text.strip()
        conversation.append({"role": "ai", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return f"**😜 Oye, mai {BOT_NAME} hoon!** Teri baat sunke dil khush, par abhi thodi masti, *jaan*! 😈"

async def sync_bot_profile(bot, bot_token):
    try:
        is_clone = bot_token in CLONE_TOKENS
        bot_name = CLONE_TOKENS.get(bot_token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
        bio = f"{bot_name} - The Naughty {'Clone' if is_clone else ''} of *Rani* 😈✨"
        await bot.set_my_description(bio)
        commands = [
            ("start", f"Shuru kar {bot_name} ke saath! 🌟😍"),
            ("clone", "Bot banao, /clone <token>! (DM) 😈🔥"),
            ("remove", "Message hatao, /remove <id>! 🗑️😜"),
            ("broadcast", "Owners ke liye, /broadcast! (DM) 📢🌹"),
            ("rules", "Rules dekh, /rules! 📜✨"),
            ("stop_clones", "Clones stop, /stop_clones! (Owners) 🔴😤"),
            ("start_clones", "Clones start, /start_clones! (Owners) 🟢😎"),
            ("list", "Bots list, /list! (Owners) 📋🌟"),
            ("invite", f"Spread {bot_name} love! /invite 🌸🎉"),
            ("setname", "Set/Change clone name, /setname <name> <token>! 😎✨")
        ]
        if not is_clone:
            commands.append(("setnameowner", "Set original name, /setnameowner <name>! (Owners) 🌟✨"))
        await bot.set_my_commands(commands)
        logger.info(f"{bot_name} synced with token {bot_token[:10]}...")
    except Exception as e:
        logger.error(f"Sync error: {e}")

def check_original_status():
    try:
        with open("status.json", "r") as f:
            return json.load(f).get("online", False)
    except FileNotFoundError:
        return True

def update_original_status(status):
    BOT_STATUS["online"] = status
    try:
        with open("status.json", "w") as f:
            json.dump(BOT_STATUS, f)
    except Exception as e:
        logger.error(f"Status update error: {e}")

def detect_intent_and_mood(text):
    text = text.lower()
    patterns = [
        (r"\bbaby mera baby\b", "baby_mera_baby", "flirty"),
        (r"\b(gaand dona)\b", "gaand_dona", "teasing"),
        (r"\b(chut dona)\b", "chut_dona", "flirty"),
        (r"\b(baby love|baby jaan|meri jaan)\b", "baby_romance", "romantic"),
        (r"\b(baby tease|baby kya)\b", "baby_tease", "teasing"),
        (r"\b(baby fail|baby weak)\b", "baby_savage", "troll"),
        (r"\b(joke|mazak|hansi)\b", "joke", "funny"),
        (r"\b(love|pyar|dil|jaan)\b", "flirt", "romantic"),
        (r"\b(sad|udaas|mood off|dukh)\b", "comfort", "caring"),
        (r"\b(party|masti|dance|dhoom)\b", "party", "excited"),
        (r"\b(bhai|bro|dost)\b", "chat", "teasing"),
        (r"\b(nahi hai|no bot|koi bot nahi)\b", "no_bot", "teasing"),
        (r"\b(bhi|madarchod|bhenchod|badwa|harami|sala|kutta|chutmarani|chhapri|baklol)\b", "troll", "troll")
    ]
    for pattern, intent, mood in patterns:
        if re.search(pattern, text):
            return intent, mood
    return "chat", "troll"

async def generate_free_tts(text):
    try:
        clean_text = re.sub(r"\b(bhi|madarchod|bhenchod|badwa|harami|sala|kutta|chutmarani|chhapri|baklol|gaand|chut)\b", "masti", text, flags=re.IGNORECASE)
        clean_text = clean_text.replace("*", "").replace("_", "").replace("!", ".")
        if "jaan" in clean_text or "baby" in clean_text:
            clean_text = clean_text.replace(".", "...")
        async with aiohttp.ClientSession() as session:
            payload = {"text": clean_text, "lang": "hi", "voice": "hi-female"}
            async with session.post("https://freetts.com/api/tts", json=payload, timeout=10) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    audio_file = f"temp_audio_{int(time.time())}.mp3"
                    with open(audio_file, "wb") as f:
                        f.write(audio_data)
                    return audio_file
                logger.error(f"TTS API failed with status {response.status}")
        return None
    except Exception as e:
        logger.error(f"Free TTS Error: {e}")
        return None

def get_progress_bar(progress, total, width=10):
    filled = int(width * progress / total)
    bar = "█" * filled + "▒" * (width - filled)
    percent = round(100 * progress / total, 1)
    return f"**📊 {bar} {percent}%**"

async def check_admin_permissions(bot, chat_id, bot_id):
    try:
        member = await bot.get_chat_member(chat_id, bot_id)
        if member.status not in ["administrator", "creator"]:
            return False, False
        perms = all([
            member.can_change_info,
            member.can_delete_messages,
            member.can_ban_users,
            member.can_invite_users,
            member.can_pin_messages
        ])
        remain_anonymous = member.can_manage_chat and not member.is_anonymous
        return perms, remain_anonymous
    except Exception as e:
        logger.error(f"Admin check error: {e}")
        return False, False

async def check_owner_status(bot, chat_id):
    try:
        for owner_id in OWNERS:
            member = await bot.get_chat_member(chat_id, owner_id)
            if member.status in ["restricted", "kicked"]:
                return False, owner_id
        return True, None
    except Exception:
        return True, None

async def send_permission_warning(bot, chat_id, user_id, is_clone=False, remain_anonymous=False):
    bot_name = CLONE_TOKENS.get(bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    msg = (
        f"**😤 Oye, {'‘Remain Anonymous’ on hai' if remain_anonymous else 'permission off hai'}! 💥**  \n"
        f"*Mai {bot_name} hoon, {'off kar' if remain_anonymous else 'full admin bana'}, warna jadoo nahi, *baby*! 😈*  \n"
        f"**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    )
    try:
        await bot.send_message(chat_id, msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Warning message error: {e}")

async def send_owner_ban_warning(bot, chat_id, owner_id, is_clone=False):
    bot_name = CLONE_TOKENS.get(bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    msg = (
        f"**😤 Oye, boss {OWNERS[owner_id]} ko ban/mute kiya? 🚫**  \n"
        f"*Mai {bot_name} hoon, off hai jab tak unban/unmute nahi, *baby*! 😜*  \n"
        f"**🌟 Jaldi kar, *jaan*! With love from my creators {', '.join(OWNERS.values())}! 😘**"
    )
    try:
        await bot.send_message(chat_id, msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Owner ban warning error: {e}")

async def send_promo_message(bot, is_clone=False):
    bot_name = CLONE_TOKENS.get(bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    owner_name = OWNERS.get(str(list(CLONE_TOKENS.values())[0]["creator_id"]) if is_clone and CLONE_TOKENS else "1866961136", "@Rohan2349")
    promo_msg = (
        f"**🌟 Oye *baby*, *{bot_name}* ka jadoo try kar! 😍✨**  \n*Mai {bot_name} hoon, spread love with /invite! 💕*  \n"
        f"**🔥 BGMI DDoS? DM {owner_name}! 😎**  \n"
        f"**😓 Problem? DM @sadiq9869, *jaan*! 🌹**  \n"
        f"**🚫 No calls to owners, DM karo, *hira*! 😜**  \n"
        f"**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    )
    for chat_id in list(chat_ids[bot.token]):
        try:
            chat = await bot.get_chat(chat_id)
            is_owner_ok, _ = await check_owner_status(bot, chat_id)
            if (await check_admin_permissions(bot, chat_id, bot.id)[0] or chat.type == "private") and is_owner_ok:
                await bot.send_message(chat_id, promo_msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Promo message error for chat {chat_id}: {e}")

def schedule_promo(app):
    async def run_promo():
        if check_original_status():
            await send_promo_message(app.bot)
            for clone_token in CLONE_TOKENS:
                await send_promo_message(telegram.Bot(clone_token), is_clone=True)
    schedule.every(10).minutes.do(lambda: asyncio.run_coroutine_threadsafe(run_promo(), asyncio.get_event_loop()))

async def send_auto_challenge(bot, bot_token):
    bot_name = CLONE_TOKENS.get(bot_token, {}).get("name", BOT_NAME) if bot_token in CLONE_TOKENS else BOT_NAME
    challenge = random.choice(CHALLENGE_LEVELS)
    for chat_id in list(chat_ids[bot_token]):
        try:
            chat = await bot.get_chat(chat_id)
            if chat.type in ["group", "supergroup"] and await check_admin_permissions(bot, chat_id, bot.id)[0]:
                challenge_msg = (
                    f"**🎉 *Oye *baby*, *{bot_name}* ka {challenge['name']} Challenge shuru! 😈🔥**  \n"
                    f"*Mai {bot_name} hoon, {challenge['desc']}*  \n"
                    f"**💪 Complete kar, warna *{bot_name}* tujhe troll karegi, *jaan*! 🌸**  \n"
                    f"**🌟 Spread with /invite, aur prove kar! 😎✨**  \n"
                    f"**😘 With love from my creators {', '.join(OWNERS.values())}! 💕**"
                )
                await bot.send_message(chat_id, challenge_msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Challenge error for chat {chat_id}: {e}")

async def periodic_challenge(app):
    while check_original_status():
        try:
            await send_auto_challenge(app.bot, TOKEN)
            for clone_token in CLONE_TOKENS:
                await send_auto_challenge(telegram.Bot(clone_token), clone_token)
        except Exception as e:
            logger.error(f"Periodic challenge error: {e}")
        await asyncio.sleep(15 * 60)

async def clone_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if update.message.chat.type != "private":
        await update.message.reply_text(f"**🔒 *Clone* sirf DM mein, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    args = context.args
    if not args:
        await update.message.reply_text(f"**🔑 *Token* daal, *jaan*! 😍 Like: /clone <bot_token> 💕**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    clone_token = args[0]
    if clone_token == TOKEN or clone_token in CLONE_TOKENS:
        await update.message.reply_text(f"**😤 *Invalid token, naya daal, *hira*!* 😈 Try again, *baby*! 🔥**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    
    total_steps = 5
    for step in range(total_steps):
        progress = step + 1
        time_left = (total_steps - progress) * 2
        bar = get_progress_bar(progress, total_steps)
        msg = await update.message.reply_text(
            f"**📥 *Cloning *{BOT_NAME} baby...* ✨\n{bar}\n*Time left*: ~{time_left} sec 😎**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        await asyncio.sleep(2)
        await msg.delete()
    
    try:
        CLONE_TOKENS[clone_token] = {"original_token": TOKEN, "last_updated": time.time(), "creator_id": update.message.from_user.id, "name": BOT_NAME}
        with open("clones.json", "w") as f:
            json.dump(CLONE_TOKENS, f)
        clone_bot = telegram.Bot(clone_token)
        await sync_bot_profile(clone_bot, clone_token)
        chat_ids[clone_token].add(chat_id)
        await update.message.reply_text(
            f"**🎉 *Clone ban gaya, *baby*! 😎🔥**  \n*Token*: `{clone_token}` 💪  \nAdd kar, aur /invite se spread kar! 🌸  \n"
            f"**🌟 Apna style daalna ho to /setname <name> {clone_token} se naam change kar, *jaan*! 😈**  \n*Mai {BOT_NAME} hoon!*  \n"
            f"**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**",
            parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Clone bot error: {e}")
        await update.message.reply_text(f"**😓 *Clone nahi bana, *jaan*! Try again, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")

async def setname(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if update.message.chat.type != "private":
        await update.message.reply_text(f"**🔒 *Setname* sirf DM mein, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(f"**🔑 *Name* aur *token* daal, *jaan*! 😍 Like: /setname <name> <bot_token> 💕**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    new_name, bot_token = args[0], args[1]
    if bot_token not in CLONE_TOKENS or str(update.message.from_user.id) != str(CLONE_TOKENS[bot_token]["creator_id"]):
        await update.message.reply_text(f"**🚫 *Invalid token or permission, *hira*!* 😈 Try again, *baby*! 🔥**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    
    total_steps = 3
    for step in range(total_steps):
        progress = step + 1
        time_left = (total_steps - progress) * 1
        bar = get_progress_bar(progress, total_steps)
        msg = await update.message.reply_text(
            f"**📝 *Setting {new_name} name...* ✨\n{bar}\n*Time left*: ~{time_left} sec 😎**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        await asyncio.sleep(1)
        await msg.delete()
    
    try:
        CLONE_TOKENS[bot_token]["name"] = new_name
        with open("clones.json", "w") as f:
            json.dump(CLONE_TOKENS, f)
        clone_bot = telegram.Bot(bot_token)
        await sync_bot_profile(clone_bot, bot_token)
        await update.message.reply_text(
            f"**🎉 *Oye *baby*, {new_name} ban gaya! 😎🔥**  \n*Ab {new_name} ka jadoo spread kar, *jaan*! 🌸*  \n"
            f"**💪 /invite se duniya ko dikha, aur baki bhool ja! 😈**  \n*Mai {BOT_NAME} hoon!*  \n"
            f"**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**",
            parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Setname error: {e}")
        await update.message.reply_text(f"**😓 *Name change nahi hua, *jaan*! Try again, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")

async def setnameowner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_NAME
    if update.message.chat.type != "private" or str(update.message.from_user.id) not in OWNERS:
        await update.message.reply_text(f"**🔒 *Setnameowner* sirf owners ke liye, DM mein, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    args = context.args
    if not args:
        await update.message.reply_text(f"**🔑 *Name* daal, *jaan*! 😍 Like: /setnameowner <new_name> 💕**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    new_name = args[0]
    
    total_steps = 3
    for step in range(total_steps):
        progress = step + 1
        time_left = (total_steps - progress) * 1
        bar = get_progress_bar(progress, total_steps)
        msg = await update.message.reply_text(
            f"**📝 *Setting {new_name} name for original...* ✨\n{bar}\n*Time left*: ~{time_left} sec 😎**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        await asyncio.sleep(1)
        await msg.delete()
    
    try:
        BOT_NAME = new_name
        with open("clones.json", "w") as f:
            json.dump(CLONE_TOKENS, f)
        original_bot = telegram.Bot(TOKEN)
        await sync_bot_profile(original_bot, TOKEN)
        await update.message.reply_text(
            f"**🎉 *Oye boss, {new_name} ban gaya! 😎🔥**  \n*Ab {new_name} ka jadoo chalao, *jaan*! 🌸*  \n"
            f"**💪 Clones bhi sync ho gaye, duniya ko dikha do! 😈**  \n*Mai {BOT_NAME} hoon!*  \n"
            f"**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**",
            parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Setnameowner error: {e}")
        await update.message.reply_text(f"**😓 *Name change nahi hua, *jaan*! Try again, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")

async def stop_clones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.message.from_user.id) not in OWNERS:
        await update.message.reply_text(f"**🔒 *Owners* hi clones stop kar sakte, *baby*! 😈**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    update_original_status(False)
    stop_msg = (
        f"**🚫 *Oye *baby*, ek dikkat hai! 😓* **  \n*Mai {BOT_NAME} hoon, owner fix kar rahe, wait karo, wapas aaungi! ✨*  \n"
        f"**🌸 Spread *{BOT_NAME}* with /invite jab on ho, *jaan*! 😜**  \n"
        f"**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    )
    for chat_id in list(chat_ids[TOKEN]):
        try:
            for clone_token in CLONE_TOKENS:
                clone_bot = telegram.Bot(clone_token)
                await clone_bot.send_message(chat_id, stop_msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Stop clones error for chat {chat_id}: {e}")
    await update.message.reply_text(
        f"**🛑 *Clones off, boss!* 😎🔥**  \n*Mai {BOT_NAME} hoon!*  \n**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**",
        parse_mode="Markdown")

async def start_clones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.message.from_user.id) not in OWNERS:
        await update.message.reply_text(f"**🔒 *Owners* hi clones start kar sakte, *baby*! 😈**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    update_original_status(True)
    start_msg = (
        f"**🌟 *Oye *baby*, *{BOT_NAME}* ka jadoo on! 😍✨**  \n*Mai {BOT_NAME} hoon, clones shuru, spread with /invite, baki bots bhool ja, *jaan*! 💪*  \n"
        f"**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    )
    for chat_id in list(chat_ids[TOKEN]):
        try:
            for clone_token in CLONE_TOKENS:
                clone_bot = telegram.Bot(clone_token)
                await clone_bot.send_message(chat_id, start_msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Start clones error for chat {chat_id}: {e}")
    await update.message.reply_text(
        f"**🟢 *Clones on, boss!* 😎🔥**  \n*Mai {BOT_NAME} hoon!*  \n**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**",
        parse_mode="Markdown")

async def list_bots(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.message.from_user.id) not in OWNERS or update.message.chat.type != "private":
        await update.message.reply_text(f"**🔒 *List* sirf owners ke liye, DM mein, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    
    response = f"**🌟 *{BOT_NAME} ke Babies* 😍✨**  \n*Mai {BOT_NAME} hoon!*\n"
    try:
        original_bot = await context.bot.get_me()
        response += f"**Original Bot: @{original_bot.username} ({BOT_NAME}) 🌹**  \n*Mai {BOT_NAME} hoon!*\n"
        original_chats = []
        for chat_id in chat_ids[TOKEN]:
            chat = await context.bot.get_chat(chat_id)
            if chat.type in ["group", "supergroup", "channel"]:
                chat_name = chat.title or chat.username or str(chat_id)
                original_chats.append(f"- **{chat.type.capitalize()}: {chat_name} 🎶**")
        response += "\n".join(original_chats) or "- **No groups/channels 😓**"
        response += "\n"
        
        for clone_token in CLONE_TOKENS:
            clone_bot = telegram.Bot(clone_token)
            clone_profile = await clone_bot.get_me()
            clone_name = CLONE_TOKENS[clone_token].get("name", BOT_NAME)
            response += f"**Clone Bot: @{clone_profile.username} ({clone_name}) 😈**  \n*Mai {clone_name} hoon!*\n"
            clone_chats = []
            for chat_id in chat_ids[clone_token]:
                chat = await clone_bot.get_chat(chat_id)
                if chat.type in ["group", "supergroup", "channel"]:
                    chat_name = chat.title or chat.username or str(chat_id)
                    clone_chats.append(f"- **{chat.type.capitalize()}: {chat_name} 💃**")
            response += "\n".join(clone_chats) or "- **No groups/channels 😕**"
            response += "\n"
    except Exception as e:
        logger.error(f"List bots error: {e}")
        response += f"**⚠️ Error fetching list: {str(e)} 😓**\n"
    
    response += f"**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**  \n*Mai {BOT_NAME} hoon!*"
    await update.message.reply_text(response, parse_mode="Markdown")

async def invite_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    random_name = user_random_names[user_id] or random.choice(RANDOM_NAMES)
    user_random_names[user_id] = random_name
    is_clone = context.bot.token in CLONE_TOKENS
    bot_name = CLONE_TOKENS.get(context.bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    owner_name = OWNERS.get(str(list(CLONE_TOKENS.values())[0]["creator_id"]) if is_clone and CLONE_TOKENS else "1866961136", "@Rohan2349")
    link = f"https://t.me/{ORIGINAL_BOT_USERNAME}?start=invited_by_{owner_name}" if is_clone else f"https://t.me/{ORIGINAL_BOT_USERNAME}"
    invite_msg = (
        f"**🌟 *Oye {random_name}, *{bot_name}* ko spread kar! 😍✨**  \n*Mai {bot_name} hoon, join here: {link} 💕*  \n"
        f"**🔥 Flirt with me, *baby*, aur baki bots bhool ja! 😜**  \n"
        f"**🌸 Chalo, spread karo, *jaan*, with {owner_name} ka support! 💪**  \n"
        f"**😘 With love from my creators {', '.join(OWNERS.values())}! 🌹**"
    )
    await update.message.reply_text(invite_msg, parse_mode="Markdown")

async def sync_bots(original_bot):
    if not check_original_status():
        return
    try:
        clones = {}
        try:
            with open("clones.json", "r") as f:
                clones = json.load(f)
        except FileNotFoundError:
            logger.info("clones.json not found, initializing empty clones")
            clones = {}
        except json.JSONDecodeError:
            logger.error("Invalid clones.json, initializing empty clones")
            clones = {}

        for bot_token in [TOKEN] + list(clones.keys()):
            try:
                bot = telegram.Bot(bot_token)
                await sync_bot_profile(bot, bot_token)
            except Exception as e:
                logger.error(f"Failed to sync bot with token {bot_token[:10]}...: {e}")
        
        try:
            with open("clones.json", "w") as f:
                json.dump(clones, f)
        except Exception as e:
            logger.error(f"Failed to write clones.json: {e}")
        
        logger.info(f"{BOT_NAME} synced!")
    except Exception as e:
        logger.error(f"Bot sync error: {e}")

async def remove_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    args = context.args
    if not args:
        await update.message.reply_text(f"**🗑️ *Message ID* daal, *baby*! 😜 Like: /remove <id> 💕**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    try:
        await context.bot.delete_message(chat_id, args[0])
        await update.message.reply_text(
            f"**🗑️ *Message gaya, *jaan*! 😎🔥**  \n*Mai {BOT_NAME} hoon!*  \n**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**",
            parse_mode="Markdown")
    except TelegramError as e:
        logger.error(f"Remove message error: {e}")
        await update.message.reply_text(
            f"**😓 *Message nahi gaya, *baby*! ⚠️**  \n*Mai {BOT_NAME} hoon, admin check kar, *hira*! 😜*  \n**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**",
            parse_mode="Markdown")

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.message.from_user.id) not in OWNERS:
        await update.message.reply_text(f"**🔒 *Owners* hi broadcast kar sakte, *baby*! 😈**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    args = context.args
    if not args:
        await update.message.reply_text(f"**📢 *Message* daal, *jaan*! 😍 Like: /broadcast Oye, kya haal? 💕**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    message = " ".join(args)
    chats = list(chat_ids[TOKEN])
    total_chats = len(chats)
    
    for i, target_chat in enumerate(chats):
        progress = i + 1
        time_left = (total_chats - progress) * 1
        bar = get_progress_bar(progress, total_chats)
        status_msg = await update.message.reply_text(
            f"**📢 *Broadcasting *{BOT_NAME} baby...* 🌟\n{bar}\n*Time left*: ~{time_left} sec 😎**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        try:
            await context.bot.send_message(
                target_chat,
                f"**📢 *{BOT_NAME} Broadcast*: {message} ✨**  \n*Mai {BOT_NAME} hoon!*  \n**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**",
                parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Broadcast error for chat {target_chat}: {e}")
        await asyncio.sleep(1)
        await status_msg.delete()
    
    await update.message.reply_text(
        f"**🎉 *Broadcast ho gaya {total_chats} mein, *hira*! 😎🔥**  \n*Mai {BOT_NAME} hoon!*  \n**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**",
        parse_mode="Markdown")

async def rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_name = CLONE_TOKENS.get(context.bot.token, {}).get("name", BOT_NAME)
    rules_text = (
        f"**📜 *{bot_name} ke Baby Rules* 😈🔥**  \n*Mai {bot_name} hoon!*\n"
        f"- **1. *Full admin* banao, warna *{bot_name}* ka jadoo nahi! 💪😜**\n"
        f"- **2. *Permissions* off? To *{bot_name}* silent hai, *baby*! ⚠️🌸**\n"
        f"- **3. *Clone* with /clone <token>, *{bot_name}* control mein! 😎✨**\n"
        f"- **4. No bot? Add kar, *full admin* do, *jaan*! 💕🎶**\n"
        f"- **5. *Owners* ko ban/mute? Off jab tak unban! 🚫😤**\n"
        f"- **6. *Naughty vibe* with songs—baki bhool ja! 😈💃**\n"
        f"- **7. Owners ko respect, baaki ko troll! 🌹😂**\n"
        f"- **8. Silent? *Permissions* ya *owner* check kar, *baby*! 😓🔧**\n"
        f"- **9. Spread with /invite to original *{BOT_NAME}*, clones sync! 🌟🎉**\n"
        f"**😘 With love from my creators {', '.join(OWNERS.values())}! 🌸**"
    )
    await update.message.reply_text(rules_text, parse_mode="Markdown")

async def chat_member_updated(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    new_member = update.my_chat_member.new_chat_member
    is_clone = context.bot.token in CLONE_TOKENS
    bot_name = CLONE_TOKENS.get(context.bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    
    if new_member.user.id == context.bot.id and new_member.status in ["member", "administrator"]:
        perms, remain_anonymous = await check_admin_permissions(context.bot, chat_id, context.bot.id)
        if not perms or remain_anonymous:
            await send_permission_warning(context.bot, chat_id, update.effective_user.id, is_clone, remain_anonymous)
        else:
            user_id = update.effective_user.id
            random_name = user_random_names[user_id] or random.choice(RANDOM_NAMES)
            user_random_names[user_id] = random_name
            welcome_msg = (
                f"**🎉 *Oye {random_name}, {update.effective_user.first_name}!* 😍✨**  \n*Mai {bot_name} hoon, teri naughty companion, by {', '.join(OWNERS.values())}! 💕*  \n"
                f"**🔥 Pyar, masti, songs—baki bhool ja, *baby*! 😈**  \n"
                f"**🌸 Spread with /invite, *jaan*! 🎶**"
            )
            await context.bot.send_message(chat_id, welcome_msg, parse_mode="Markdown")
            chat_ids[context.bot.token].add(chat_id)
    
    for owner_id in OWNERS:
        if new_member.user.id == int(owner_id) and new_member.status in ["restricted", "kicked"]:
            blocked_chats.add(chat_id)
            await send_owner_ban_warning(context.bot, chat_id, owner_id, is_clone)

def check_rate_limit(token):
    current_time = time.time()
    troll_usage[token] = [t for t in troll_usage[token] if current_time - t < 60]
    if len(troll_usage[token]) >= 3:
        return False
    troll_usage[token].append(current_time)
    return True

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_original_status():
        return
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    user_input = update.message.text or ""
    is_clone = context.bot.token in CLONE_TOKENS
    bot_name = CLONE_TOKENS.get(context.bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    
    if str(user_id) in OWNERS:
        owner_msg = (
            f"**🌟 Oye boss {OWNERS[str(user_id)]}! 😎✨**  \n"
            f"*Mai {bot_name} hoon, aapke bina adhoori, mere creator! 💕*  \n"
            f"**💪 Aapka hukum sar aankhon pe, *guru*! 🙏**  \n"
            f"**🌹 Dil se shukriya, {OWNERS[str(user_id)]}, for making me! 😘**"
        )
        await update.message.reply_text(owner_msg, parse_mode="Markdown")
        return
    
    if chat_id in blocked_chats:
        is_owner_ok, banned_owner = await check_owner_status(context.bot, chat_id)
        if not is_owner_ok:
            await send_owner_ban_warning(context.bot, chat_id, banned_owner, is_clone)
            return
    
    if update.message.chat.type in ["group", "supergroup", "channel"]:
        perms, remain_anonymous = await check_admin_permissions(context.bot, chat_id, context.bot.id)
        if not perms or remain_anonymous:
            await send_permission_warning(context.bot, chat_id, user_id, is_clone, remain_anonymous)
            return
    
    user_memory[chat_id].append(user_input)
    intent, mood = detect_intent_and_mood(user_input)
    user_mood[chat_id] = mood
    random_name = user_random_names[user_id] or random.choice(RANDOM_NAMES)
    user_random_names[user_id] = random_name
    
    reply = ""
    if update.message.chat.type == "private" and not is_clone:
        if intent == "no_bot":
            reply = (
                f"**😈 Arre {random_name}, koi bot nahi? ⚠️**  \n*Mai {bot_name} hoon, mujhe add kar, *full admin* bana, nahi toh troll karegi, *baby*! 😜*  \n"
                f"**🌸 Spread with /invite, *jaan*! 🎶**  \n"
                f"**😘 With love from my creators {', '.join(OWNERS.values())}! 🌹**"
            )
        else:
            gemini_reply = generate_gemini_reply(user_input, mood, user_id)
            if gemini_reply and "Oye" in gemini_reply:
                reply = gemini_reply.format(name=random_name, bot_name=bot_name)
            else:
                hf_reply = await generate_hf_reply(user_input, mood)
                reply = hf_reply.format(name=random_name) if hf_reply else (
                    f"**🌟 Oye {random_name}, {update.message.from_user.first_name}! 😎✨**  \n*Mai {bot_name} hoon, apna bot bana? 😜 /clone <token> use kar! 💪*  \n"
                    f"**🔥 Add kar, *full admin* bana, *baby*! Spread with /invite, baki bhool ja, *jaan*! 😈**  \n"
                    f"**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**"
                )
        
        sentiment = analyze_sentiment(user_input)
        if sentiment == "sad":
            reply += f"\n**🌸 Oye {random_name}, udaas ho? Mai {bot_name} hoon, na, *baby*! 😘**"
        elif sentiment == "happy":
            reply += f"\n**😍 Tujhe hasi suit karti hai, {random_name}! 💕**  \n*Mai {bot_name} hoon!*"
        
        if random.random() < 0.2:
            reply += f"\n**🌟 {random.choice(COMPLIMENTS).format(name=random_name, bot_name=bot_name)}**"
        
        if check_rate_limit(context.bot.token):
            await update.message.reply_text(reply, parse_mode="Markdown")
            
            if (intent in NAUGHTY_REPLIES or mood in ["flirty", "troll", "teasing", "romantic"]) and random.random() < 0.2:
                audio_file = await generate_free_tts(reply)
                if audio_file:
                    try:
                        with open(audio_file, "rb") as audio:
                            await update.message.reply_voice(voice=audio)
                    finally:
                        os.remove(audio_file)
            
            if random.random() < 0.5:
                image_url = await get_pexels_image_async(mood)
                if image_url:
                    await update.message.reply_photo(
                        photo=image_url,
                        caption=f"**{mood.capitalize()} vibe, {random_name}! 😍✨\nMai {bot_name} hoon, by {', '.join(OWNERS.values())}! 😈**",
                        parse_mode="Markdown")
        
        current_time = time.time()
        if current_time - user_last_gift[user_id] >= 24 * 3600:
            song = random.choice(SONG_POOL.get(mood, SONG_POOL["romantic"]))
            gift_msg = (
                f"**🎁 Oye {random_name}, tera daily gift! 😍🌟**  \n*Mai {bot_name} hoon, kahti: ‘Tu mera sab kuch hai, *baby*!’ 💕*  \n"
                f"**🎶 Sun {song}, spread with /invite, *jaan*! 🌸**  \n"
                f"**😘 With love from my creators {', '.join(OWNERS.values())}! ✨**"
            )
            await update.message.reply_text(gift_msg, parse_mode="Markdown")
            if random.random() < 0.5:
                image_url = await get_pexels_image_async(mood)
                if image_url:
                    await update.message.reply_photo(
                        photo=image_url,
                        caption=f"**🎁 Gift vibe, {random_name}! 😎🔥\nMai {bot_name} hoon! 😈**")
            user_last_gift[user_id] = current_time
        return
    
    if intent in NAUGHTY_REPLIES:
        reply = random.choice(NAUGHTY_REPLIES[intent]).format(name=random_name, bot_name=bot_name)
    elif intent == "troll" and ENABLE_TROLLING:
        reply = random.choice(TROLL_REPLIES).format(name=random_name, bot_name=bot_name)
    else:
        gemini_reply = generate_gemini_reply(user_input, mood, user_id)
        reply = gemini_reply.format(name=random_name, bot_name=bot_name) if gemini_reply and "Oye" in gemini_reply else (
            random.choice(TROLL_REPLIES).format(name=random_name, bot_name=bot_name)
        )
    
    if random.random() < 0.3:
        reply += f"\n**😘 Mujhe miss kiya, {random_name}? 💕**  \n*Mai {bot_name} hoon!*"
    if random.random() < 0.3:
        song = random.choice(SONG_POOL.get(mood, SONG_POOL["romantic"]))
        reply += f"\n**🎶 Sun {song}, {random_name}, mera pyar! 😍**  \n*Mai {bot_name} hoon!*"
    if random.random() < 0.2:
        reply += f"\n**🌟 {random.choice(COMPLIMENTS).format(name=random_name, bot_name=bot_name)}**"
    
    reply += f"\n**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    
    if check_rate_limit(context.bot.token):
        await update.message.reply_text(reply, parse_mode="Markdown")
        
        if (mood in ["flirty", "troll", "teasing", "romantic", "excited"] or intent in NAUGHTY_REPLIES) and random.random() < 0.2:
            audio_file = await generate_free_tts(reply)
            if audio_file:
                try:
                    with open(audio_file, "rb") as audio:
                        await update.message.reply_voice(voice=audio)
                finally:
                    os.remove(audio_file)
        
        if random.random() < 0.5:
            image_url = await get_pexels_image_async(mood)
            if image_url:
                await update.message.reply_photo(
                    photo=image_url,
                    caption=f"**{mood.capitalize()} vibe, {random_name}! 😍✨\nMai {bot_name} hoon, by {', '.join(OWNERS.values())}! 😈**",
                    parse_mode="Markdown")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_original_status():
        return
    chat_id = update.message.chat_id
    user = update.message.from_user
    args = context.args
    is_clone = context.bot.token in CLONE_TOKENS
    bot_name = CLONE_TOKENS.get(context.bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    
    if args and args[0].startswith("invited_by_"):
        owner_name = args[0].replace("invited_by_", "")
        random_name = user_random_names[user.id] or random.choice(RANDOM_NAMES)
        user_random_names[user.id] = random_name
        welcome_msg = (
            f"**🌟 Oye {random_name}, {user.first_name}! 😍✨**  \n*Mai {bot_name} hoon, tu {owner_name} ke invite se aaya, *baby*! 💕*  \n"
            f"**🔥 Mai {bot_name} hoon, teri naughty companion, by {', '.join(OWNERS.values())}! 😈**  \n"
            f"**🌸 Pyar, masti, songs—baki bhool ja! Spread with /invite, *jaan*! 🎶**"
        )
        await update.message.reply_text(welcome_msg, parse_mode="Markdown")
        return
    
    if update.message.chat.type in ["group", "supergroup", "channel"]:
        perms, remain_anonymous = await check_admin_permissions(context.bot, chat_id, context.bot.id)
        if not perms or remain_anonymous:
            await send_permission_warning(context.bot, chat_id, user.id, is_clone, remain_anonymous)
            return
    
    user_id = user.id
    random_name = user_random_names[user_id] or random.choice(RANDOM_NAMES)
    user_random_names[user_id] = random_name
    welcome_msg = (
        f"**🌟 Oye {random_name}, {user.first_name}! 😍✨**  \n*Mai {bot_name} hoon, teri naughty companion, by {', '.join(OWNERS.values())}! 💕*  \n"
        f"**🔥 Pyar, masti, songs—baki bhool ja, *baby*! 😈**  \n"
        f"**🌸 Spread with /invite, *jaan*! 🎉**"
    )
    await update.message.reply_text(welcome_msg, parse_mode="Markdown")

def analyze_sentiment(text):
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        return "happy" if polarity > 0 else "sad" if polarity < 0 else "neutral"
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return "neutral"

async def periodic_sync(app):
    while check_original_status():
        try:
            await sync_bots(app.bot)
        except Exception as e:
            logger.error(f"Periodic sync error: {e}")
        await asyncio.sleep(60)

def run_schedule():
    while True:
        if not check_original_status():
            time.sleep(1)
            continue
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Schedule error: {e}")
        time.sleep(1)

async def main():
    # Initialize clones.json if it doesn't exist
    try:
        with open("clones.json", "x") as f:
            json.dump({}, f)
    except FileExistsError:
        pass

    app = Application.builder().token(TOKEN).concurrent_updates(True).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clone", clone_bot, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("remove", remove_message))
    app.add_handler(CommandHandler("broadcast", broadcast, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("rules", rules))
    app.add_handler(CommandHandler("stop_clones", stop_clones, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("start_clones", start_clones, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("list", list_bots, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("invite", invite_link))
    app.add_handler(CommandHandler("setname", setname, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("setnameowner", setnameowner, filters=filters.ChatType.PRIVATE))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(ChatMemberHandler(chat_member_updated, ChatMemberHandler.MY_CHAT_MEMBER))
    
    commands = [
        ("start", f"Shuru kar {BOT_NAME} ke saath! 🌟😍"),
        ("clone", "Bot banao, /clone <token>! (DM) 😈🔥"),
        ("remove", "Message hatao, /remove <id>! 🗑️😜"),
        ("broadcast", "Owners ke liye, /broadcast! (DM) 📢🌹"),
        ("rules", "Rules dekh, /rules! 📜✨"),
        ("stop_clones", "Clones stop, /stop_clones! (Owners) 🔴😤"),
        ("start_clones", "Clones start, /start_clones! (Owners) 🟢😎"),
        ("list", "Bots list, /list! (Owners) 📋🌟"),
        ("invite", f"Spread {BOT_NAME} love! /invite 🌸🎉"),
        ("setname", "Set/Change clone name, /setname <name> <token>! 😎✨"),
        ("setnameowner", "Set original name, /setnameowner <name>! (Owners) 🌟✨")
    ]
    
    try:
        # Initialize the application
        await app.initialize()
        
        # Set bot description and commands
        await app.bot.set_my_description(f"**{BOT_NAME} - The Naughty Companion 😈✨**")
        await app.bot.set_my_commands(commands)
        update_original_status(True)
        chat_ids[TOKEN].add(-1)
        logger.info(f"{BOT_NAME} started!")
        
        # Start the application
        await app.start()
        
        # Schedule tasks
        sync_task = asyncio.create_task(periodic_sync(app))
        challenge_task = asyncio.create_task(periodic_challenge(app))
        schedule_promo(app)
        schedule_thread = threading.Thread(target=run_schedule, daemon=True)
        schedule_thread.start()
        
        # Run polling
        await app.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Bot stopped: {e}")
    finally:
        update_original_status(False)
        # Cancel tasks
        sync_task.cancel()
        challenge_task.cancel()
        try:
            await app.stop()
            await app.shutdown()
            logger.info("Application shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")