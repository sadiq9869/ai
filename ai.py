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

# Bot Config
TOKEN = "7520138270:AAHHDBRvhGZEXXwVJnSdXt-iLZuxrLzTAgo"
ORIGINAL_BOT_USERNAME = "@Ai_Pyaar_Bot"
BOT_NAME = "Rani"  # Default name
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
conversation_history = defaultdict(list)  # Per-user conversation history
troll_usage = defaultdict(list)  # For rate limiting trolling replies
ENABLE_TROLLING = True  # Toggle for trolling replies

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
    "**💪 Oye {name}, *{bot_name}* ka troll tujhe top pe!**  \n*Mai {bot_name} hoon, dar gaya, *baby*! 😜",
    # Added trolling with bhi and others
    "**😝 Oye {name}, yeh kya *bhi* bol raha, *madarchod* wala drama?**  \n*Mai {bot_name} hoon, thodi masti dikha, *baby*! 😎",
    "**😂 Arre {name}, *bhenchod* aur *bhi* kya karega?**  \n*Mai {bot_name} hoon, akal la, *jaan*! 🔥",
    "**😈 Haila {name}, *badwa* aur *bhi* kya plan hai?**  \n*Mai {bot_name} hoon, dimaag chalao, *hira*! 😜",
    "**😜 Oye {name}, *bhi* bolke *harami* wali harkat?**  \n*Mai {bot_name} hoon, style mein bol, *baby*! 💪",
    "**😂 Arre {name}, *sala* aur *bhi* kya troll karega?**  \n*Mai {bot_name} hoon, game badal, *jaan*! 😝",
    "**😤 {name}, *kutta* aur *bhi* kya plan?**  \n*Mai {bot_name} hoon, top pe aa, warna roast, *hira*! 😈",
    "**😝 Oye {name}, *chutmarani* aur *bhi* kya baat?**  \n*Mai {bot_name} hoon, pyar se bol, *baby*! 💃",
    "**😂 Arre {name}, *bhi* ke saath *chhapri* wala scene?**  \n*Mai {bot_name} hoon, swag dikha, *jaan*! 😎",
    "**😜 Oye {name}, *bhi* bol raha aur *baklol* wali baat?**  \n*Mai {bot_name} hoon, level up kar, *baby*! 😝"
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

# Hugging Face API
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
            return reply_cache[cache_key]
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    reply = result[0]["generated_text"].strip()
                    reply_cache[cache_key] = reply
                    return reply
        return None
    except Exception as e:
        print(f"HF API Error: {e}")
        return None

# Pexels API
async def get_pexels_image_async(mood):
    if mood in image_cache:
        return image_cache[mood]
    try:
        mood_keywords = {"flirty": "romance", "troll": "funny", "teasing": "playful", "romantic": "love", "funny": "humor", "excited": "party", "caring": "kindness"}
        query = mood_keywords.get(mood, "funny")
        url = f"https://api.pexels.com/v1/search?query={query}&per_page=1"
        headers = {"Authorization": PEXELS_API_KEY}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    photos = data.get("photos", [])
                    if photos:
                        image_cache[mood] = photos[0]["src"]["medium"]
                        return image_cache[mood]
        return None
    except Exception as e:
        print(f"Pexels API Error: {e}")
        return None

# Gemini API Response (with fallback)
def generate_gemini_reply(prompt, mood, user_id):
    try:
        # Attempt to use Gemini if available
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        conversation = conversation_history[user_id]
        conversation.append({"role": "user", "content": prompt})
        
        prompt_with_context = (
            "You are a caring, witty, desi girlfriend named {bot_name}. Respond in Hindi with warmth, humor, and a flirty tone. "
            "Maintain context from our conversation. Mood: {mood}. History: {history}. User said: {prompt}. "
            "Introduce yourself as '{bot_name}' in every reply, end with 'baby' or 'jaan' and use emojis. Keep it under 50 words."
        ).format(bot_name=BOT_NAME, mood=mood, history=str(conversation), prompt=prompt)
        
        response = model.generate_content(prompt_with_context)
        reply = response.text.strip()
        conversation.append({"role": "ai", "content": reply})
        return reply
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Fallback to a default reply
        return f"**😜 Oye, mai {BOT_NAME} hoon!** Teri baat sunke dil khush, par abhi thodi masti, *jaan*! 😈"

# Sync Bot Profile
async def sync_bot_profile(bot, bot_token):
    try:
        is_clone = bot_token in CLONE_TOKENS
        bot_name = CLONE_TOKENS.get(bot_token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
        bio = f"{bot_name} - The Naughty {'Clone' if is_clone else ''} of *Rani* 😈✨"
        await bot.set_my_description(bio)
        
        if not is_clone:
            original_profile = await bot.get_user_profile_photos(bot.id)
            if original_profile.photos:
                photo = original_profile.photos[0][-1]
                photo_file = await bot.get_file(photo.file_id)
                async with aiohttp.ClientSession() as session:
                    async with session.get(photo_file.file_path) as response:
                        photo_data = await response.read()
                await bot.set_profile_photo(photo_data)
        
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
        print(f"{bot_name} synced with token {bot_token}! 😍")
    except Exception as e:
        print(f"Sync error: {e}")

# Check Original Bot Status
def check_original_status():
    try:
        with open("status.json", "r") as f:
            return json.load(f).get("online", False)
    except FileNotFoundError:
        return True

# Update Original Bot Status
def update_original_status(status):
    BOT_STATUS["online"] = status
    with open("status.json", "w") as f:
        json.dump(BOT_STATUS, f)

# Enhanced Mood & Intent Detection
def detect_intent_and_mood(text):
    text = text.lower()
    if re.search(r"\bbaby mera baby\b", text):
        return "baby_mera_baby", "flirty"
    elif re.search(r"\b(gaand dona)\b", text):
        return "gaand_dona", "teasing"
    elif re.search(r"\b(chut dona)\b", text):
        return "chut_dona", "flirty"
    elif re.search(r"\b(baby love|baby jaan|meri jaan)\b", text):
        return "baby_romance", "romantic"
    elif re.search(r"\b(baby tease|baby kya)\b", text):
        return "baby_tease", "teasing"
    elif re.search(r"\b(baby fail|baby weak)\b", text):
        return "baby_savage", "troll"
    elif re.search(r"\b(joke|mazak|hansi)\b", text):
        return "joke", "funny"
    elif re.search(r"\b(love|pyar|dil|jaan)\b", text):
        return "flirt", "romantic"
    elif re.search(r"\b(sad|udaas|mood off|dukh)\b", text):
        return "comfort", "caring"
    elif re.search(r"\b(party|masti|dance|dhoom)\b", text):
        return "party", "excited"
    elif re.search(r"\b(bhai|bro|dost)\b", text):
        return "chat", "teasing"
    elif re.search(r"\b(nahi hai|no bot|koi bot nahi)\b", text):
        return "no_bot", "teasing"
    elif re.search(r"\b(bhi|madarchod|bhenchod|badwa|harami|sala|kutta|chutmarani|chhapri|baklol)\b", text):
        return "troll", "troll"  # Trigger trolling for bhi and others
    return "chat", "troll"

# Free Human-Like Voice
async def generate_free_tts(text):
    try:
        clean_text = text.replace("*", "").replace("_", "").replace("!", ".")
        clean_text = re.sub(r"\b(bhi|madarchod|bhenchod|badwa|harami|sala|kutta|chutmarani|chhapri|baklol|gaand|chut)\b", "masti", clean_text, flags=re.IGNORECASE)
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
        return None
    except Exception as e:
        print(f"Free TTS Error: {e}")
        return None

# Progress Bar
def get_progress_bar(progress, total, width=10):
    filled = int(width * progress / total)
    bar = "█" * filled + "▒" * (width - filled)
    percent = round(100 * progress / total, 1)
    return f"**📊 {bar} {percent}%**"

# Check Full Admin Permissions
async def check_admin_permissions(bot, chat_id, bot_id):
    try:
        member = await bot.get_chat_member(chat_id, bot_id)
        if member.status not in ["administrator", "creator"]:
            return False, False
        perms = member.can_change_info or False
        perms &= member.can_delete_messages or False
        perms &= member.can_ban_users or False
        perms &= member.can_invite_users or False
        perms &= member.can_pin_messages or False
        perms &= member.can_manage_stories or False
        perms &= member.can_manage_live_streams or False
        perms &= member.can_add_admins or False
        remain_anonymous = member.can_manage_chat and not member.is_anonymous
        return perms, remain_anonymous
    except Exception:
        return False, False

# Check Owner Status
async def check_owner_status(bot, chat_id):
    try:
        for owner_id in OWNERS:
            member = await bot.get_chat_member(chat_id, owner_id)
            if member.status in ["restricted", "kicked"]:
                return False, owner_id
        return True, None
    except Exception:
        return True, None

# Send Warning Message
async def send_permission_warning(bot, chat_id, user_id, is_clone=False, remain_anonymous=False):
    bot_name = CLONE_TOKENS.get(bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    if remain_anonymous:
        warning_msg = (
            "**😤 Oye, ‘Remain Anonymous’ on hai! 💥**  \n*Mai {bot_name} hoon, chhupna nahi, off kar, warna pyar nahi, *baby*!* 😈  \n"
            "**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**"
        )
    else:
        warning_msg = (
            "**😓 Oye, permission off hai! ⚠️**  \n*Mai {bot_name} hoon, full admin bana, nahi toh jadoo nahi, *jaan*! 😜  \n"
            "**🌟 Fix kar jaldi! With love from my creators {', '.join(OWNERS.values())}! 😘**"
        )
    try:
        await bot.send_message(chat_id, warning_msg.format(bot_name=bot_name), parse_mode="Markdown")
    except Exception as e:
        print(f"Warning message error: {e}")

# Send Owner Ban/Mute Warning
async def send_owner_ban_warning(bot, chat_id, owner_id, is_clone=False):
    bot_name = CLONE_TOKENS.get(bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    warning_msg = (
        "**😤 Oye, boss {OWNERS[owner_id]} ko ban/mute kiya? 🚫**  \n*Mai {bot_name} hoon, off hai jab tak unban/unmute nahi, *baby*! 😜  \n"
        "**🌟 Jaldi kar, *jaan*! With love from my creators {', '.join(OWNERS.values())}! 😘**"
    )
    try:
        await bot.send_message(chat_id, warning_msg.format(bot_name=bot_name), parse_mode="Markdown")
    except Exception as e:
        print(f"Owner ban warning error: {e}")

# Auto-Promotion Every 10 Minutes
async def send_promo_message(bot, is_clone=False):
    bot_name = CLONE_TOKENS.get(bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    owner_name = OWNERS.get(str(list(CLONE_TOKENS.values())[0]["creator_id"]) if is_clone and CLONE_TOKENS else "1866961136", "@Rohan2349")
    promo_msg = (
        "**🌟 Oye *baby*, *{bot_name}* ka jadoo try kar! 😍✨**  \n*Mai {bot_name} hoon, spread love with /invite! 💕*  \n"
        "**🔥 BGMI DDoS? DM {owner_name}! 😎**  \n"
        "**😓 Problem? DM @sadiq9869, *jaan*! 🌹**  \n"
        "**🚫 No calls to owners, DM karo, *hira*! 😜**  \n"
        "**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    ).format(bot_name=bot_name, owner_name=owner_name)
    for chat_id in list(chat_ids[bot.token]):
        chat = await bot.get_chat(chat_id)
        is_owner_ok, _ = await check_owner_status(bot, chat_id)
        if (await check_admin_permissions(bot, chat_id, bot.id)[0] or chat.type == "private") and is_owner_ok:
            try:
                await bot.send_message(chat_id, promo_msg, parse_mode="Markdown")
            except Exception:
                pass

def schedule_promo(app):
    async def run_promo():
        if check_original_status():
            await send_promo_message(app.bot)
            for clone_token in CLONE_TOKENS:
                await send_promo_message(telegram.Bot(clone_token), is_clone=True)
    schedule.every(10).minutes.do(lambda: asyncio.run_coroutine_threadsafe(run_promo(), asyncio.get_event_loop()))

# Automatic Challenge Sender
async def send_auto_challenge(bot, bot_token):
    bot_name = CLONE_TOKENS.get(bot_token, {}).get("name", BOT_NAME) if bot_token in CLONE_TOKENS else BOT_NAME
    challenge = random.choice(CHALLENGE_LEVELS)
    for chat_id in list(chat_ids[bot_token]):
        try:
            chat = await bot.get_chat(chat_id)
            if chat.type in ["group", "supergroup"] and await check_admin_permissions(bot, chat_id, bot.id)[0]:
                challenge_msg = (
                    "**🎉 *Oye *baby*, *{bot_name}* ka {challenge[name]} Challenge shuru! 😈🔥**  \n"
                    f"*Mai {bot_name} hoon, {challenge[desc]}*  \n"
                    "**💪 Complete kar, warna *{bot_name}* tujhe troll karegi, *jaan*! 🌸**  \n"
                    "**🌟 Spread with /invite, aur prove kar! 😎✨**  \n"
                    "**😘 With love from my creators {', '.join(OWNERS.values())}! 💕**"
                ).format(bot_name=bot_name, challenge=challenge)
                await bot.send_message(chat_id, challenge_msg, parse_mode="Markdown")
        except Exception:
            pass

def schedule_challenge(app):
    async def run_challenge():
        if check_original_status():
            await send_auto_challenge(app.bot, TOKEN)
            for clone_token in CLONE_TOKENS:
                await send_auto_challenge(telegram.Bot(clone_token), clone_token)
    schedule.every(15).minutes.do(lambda: asyncio.run_coroutine_threadsafe(run_challenge(), asyncio.get_event_loop()))

# Clone Bot Handler
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
    if clone_token == TOKEN:
        await update.message.reply_text(f"**😤 *Oye, original token nahi, naya daal, *hira*!* 😈 Warna *{BOT_NAME}* troll karegi! 🔥**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    if clone_token in CLONE_TOKENS:
        await update.message.reply_text(f"**😈 Yeh token clone hai, *baby*! 😜 Naya try kar, *jaan*! 🌹**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
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
    
    CLONE_TOKENS[clone_token] = {"original_token": TOKEN, "last_updated": time.time(), "creator_id": update.message.from_user.id, "name": BOT_NAME}
    with open("clones.json", "w") as f:
        json.dump(CLONE_TOKENS, f)
    
    clone_bot = telegram.Bot(clone_token)
    await sync_bot_profile(clone_bot, clone_token)
    chat_ids[clone_token].add(chat_id)
    
    await update.message.reply_text(
        f"**🎉 *Clone ban gaya, *baby*! 😎🔥**  \n*Token*: `{clone_token}` 💪  \nAdd kar, aur /invite se spread kar! 🌸  \n"
        f"**🌟 Apna style daalna ho to /setname <name> {clone_token} se naam change kar, *jaan*! 😈**  \n*Mai {BOT_NAME} hoon!*  \n"
        "**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**".format(clone_token=clone_token),
        parse_mode="Markdown")

# Set/Change Clone Bot Name
async def setname(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if update.message.chat.type != "private":
        await update.message.reply_text(f"**🔒 *Setname* sirf DM mein, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(f"**🔑 *Name* aur *token* daal, *jaan*! 😍 Like: /setname <name> <bot_token> 💕**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    new_name = args[0]
    bot_token = args[1]
    if bot_token == TOKEN:
        await update.message.reply_text(f"**😤 *Original bot ka naam nahi change, *hira*!* 😈 Try clone token, *baby*! 🔥**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    if bot_token not in CLONE_TOKENS:
        await update.message.reply_text(f"**😓 Yeh token clone nahi, *jaan*! 😜 Valid token daal, *baby*! 🌹**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    if str(update.message.from_user.id) != str(CLONE_TOKENS[bot_token]["creator_id"]):
        await update.message.reply_text(f"**🚫 *Sirf creator hi naam change kar sakta, *hira*!* 😈 DM owner ko, *baby*! 🔥**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
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
    
    CLONE_TOKENS[bot_token]["name"] = new_name
    with open("clones.json", "w") as f:
        json.dump(CLONE_TOKENS, f)
    
    clone_bot = telegram.Bot(bot_token)
    await sync_bot_profile(clone_bot, bot_token)
    
    await update.message.reply_text(
        f"**🎉 *Oye *baby*, {new_name} ban gaya! 😎🔥**  \n*Ab {new_name} ka jadoo spread kar, *jaan*! 🌸*  \n"
        f"**💪 /invite se duniya ko dikha, aur baki bhool ja! 😈**  \n*Mai {BOT_NAME} hoon!*  \n"
        "**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**",
        parse_mode="Markdown")

# Set/Change Original Bot Name (Owner Only)
async def setnameowner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_NAME  # Declare at the start
    chat_id = update.message.chat_id
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
    
    BOT_NAME = new_name  # Modify after global declaration
    with open("clones.json", "w") as f:
        json.dump(CLONE_TOKENS, f)
    
    original_bot = telegram.Bot(TOKEN)
    await sync_bot_profile(original_bot, TOKEN)
    
    await update.message.reply_text(
        f"**🎉 *Oye boss, {new_name} ban gaya! 😎🔥**  \n*Ab {new_name} ka jadoo chalao, *jaan*! 🌸*  \n"
        f"**💪 Clones bhi sync ho gaye, duniya ko dikha do! 😈**  \n*Mai {BOT_NAME} hoon!*  \n"
        "**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**",
        parse_mode="Markdown")

# Stop Clones
async def stop_clones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.message.from_user.id) not in OWNERS:
        await update.message.reply_text(f"**🔒 *Owners* hi clones stop kar sakte, *baby*! 😈**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    update_original_status(False)
    stop_msg = (
        "**🚫 *Oye *baby*, ek dikkat hai! 😓* **  \n*Mai {BOT_NAME} hoon, owner fix kar rahe, wait karo, wapas aaungi! ✨*  \n"
        "**🌸 Spread *{BOT_NAME}* with /invite jab on ho, *jaan*! 😜**  \n"
        "**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    )
    for chat_id in list(chat_ids[TOKEN]):
        try:
            chat = await context.bot.get_chat(chat_id)
            for clone_token in CLONE_TOKENS:
                clone_bot = telegram.Bot(clone_token)
                if await check_admin_permissions(clone_bot, chat_id, clone_bot.id)[0] or chat.type == "private":
                    await clone_bot.send_message(chat_id, stop_msg.format(BOT_NAME=BOT_NAME), parse_mode="Markdown")
        except Exception:
            pass
    await update.message.reply_text(f"**🛑 *Clones off, boss!* 😎🔥**  \n*Mai {BOT_NAME} hoon!*  \n**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**", parse_mode="Markdown")

# Start Clones
async def start_clones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.message.from_user.id) not in OWNERS:
        await update.message.reply_text(f"**🔒 *Owners* hi clones start kar sakte, *baby*! 😈**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    update_original_status(True)
    start_msg = (
        "**🌟 *Oye *baby*, *{BOT_NAME}* ka jadoo on! 😍✨**  \n*Mai {BOT_NAME} hoon, clones shuru, spread with /invite, baki bots bhool ja, *jaan*! 💪*  \n"
        "**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    )
    for chat_id in list(chat_ids[TOKEN]):
        try:
            chat = await context.bot.get_chat(chat_id)
            for clone_token in CLONE_TOKENS:
                clone_bot = telegram.Bot(clone_token)
                if await check_admin_permissions(clone_bot, chat_id, clone_bot.id)[0] or chat.type == "private":
                    await clone_bot.send_message(chat_id, start_msg.format(BOT_NAME=BOT_NAME), parse_mode="Markdown")
        except Exception:
            pass
    await update.message.reply_text(f"**🟢 *Clones on, boss!* 😎🔥**  \n*Mai {BOT_NAME} hoon!*  \n**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**", parse_mode="Markdown")

# List Bots
async def list_bots(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.message.from_user.id) not in OWNERS:
        await update.message.reply_text(f"**🔒 *List* sirf owners ke liye, *baby*! 😜**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    if update.message.chat.type != "private":
        await update.message.reply_text(f"**🔒 *List* DM mein, boss! 😎**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    
    response = f"**🌟 *{BOT_NAME} ke Babies* 😍✨**  \n*Mai {BOT_NAME} hoon!*\n".format(BOT_NAME=BOT_NAME)
    original_bot = await context.bot.get_me()
    response += f"**Original Bot: @{original_bot.username} ({BOT_NAME}) 🌹**  \n*Mai {BOT_NAME} hoon!*\n"
    original_chats = []
    for chat_id in chat_ids[TOKEN]:
        try:
            chat = await context.bot.get_chat(chat_id)
            if chat.type in ["group", "supergroup", "channel"]:
                chat_name = chat.title or chat.username or str(chat_id)
                original_chats.append(f"- **{chat.type.capitalize()}: {chat_name} 🎶**")
        except Exception:
            continue
    response += "\n".join(original_chats) or "- **No groups/channels 😓**"
    response += "\n"
    
    for clone_token in CLONE_TOKENS:
        try:
            clone_bot = telegram.Bot(clone_token)
            clone_profile = await clone_bot.get_me()
            clone_name = CLONE_TOKENS[clone_token].get("name", BOT_NAME)
            response += f"**Clone Bot: @{clone_profile.username} ({clone_name}) 😈**  \n*Mai {clone_name} hoon!*\n"
            clone_chats = []
            for chat_id in chat_ids[clone_token]:
                try:
                    chat = await clone_bot.get_chat(chat_id)
                    if chat.type in ["group", "supergroup", "channel"]:
                        chat_name = chat.title or chat.username or str(chat_id)
                        clone_chats.append(f"- **{chat.type.capitalize()}: {chat_name} 💃**")
                except Exception:
                    continue
            response += "\n".join(clone_chats) or "- **No groups/channels 😕**"
            response += "\n"
        except Exception as e:
            response += f"**Clone Bot: {clone_token} ⚠️**  \n*Mai {BOT_NAME} hoon!*\n- **Error: {str(e)} 😓**\n"
    
    response += f"**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**  \n*Mai {BOT_NAME} hoon!*"
    await update.message.reply_text(response, parse_mode="Markdown")

# Invite Command
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
        "**🔥 Flirt with me, *baby*, aur baki bots bhool ja! 😜**  \n"
        f"**🌸 Chalo, spread karo, *jaan*, with {owner_name} ka support! 💪**  \n"
        "**😘 With love from my creators {', '.join(OWNERS.values())}! 🌹**"
    ).format(random_name=random_name, bot_name=bot_name, link=link, owner_name=owner_name)
    await update.message.reply_text(invite_msg, parse_mode="Markdown")

# Auto-Update Bots
async def sync_bots(original_bot):
    if not check_original_status():
        return
    try:
        with open("clones.json", "r") as f:
            clones = json.load(f)
        for bot_token in [TOKEN] + list(clones.keys()):
            bot = telegram.Bot(bot_token)
            await sync_bot_profile(bot, bot_token)
        with open("clones.json", "w") as f:
            json.dump(clones, f)
        print(f"{BOT_NAME} synced! 😎")
    except Exception as e:
        print(f"Bot sync error: {e}")

# Remove Message
async def remove_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    args = context.args
    if not args:
        await update.message.reply_text(f"**🗑️ *Message ID* daal, *baby*! 😜 Like: /remove <id> 💕**  \n*Mai {BOT_NAME} hoon!*", parse_mode="Markdown")
        return
    message_id = args[0]
    try:
        await context.bot.delete_message(chat_id, message_id)
        await update.message.reply_text(f"**🗑️ *Message gaya, *jaan*! 😎🔥**  \n*Mai {BOT_NAME} hoon!*  \n**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**", parse_mode="Markdown")
    except TelegramError:
        await update.message.reply_text(f"**😓 *Message nahi gaya, *baby*! ⚠️**  \n*Mai {BOT_NAME} hoon, admin check kar, *hira*! 😜*  \n**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**", parse_mode="Markdown")

# Broadcast
async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
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
            await context.bot.send_message(target_chat, f"**📢 *{BOT_NAME} Broadcast*: {message} ✨**  \n*Mai {BOT_NAME} hoon!*  \n**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**", parse_mode="Markdown")
        except Exception:
            pass
        await asyncio.sleep(1)
        await status_msg.delete()
    
    await update.message.reply_text(f"**🎉 *Broadcast ho gaya {total_chats} mein, *hira*! 😎🔥**  \n*Mai {BOT_NAME} hoon!*  \n**🌟 With love from my creators {', '.join(OWNERS.values())}! 😘**", parse_mode="Markdown")

# Rules Command
async def rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_name = CLONE_TOKENS.get(context.bot.token, {}).get("name", BOT_NAME)
    rules_text = (
        f"**📜 *{bot_name} ke Baby Rules* 😈🔥**  \n*Mai {bot_name} hoon!*\n"
        "- **1. *Full admin* banao, warna *{bot_name}* ka jadoo nahi! 💪😜**\n"
        "- **2. *Permissions* off? To *{bot_name}* silent hai, *baby*! ⚠️🌸**\n"
        "- **3. *Clone* with /clone <token>, *{bot_name}* control mein! 😎✨**\n"
        "- **4. No bot? Add kar, *full admin* do, *jaan*! 💕🎶**\n"
        "- **5. *Owners* ko ban/mute? Off jab tak unban! 🚫😤**\n"
        "- **6. *Naughty vibe* with songs—baki bhool ja! 😈💃**\n"
        "- **7. Owners ko respect, baaki ko troll! 🌹😂**\n"
        "- **8. Silent? *Permissions* ya *owner* check kar, *baby*! 😓🔧**\n"
        "- **9. Spread with /invite to original *{BOT_NAME}*, clones sync! 🌟🎉**\n"
        "**😘 With love from my creators {', '.join(OWNERS.values())}! 🌸**"
    ).format(bot_name=bot_name, BOT_NAME=BOT_NAME)
    await update.message.reply_text(rules_text, parse_mode="Markdown")

# Chat Member Updated Handler
async def chat_member_updated(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    new_member = update.my_chat_member.new_chat_member
    old_member = update.my_chat_member.old_chat_member
    is_clone = context.bot.token in CLONE_TOKENS
    bot_name = CLONE_TOKENS.get(context.bot.token, {}).get("name", f"Clone {BOT_NAME}") if is_clone else BOT_NAME
    
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
                "**🔥 Pyar, masti, songs—baki bhool ja, *baby*! 😈**  \n"
                "**🌸 Spread with /invite, *jaan*! 🎶**"
            ).format(random_name=random_name, update=update, bot_name=bot_name)
            await context.bot.send_message(chat_id, welcome_msg, parse_mode="Markdown")
            chat_ids[context.bot.token].add(chat_id)
    
    for owner_id in OWNERS:
        if new_member.user.id == int(owner_id) and new_member.status in ["restricted", "kicked"]:
            blocked_chats.add(chat_id)
            await send_owner_ban_warning(context.bot, chat_id, owner_id, is_clone)
        elif old_member.user.id == int(owner_id) and old_member.status in ["restricted", "kicked"] and new_member.status not in ["restricted", "kicked"]:
            blocked_chats.discard(chat_id)

# Rate Limit Check
def check_rate_limit(token):
    current_time = time.time()
    troll_usage[token] = [t for t in troll_usage[token] if current_time - t < 60]
    if len(troll_usage[token]) >= 3:  # Max 3 trolling replies per minute
        return False
    troll_usage[token].append(current_time)
    return True

# Message Handler with Fallback
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_original_status():
        return
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    user_input = update.message.text or ""
    is_clone = context.bot.token in CLONE_TOKENS
    bot_name = CLONE_TOKENS.get(context.bot.token, {}).get("name", BOT_NAME) if is_clone else BOT_NAME
    
    # Owner treatment: respect, praise, gratitude
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
    
    if update.message.chat.type == "private" and not is_clone:
        if intent == "no_bot":
            reply = (
                f"**😈 Arre {random_name}, koi bot nahi? ⚠️**  \n*Mai {bot_name} hoon, mujhe add kar, *full admin* bana, nahi toh troll karegi, *baby*! 😜*  \n"
                f"**🌸 Spread with /invite, *jaan*! 🎶**  \n"
                f"**😘 With love from my creators {', '.join(OWNERS.values())}! 🌹**"
            ).format(random_name=random_name, bot_name=bot_name)
        else:
            # Try Gemini, fall back to Hugging Face or default
            gemini_reply = generate_gemini_reply(user_input, mood, user_id)
            if gemini_reply and "Oye" in gemini_reply:
                reply = gemini_reply.format(name=random_name, bot_name=bot_name)
            else:
                hf_reply = await generate_hf_reply(user_input, mood)
                if hf_reply:
                    reply = hf_reply.format(name=random_name)
                else:
                    reply = (
                        f"**🌟 Oye {random_name}, {update.message.from_user.first_name}! 😎✨**  \n*Mai {bot_name} hoon, apna bot bana? 😜 /clone <token> use kar! 💪*  \n"
                        f"**🔥 Add kar, *full admin* bana, *baby*! Spread with /invite, baki bhool ja, *jaan*! 😈**  \n"
                        f"**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**"
                    ).format(random_name=random_name, update=update, bot_name=bot_name)
        
        # Sentiment Analysis
        sentiment = analyze_sentiment(user_input)
        if sentiment == "sad":
            reply += f"\n**🌸 Oye {random_name}, udaas ho? Mai {bot_name} hoon, na, *baby*! 😘**"
        elif sentiment == "happy":
            reply += f"\n**😍 Tujhe hasi suit karti hai, {random_name}! 💕**  \n*Mai {bot_name} hoon!*"
        
        if random.random() < 0.2:
            reply += f"\n**🌟 {random.choice(COMPLIMENTS).format(name=random_name, bot_name=bot_name)}**"
        
        if random.random() < 0.5 and user_memory[chat_id] and random.random() < 0.5:
            last_msg = user_memory[chat_id][-1]
            reply += f"\n**😜 Tune bola tha '{last_msg}', ab kya, {random_name}? 🔥**  \n*Mai {bot_name} hoon!*"
        
        if check_rate_limit(context.bot.token):
            await update.message.reply_text(reply, parse_mode="Markdown")
            
            if (intent in NAUGHTY_REPLIES or mood in ["flirty", "troll", "teasing", "romantic"]) and random.random() < 0.2:
                audio_file = await generate_free_tts(reply)
                if audio_file:
                    with open(audio_file, "rb") as audio:
                        await update.message.reply_voice(voice=audio)
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
            ).format(random_name=random_name, bot_name=bot_name)
            await update.message.reply_text(gift_msg, parse_mode="Markdown")
            if random.random() < 0.5:
                image_url = await get_pexels_image_async(mood)
                if image_url:
                    await update.message.reply_photo(
                        photo=image_url,
                        caption=f"**🎁 Gift vibe, {random_name}! 😎🔥\nMai {bot_name} hoon! 😈**")
            user_last_gift[user_id] = current_time
        return
    
    # Group/supergroup/channel handling
    if intent in NAUGHTY_REPLIES:
        reply = random.choice(NAUGHTY_REPLIES[intent]).format(name=random_name, bot_name=bot_name)
    elif intent == "troll" and ENABLE_TROLLING:
        reply = random.choice(TROLL_REPLIES).format(name=random_name, bot_name=bot_name)
    else:
        gemini_reply = generate_gemini_reply(user_input, mood, user_id)
        if gemini_reply and "Oye" in gemini_reply:
            reply = gemini_reply.format(name=random_name, bot_name=bot_name)
        else:
            reply = random.choice(TROLL_REPLIES).format(name=random_name, bot_name=bot_name)
    
    if random.random() < 0.3:
        reply += f"\n**😘 Mujhe miss kiya, {random_name}? 💕**  \n*Mai {bot_name} hoon!*"
    if random.random() < 0.3:
        song = random.choice(SONG_POOL.get(mood, SONG_POOL["romantic"]))
        reply += f"\n**🎶 Sun {song}, {random_name}, mera pyar! 😍**  \n*Mai {bot_name} hoon!*"
    if random.random() < 0.2:
        reply += f"\n**🌟 {random.choice(COMPLIMENTS).format(name=random_name, bot_name=bot_name)}**"
    
    if random.random() < 0.5 and user_memory[chat_id] and random.random() < 0.5:
        last_msg = user_memory[chat_id][-1]
        reply += f"\n**😜 Tune bola tha '{last_msg}', ab kya, {random_name}? 🔥**  \n*Mai {bot_name} hoon!*"
    
    reply += f"\n**🌸 With love from my creators {', '.join(OWNERS.values())}! 😘**"
    
    if check_rate_limit(context.bot.token):
        await update.message.reply_text(reply, parse_mode="Markdown")
        
        if (mood in ["flirty", "troll", "teasing", "romantic", "excited"] or intent in NAUGHTY_REPLIES) and random.random() < 0.2:
            audio_file = await generate_free_tts(reply)
            if audio_file:
                with open(audio_file, "rb") as audio:
                    await update.message.reply_voice(voice=audio)
                os.remove(audio_file)
        
        if random.random() < 0.5:
            image_url = await get_pexels_image_async(mood)
            if image_url:
                await update.message.reply_photo(
                    photo=image_url,
                    caption=f"**{mood.capitalize()} vibe, {random_name}! 😍✨\nMai {bot_name} hoon, by {', '.join(OWNERS.values())}! 😈**",
                    parse_mode="Markdown")

# Start Command
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
        ).format(random_name=random_name, user=user, owner_name=owner_name, bot_name=bot_name)
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
    if str(user.id) in OWNERS:
        welcome_msg = (
            f"**🌟 Oye boss {OWNERS[str(user.id)]}! 😎✨**  \n*Mai {bot_name} hoon, aapke bina adhoori, {random_name}! 😍*  \n"
            f"**💪 Plan bata, *jaan*! 🙏**  \n"
            f"**🌹 With love from my creators {', '.join(OWNERS.values())}! 😘**"
        ).format(random_name=random_name, OWNERS=OWNERS, bot_name=bot_name)
    else:
        welcome_msg = (
            f"**🌟 Oye {random_name}, {user.first_name}! 😍✨**  \n*Mai {bot_name} hoon, teri naughty companion, by {', '.join(OWNERS.values())}! 💕*  \n"
            f"**🔥 Pyar, masti, songs—baki bhool ja, *baby*! 😈**  \n"
            f"**🌸 Spread with /invite, *jaan*! 🎉**"
        ).format(random_name=random_name, user=user, bot_name=bot_name)
    await update.message.reply_text(welcome_msg, parse_mode="Markdown")

# Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "happy"
    elif analysis.sentiment.polarity < 0:
        return "sad"
    return "neutral"

# Main Function
async def main():
    app = Application.builder().token(TOKEN).concurrent_updates(True).build()
    
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
    
    async def periodic_sync():
        while True:
            if check_original_status():
                await sync_bots(app.bot)
            await asyncio.sleep(60)
    
    async def periodic_challenge():
        while True:
            if check_original_status():
                await send_auto_challenge(app.bot, TOKEN)
                for clone_token in CLONE_TOKENS:
                    await send_auto_challenge(telegram.Bot(clone_token), clone_token)
            await asyncio.sleep(900)
    
    app.create_task(periodic_sync())
    app.create_task(periodic_challenge())
    schedule_promo(app)
    threading.Thread(target=run_schedule, daemon=True).start()
    
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
    
    await app.bot.set_my_description(f"**{BOT_NAME} - The Naughty Companion 😈✨**")
    await app.bot.set_my_commands(commands)
    update_original_status(True)
    chat_ids[TOKEN].add(-1)
    print(f"**{BOT_NAME} shuru ho gaya! 🌟😎**")
    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        print(f"**Bot stopped: {e} 😓**")
        update_original_status(False)
    finally:
        update_original_status(False)

def run_schedule():
    while check_original_status():
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())