#!/usr/bin/env python3
"""
Simple Telegram bot that proxies messages to an OpenRouter / generic LLM API
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import psycopg2
import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Files
CONFIG_FILE = Path("config.json")
ENV_FILE = Path(".env")

# Conversation states for first-time setup
SET_ENDPOINT, SET_MODEL, SET_APIKEY = range(3)

# Conversation states for settings
(
    SELECTING_SETTING,
    UPDATING_SETTING,
) = range(3, 5)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Defaults
DEFAULT_CONFIG = {
    "endpoint": "",
    "model": "",
    "system_prompt": "You are a helpful assistant.",
    "context_messages_count": 10,
}


# --- Config helpers ---
def read_config() -> dict:
    if not CONFIG_FILE.exists():
        write_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        # ensure all keys are present
        cfg = json.loads(CONFIG_FILE.read_text())
        for key, value in DEFAULT_CONFIG.items():
            if key not in cfg:
                cfg[key] = value
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()


def write_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


# --- DB helpers ---
def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def init_db() -> None:
    retries = 5
    while retries > 0:
        try:
            conn = get_db_conn()
            c = conn.cursor()
            c.execute(
                '''
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    role TEXT,
                    content TEXT,
                    created_at BIGINT
                )
                '''
            )
            conn.commit()
            c.close()
            conn.close()
            logger.info("Database initialized.")
            return
        except psycopg2.OperationalError as e:
            logger.warning("DB not ready, retrying... (%s)", e)
            retries -= 1
            time.sleep(5)
    raise RuntimeError("Could not connect to database.")


def add_message(chat_id: int, role: str, content: str) -> None:
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
        (chat_id, role, content, int(time.time())),
    )
    conn.commit()
    c.close()
    conn.close()


def get_recent_messages(chat_id: int, limit: int = 10) -> List[Tuple[str, str]]:
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s",
        (chat_id, limit),
    )
    rows = c.fetchall()
    c.close()
    conn.close()
    rows.reverse()
    return rows

def get_first_message(chat_id: int) -> Optional[Tuple[str, str]]:
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY created_at ASC LIMIT 1",
        (chat_id,),
    )
    row = c.fetchone()
    c.close()
    conn.close()
    return row


# --- Setup conversation (/start) ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = read_config()
    api_key = os.getenv("API_KEY")
    if cfg.get("endpoint") and cfg.get("model") and api_key:
        await update.message.reply_text(
            "Bot настроен. Отправляйте сообщения — они будут пересланы к модели. Используйте /settings для изменения конфигурации."
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Добро пожаловать. Пожалуйста, пришлите URL endpoint'а API (например https://openrouter.ai/api/v1/chat/completions)."
    )
    return SET_ENDPOINT


async def set_endpoint(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["endpoint"] = update.message.text.strip()
    await update.message.reply_text("Укажите название модели (пример: deepseek/deepseek-r1-0528:free):")
    return SET_MODEL


async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["model"] = update.message.text.strip()
    await update.message.reply_text(
        "Теперь пришлите API-ключ. Если хотите использовать переменную окружения API_KEY, пришлите SKIP."
    )
    return SET_APIKEY


async def set_apikey(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.lower() != "skip":
        with open(ENV_FILE, "a") as f:
            f.write(f"\nAPI_KEY={text}\n")
        os.environ["API_KEY"] = text

    cfg = read_config()
    cfg["endpoint"] = context.user_data.get("endpoint", cfg.get("endpoint"))
    cfg["model"] = context.user_data.get("model", cfg.get("model"))
    write_config(cfg)

    await update.message.reply_text("Конфигурация сохранена. Отправьте сообщение, чтобы начать чат с моделью. Используйте /settings для просмотра/изменения настроек.")
    return ConversationHandler.END


async def cancel_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Настройка отменена.")
    return ConversationHandler.END


# --- Settings conversation ---
async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the settings conversation."""
    cfg = read_config()
    text = (
        f"Текущая конфигурация:\n"
        f"Endpoint: `{cfg.get('endpoint')}`\n"
        f"Model: `{cfg.get('model')}`\n"
        f"System prompt: `{cfg.get('system_prompt')}`\n"
        f"Context messages: `{cfg.get('context_messages_count')}`\n\n"
        "Какую настройку вы хотите изменить?"
    )
    keyboard = [
        [InlineKeyboardButton("API Key", callback_data="settings_apikey")],
        [InlineKeyboardButton("System prompt", callback_data="settings_system_prompt")],
        [InlineKeyboardButton("Endpoint", callback_data="settings_endpoint")],
        [InlineKeyboardButton("Model", callback_data="settings_model")],
        [InlineKeyboardButton("Context messages", callback_data="settings_context_messages_count")],
        [InlineKeyboardButton("Отмена", callback_data="settings_cancel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="MarkdownV2")
    return SELECTING_SETTING


async def settings_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handle settings button clicks."""
    query = update.callback_query
    await query.answer()
    choice = query.data

    if choice == "settings_cancel":
        await query.edit_message_text("Изменение настроек отменено.")
        return ConversationHandler.END

    context.user_data["setting_to_change"] = choice
    
    if choice == "settings_apikey":
        prompt = "Введите новый API Key:"
    elif choice == "settings_system_prompt":
        prompt = "Введите новый system prompt:"
    elif choice == "settings_endpoint":
        prompt = "Введите новый URL endpoint'а:"
    elif choice == "settings_model":
        prompt = "Введите новое название модели:"
    elif choice == "settings_context_messages_count":
        prompt = "Введите новое количество сообщений в контексте:"
    else:
        await query.edit_message_text("Неизвестная опция.")
        return ConversationHandler.END

    await query.edit_message_text(text=prompt)
    return UPDATING_SETTING


async def set_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Saves the new setting value."""
    new_value = update.message.text.strip()
    setting_to_change = context.user_data.get("setting_to_change")

    if not setting_to_change:
        await update.message.reply_text("Произошла ошибка. Попробуйте снова /settings.")
        return ConversationHandler.END

    cfg = read_config()
    if setting_to_change == "settings_apikey":
        with open(ENV_FILE, "a") as f:
            f.write(f"\nAPI_KEY={new_value}\n")
        os.environ["API_KEY"] = new_value
        await update.message.reply_text("API key сохранён.")
    elif setting_to_change == "settings_system_prompt":
        cfg["system_prompt"] = new_value
        write_config(cfg)
        await update.message.reply_text("System prompt обновлён.")
    elif setting_to_change == "settings_endpoint":
        cfg["endpoint"] = new_value
        write_config(cfg)
        await update.message.reply_text("Endpoint обновлён.")
    elif setting_to_change == "settings_model":
        cfg["model"] = new_value
        write_config(cfg)
        await update.message.reply_text("Модель обновлена.")
    elif setting_to_change == "settings_context_messages_count":
        try:
            cfg["context_messages_count"] = int(new_value)
            write_config(cfg)
            await update.message.reply_text("Количество сообщений в контексте обновлено.")
        except ValueError:
            await update.message.reply_text("Ошибка: введите число.")

    context.user_data.clear()
    return ConversationHandler.END


async def cancel_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the settings conversation."""
    await update.message.reply_text("Изменение настроек отменено.")
    context.user_data.clear()
    return ConversationHandler.END


# --- Helper to extract assistant text ---
def extract_text_from_response(resp_json: dict) -> str:
    try:
        if "output" in resp_json:
            out = resp_json["output"]
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if "content" in first and isinstance(first["content"], list):
                    for item in first["content"]:
                        if item.get("type") in ("output_text", "message"):
                            return item.get("text") or item.get("content") or str(item)
                    return "\n".join([c.get("text", str(c)) for c in first.get("content", [])])
    except Exception:
        pass

    try:
        if "choices" in resp_json and isinstance(resp_json["choices"], list) and len(resp_json["choices"]) > 0:
            ch = resp_json["choices"][0]
            if "message" in ch and "content" in ch["message"]:
                if isinstance(ch["message"]["content"], dict):
                    return ch["message"]["content"].get("text", json.dumps(ch["message"]["content"]))
                return ch["message"]["content"]
            if "text" in ch:
                return ch["text"]
    except Exception:
        pass

    return json.dumps(resp_json)[:2000]


# --- Main message handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages: send typing actions while waiting for the LLM API."""
    # ignore commands
    if update.message and update.message.text and update.message.text.startswith("/"):
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    add_message(chat_id, "user", user_text)

    cfg = read_config()
    endpoint = cfg.get("endpoint")
    model = cfg.get("model")
    system_prompt = cfg.get("system_prompt") or DEFAULT_CONFIG["system_prompt"]
    api_key = os.getenv("API_KEY")
    context_messages_count = cfg.get("context_messages_count", 10)

    if not endpoint or not model or not api_key:
        await update.message.reply_text(
            "Конфигурация неполная. Используйте /start для первоначальной настройки или /settings для изменения."
        )
        return

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add first message for long-term context
    first_message = get_first_message(chat_id)
    
    # Get recent messages
    history = get_recent_messages(chat_id, limit=context_messages_count)

    if first_message and first_message not in history:
        messages.append({"role": first_message[0], "content": first_message[1]})

    for role, content in history:
        messages.append({"role": role, "content": content})

    # Ensure the current message is included if history was full
    if not any(m["role"] == "user" and m["content"] == user_text for m in messages):
         messages.append({"role": "user", "content": user_text})


    payload = {"model": model, "messages": messages}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Worker that periodically sends "typing" chat action until cancelled
    async def typing_worker():
        try:
            while True:
                # send_chat_action supports ChatAction constants
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            return

    # Blocking request function (returns status, text, headers)
    def do_request():
        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            return r.status_code, r.text, r.headers
        except Exception as e:
            return None, str(e), {}

    # start typing indicator
    typing_task = asyncio.create_task(typing_worker())
    try:
        status, text, resp_headers = await asyncio.to_thread(do_request)
    finally:
        # stop typing indicator
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    if status is None:
        await update.message.reply_text(f"Ошибка запроса к API: {text}")
        logger.error("API request error: %s", text)
        return

    # detect if response is HTML (indicates wrong endpoint or invalid key)
    text_start = (text or "")[:200].lstrip()
    content_type = (resp_headers or {}).get("Content-Type", "")
    if text_start.startswith("<!DOCTYPE") or text_start.startswith("<html") or "text/html" in content_type:
        await update.message.reply_text(
            "Сервер вернул HTML-страницу вместо JSON. Проверьте endpoint и API-ключ."
        )
        logger.error("API returned HTML (first 200 chars): %s", text_start)
        return

    if status != 200:
        await update.message.reply_text(f"API returned status {status}: {text[:1000]}")
        logger.error("API returned status %s: %s", status, text[:1000])
        return

    try:
        resp_json = json.loads(text)
    except Exception:
        resp_json = {"raw": text}

    assistant_text = extract_text_from_response(resp_json)
    add_message(chat_id, "assistant", assistant_text)

    await update.message.reply_text(assistant_text)


# --- Entry point ---
def main():
    init_db()
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set in environment (.env or export)")

    app = ApplicationBuilder().token(token).build()

    # Conversation handler for initial setup
    setup_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SET_ENDPOINT: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_endpoint)],
            SET_MODEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_model)],
            SET_APIKEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_apikey)],
        },
        fallbacks=[CommandHandler("cancel", cancel_setup)],
    )

    # Conversation handler for settings
    settings_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("settings", settings)],
        states={
            SELECTING_SETTING: [
                CallbackQueryHandler(settings_callback_handler, pattern="^settings_")
            ],
            UPDATING_SETTING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_setting_value)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_settings)],
    )

    app.add_handler(setup_conv_handler)
    app.add_handler(settings_conv_handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting bot")
    return app


if __name__ == "__main__":
    try:
        application = main()
        application.run_polling(close_loop=False)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
