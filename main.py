#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import time
import base64
import io
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse

import psycopg2
import requests
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    Document,
    PhotoSize,
)
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from PIL import Image
import PyPDF2

# --- Setup ---
ENV_FILE = Path(".env")
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


def setup_encryption() -> Fernet:
    """
    Loads or generates and saves the Fernet encryption key.
    Returns an initialized Fernet instance.
    """
    env_path = Path(".env")
    load_dotenv(dotenv_path=env_path)
    encryption_key = os.getenv("ENCRYPTION_KEY")

    def is_valid_key(k):
        if not k:
            return False
        try:
            Fernet(k.encode())
            return True
        except (ValueError, TypeError):
            return False

    if is_valid_key(encryption_key):
        logger.info("ENCRYPTION_KEY loaded successfully.")
        return Fernet(encryption_key.encode())

    # If key is missing or invalid, generate a new one
    if encryption_key:
        logger.warning("Existing ENCRYPTION_KEY is invalid. Generating a new one.")
    else:
        logger.info("ENCRYPTION_KEY not found. Generating a new one.")

    new_key = Fernet.generate_key().decode()

    # Update .env file
    if env_path.exists():
        lines = env_path.read_text().splitlines()
        key_found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("ENCRYPTION_KEY="):
                lines[i] = f"ENCRYPTION_KEY={new_key}"
                key_found = True
                break
        if not key_found:
            lines.append(f"ENCRYPTION_KEY={new_key}")
        env_path.write_text("\n".join(lines) + "\n")
    else:
        env_path.write_text(f"ENCRYPTION_KEY={new_key}\n")

    logger.info("A new ENCRYPTION_KEY has been generated and saved to .env")

    # Update environment for the current process
    os.environ["ENCRYPTION_KEY"] = new_key

    return Fernet(new_key.encode())


# Encryption key
fernet = setup_encryption()

# Conversation states
(
    ASK_ENDPOINT,
    ASK_MODEL,
    ASK_APIKEY,
    SELECTING_SETTING,
    UPDATING_SETTING,
) = range(5)


# --- Encryption helpers ---
def encrypt_data(data: str) -> str:
    """Encrypts a string."""
    return fernet.encrypt(data.encode()).decode()


def decrypt_data(encrypted_data: str) -> Optional[str]:
    """Decrypts a string, returns None on failure."""
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except (InvalidToken, TypeError):
        logger.error("Failed to decrypt API key. It might be invalid or corrupted.")
        return None


# --- Database helpers ---
def get_db_conn():
    """Get PostgreSQL connection."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def init_db() -> None:
    """Initialize PostgreSQL database with a users table."""
    retries = 5
    while retries > 0:
        try:
            with get_db_conn() as conn:
                with conn.cursor() as c:
                    # Users table for multi-user support
                    c.execute(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                            chat_id BIGINT PRIMARY KEY,
                            api_key TEXT,
                            endpoint TEXT,
                            model TEXT,
                            reasoning_model TEXT,
                            router_model TEXT,
                            system_prompt TEXT,
                            context_messages_count INTEGER,
                            auto_thinking BOOLEAN,
                            search_enabled BOOLEAN,
                            created_at BIGINT
                        )
                        """
                    )
                    # Messages table
                    c.execute(
                        """
                        CREATE TABLE IF NOT EXISTS messages (
                            id SERIAL PRIMARY KEY,
                            chat_id BIGINT,
                            role TEXT,
                            content TEXT,
                            created_at BIGINT,
                            FOREIGN KEY (chat_id) REFERENCES users(chat_id)
                        )
                        """
                    )
                    # File contexts table
                    c.execute(
                        """
                        CREATE TABLE IF NOT EXISTS file_contexts (
                            id SERIAL PRIMARY KEY,
                            chat_id BIGINT,
                            filename TEXT,
                            file_type TEXT,
                            content TEXT,
                            file_path TEXT,
                            created_at BIGINT,
                            FOREIGN KEY (chat_id) REFERENCES users(chat_id)
                        )
                        """
                    )
            logger.info("PostgreSQL database initialized successfully.")
            return
        except psycopg2.OperationalError as e:
            logger.warning("DB not ready, retrying... (%s)", e)
            retries -= 1
            time.sleep(5)
    raise RuntimeError("Could not connect to PostgreSQL database.")


def get_user_settings(chat_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a user's settings from the database."""
    with get_db_conn() as conn:
        with conn.cursor() as c:
            c.execute("SELECT * FROM users WHERE chat_id = %s", (chat_id,))
            user_data = c.fetchone()
            if user_data:
                columns = [desc[0] for desc in c.description]
                settings = dict(zip(columns, user_data))
                # Decrypt API key
                if settings.get("api_key"):
                    decrypted_key = decrypt_data(settings["api_key"])
                    if not decrypted_key:
                        # Handle decryption failure - maybe notify user
                        logger.error(f"Could not decrypt API key for chat_id {chat_id}")
                        settings["api_key"] = None
                    else:
                        settings["api_key"] = decrypted_key
                return settings
    return None


def create_or_update_user(chat_id: int, settings: Dict[str, Any]) -> None:
    """Create a new user or update existing user's settings."""
    # Encrypt API key before storing
    if "api_key" in settings and settings["api_key"]:
        settings["api_key"] = encrypt_data(settings["api_key"])

    columns = list(settings.keys())
    values = [settings[key] for key in columns]

    # Ensure chat_id is included for the ON CONFLICT clause
    if "chat_id" not in columns:
        columns.append("chat_id")
        values.append(chat_id)

    # Prepare SQL for INSERT and UPDATE
    insert_cols = ", ".join(columns)
    insert_vals = ", ".join(["%s"] * len(values))
    update_set = ", ".join(
        [f"{col} = EXCLUDED.{col}" for col in columns if col != "chat_id"]
    )

    sql = f"""
        INSERT INTO users ({insert_cols}, created_at)
        VALUES ({insert_vals}, %s)
        ON CONFLICT (chat_id) DO UPDATE SET {update_set}
    """

    with get_db_conn() as conn:
        with conn.cursor() as c:
            c.execute(sql, values + [int(time.time())])


def get_default_settings() -> Dict[str, Any]:
    """Returns a dictionary of default settings."""
    return {
        "api_key": None,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "mistralai/mistral-7b-instruct:free",
        "reasoning_model": "deepseek/deepseek-r1-distill-llama-70b",
        "router_model": "anthropic/claude-3.5-sonnet:beta",
        "system_prompt": "You are a helpful assistant.",
        "context_messages_count": 10,
        "auto_thinking": True,
        "search_enabled": True,
    }


# Other DB functions (add_message, get_recent_messages, etc.) remain largely the same
# but should ensure they handle DB connections properly.


def add_message(chat_id: int, role: str, content: str) -> None:
    """Add message to PostgreSQL"""
    with get_db_conn() as conn:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
                (chat_id, role, content, int(time.time())),
            )


def get_recent_messages(chat_id: int, limit: int = 10) -> List[Tuple[str, str]]:
    """Get recent messages from PostgreSQL"""
    with get_db_conn() as conn:
        with conn.cursor() as c:
            c.execute(
                "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s",
                (chat_id, limit),
            )
            rows = c.fetchall()
    rows.reverse()
    return rows


def get_first_message(chat_id: int) -> Optional[Tuple[str, str]]:
    """Get first message for long-term context"""
    with get_db_conn() as conn:
        with conn.cursor() as c:
            c.execute(
                "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY created_at ASC LIMIT 1",
                (chat_id,),
            )
            return c.fetchone()


def add_file_context(
    chat_id: int, filename: str, file_type: str, content: str, file_path: str
) -> None:
    """Add file context to PostgreSQL"""
    with get_db_conn() as conn:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO file_contexts (chat_id, filename, file_type, content, file_path, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (chat_id, filename, file_type, content, file_path, int(time.time())),
            )


def get_file_contexts(chat_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Get file contexts from PostgreSQL"""
    with get_db_conn() as conn:
        with conn.cursor() as c:
            c.execute(
                "SELECT filename, file_type, content FROM file_contexts WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s",
                (chat_id, limit),
            )
            rows = c.fetchall()
    return [{"filename": r[0], "file_type": r[1], "content": r[2]} for r in rows]


def clear_file_contexts(chat_id: int) -> None:
    """Clear file contexts for a chat"""
    with get_db_conn() as conn:
        with conn.cursor() as c:
            c.execute("DELETE FROM file_contexts WHERE chat_id = %s", (chat_id,))


# --- Setup conversation for new users ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the /start command."""
    chat_id = update.effective_chat.id
    user_settings = get_user_settings(chat_id)

    if user_settings and user_settings.get("api_key"):
        await update.message.reply_text(
            "🤖 С возвращением! Я готов к работе.\n"
            "Используйте /settings для изменения настроек или /help для просмотра всех команд."
        )
        return ConversationHandler.END

    # New user
    await update.message.reply_text(
        "👋 Добро пожаловать! Я — ваш персональный AI-ассистент.\n\n"
        "Для начала работы мне нужно узнать несколько деталей. "
        "Давайте настроим ваш аккаунт.\n\n"
        "Пожалуйста, пришлите URL endpoint'а API (например, https://openrouter.ai/api/v1/chat/completions)."
    )
    context.user_data["settings"] = {}
    return ASK_ENDPOINT


async def ask_endpoint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Saves endpoint and asks for model."""
    context.user_data["settings"]["endpoint"] = update.message.text.strip()
    await update.message.reply_text(
        "✅ Отлично! Теперь укажите название основной модели (например, mistralai/mistral-7b-instruct:free)."
    )
    return ASK_MODEL


async def ask_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Saves model and asks for API key."""
    context.user_data["settings"]["model"] = update.message.text.strip()
    await update.message.reply_text(
        "🔑 Теперь пришлите ваш API-ключ. Он будет зашифрован для безопасности."
    )
    return ASK_APIKEY


async def ask_apikey(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Saves API key and completes setup."""
    chat_id = update.effective_chat.id
    api_key = update.message.text.strip()

    # Get default settings and update with user-provided info
    settings = get_default_settings()
    settings.update(context.user_data["settings"])
    settings["api_key"] = api_key

    # Create the user in the database
    create_or_update_user(chat_id, settings)

    await update.message.reply_text(
        "🎉 Настройка завершена! Ваш аккаунт создан.\n\n"
        "Я готов к работе. Отправьте мне сообщение, файл или воспользуйтесь командой /help для получения дополнительной информации."
    )
    context.user_data.clear()
    return ConversationHandler.END


async def cancel_setup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the setup process."""
    await update.message.reply_text(
        "Настройка отменена. Вы можете начать заново в любой момент командой /start."
    )
    context.user_data.clear()
    return ConversationHandler.END


# --- Settings conversation for existing users ---
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the settings conversation for an existing user."""
    chat_id = update.effective_chat.id
    user_settings = get_user_settings(chat_id)

    if not user_settings:
        await update.message.reply_text(
            "Сначала вам нужно зарегистрироваться. Пожалуйста, используйте команду /start."
        )
        return ConversationHandler.END

    text = (
        f"🛠 Ваши текущие настройки:\n"
        f"• Endpoint: `{user_settings.get('endpoint')}`\n"
        f"• Основная модель: `{user_settings.get('model')}`\n"
        f"• Thinking модель: `{user_settings.get('reasoning_model')}`\n"
        f"• Роутер модель: `{user_settings.get('router_model')}`\n"
        f"• Авто thinking: `{user_settings.get('auto_thinking', True)}`\n"
        f"• Поиск включен: `{user_settings.get('search_enabled', True)}`\n"
        f"• Контекст сообщений: `{user_settings.get('context_messages_count')}`\n"
        f"• System prompt: `{user_settings.get('system_prompt')}`\n\n"
        "Что хотите изменить?"
    )
    keyboard = [
        [
            InlineKeyboardButton("🔑 API Key", callback_data="settings_api_key"),
            InlineKeyboardButton(
                "💭 System prompt", callback_data="settings_system_prompt"
            ),
        ],
        [
            InlineKeyboardButton("🌐 Endpoint", callback_data="settings_endpoint"),
            InlineKeyboardButton("🤖 Основная модель", callback_data="settings_model"),
        ],
        [
            InlineKeyboardButton(
                "🧠 Thinking модель", callback_data="settings_reasoning_model"
            ),
            InlineKeyboardButton(
                "🎯 Роутер модель", callback_data="settings_router_model"
            ),
        ],
        [
            InlineKeyboardButton(
                "⚡ Авто thinking", callback_data="settings_auto_thinking"
            ),
            InlineKeyboardButton("🔍 Поиск", callback_data="settings_search_enabled"),
        ],
        [
            InlineKeyboardButton(
                "📝 Контекст", callback_data="settings_context_messages_count"
            ),
            InlineKeyboardButton("❌ Отмена", callback_data="settings_cancel"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        text, reply_markup=reply_markup, parse_mode="MarkdownV2"
    )
    return SELECTING_SETTING


async def settings_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handle settings button clicks."""
    query = update.callback_query
    await query.answer()
    choice = query.data

    if choice == "settings_cancel":
        await query.edit_message_text("❌ Изменение настроек отменено.")
        return ConversationHandler.END

    context.user_data["setting_to_change"] = choice.replace("settings_", "")

    prompts = {
        "api_key": "🔑 Введите новый API Key:",
        "system_prompt": "💭 Введите новый system prompt:",
        "endpoint": "🌐 Введите новый URL endpoint'а:",
        "model": "🤖 Введите новую основную модель:",
        "reasoning_model": "🧠 Введите модель для thinking режима:",
        "router_model": "🎯 Введите модель для выбора режима:",
        "auto_thinking": "⚡ Автоматический thinking (true/false):",
        "search_enabled": "🔍 Поиск в интернете (true/false):",
        "context_messages_count": "📝 Количество сообщений в контексте (число):",
    }

    prompt = prompts.get(
        context.user_data["setting_to_change"], "Введите новое значение:"
    )
    await query.edit_message_text(text=prompt)
    return UPDATING_SETTING


async def set_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Saves the new setting value for the user."""
    chat_id = update.effective_chat.id
    new_value = update.message.text.strip()
    setting_key = context.user_data.get("setting_to_change")

    if not setting_key:
        await update.message.reply_text(
            "❌ Произошла ошибка. Попробуйте снова /settings."
        )
        return ConversationHandler.END

    # Type conversion and validation
    if setting_key in ["auto_thinking", "search_enabled"]:
        value_to_save = new_value.lower() in ["true", "1", "yes", "on", "включен"]
    elif setting_key == "context_messages_count":
        try:
            value_to_save = int(new_value)
        except ValueError:
            await update.message.reply_text("❌ Ошибка: введите число.")
            return UPDATING_SETTING  # Ask again
    else:
        value_to_save = new_value

    # Update database
    create_or_update_user(chat_id, {setting_key: value_to_save})
    await update.message.reply_text(f"✅ Настройка '{setting_key}' обновлена.")

    context.user_data.clear()
    return ConversationHandler.END


# --- Main Logic ---


# --- Helper to extract assistant text ---
def extract_text_from_response(resp_json: dict) -> str:
    # This function remains the same
    try:
        if "output" in resp_json:
            out = resp_json["output"]
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if "content" in first and isinstance(first["content"], list):
                    for item in first["content"]:
                        if item.get("type") in ("output_text", "message"):
                            return item.get("text") or item.get("content") or str(item)
                    return "\n".join(
                        [c.get("text", str(c)) for c in first.get("content", [])]
                    )
    except Exception:
        pass

    try:
        if (
            "choices" in resp_json
            and isinstance(resp_json["choices"], list)
            and len(resp_json["choices"]) > 0
        ):
            ch = resp_json["choices"][0]
            if "message" in ch and "content" in ch["message"]:
                if isinstance(ch["message"]["content"], dict):
                    return ch["message"]["content"].get(
                        "text", json.dumps(ch["message"]["content"])
                    )
                return ch["message"]["content"]
            if "text" in ch:
                return ch["text"]
    except Exception:
        pass

    return json.dumps(resp_json)[:2000]


# --- Web search function ---
def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search web using DuckDuckGo API"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1",
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = []

            if data.get("AbstractText"):
                results.append(
                    {
                        "title": data.get("AbstractText", "")[:100],
                        "snippet": data.get("AbstractText", ""),
                        "url": data.get("AbstractURL", ""),
                    }
                )

            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(
                        {
                            "title": topic.get("Text", "")[:50],
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", ""),
                        }
                    )

            return results[:num_results]
    except Exception as e:
        logger.error(f"Web search error: {e}")

    return []


def should_use_thinking_mode(
    user_message: str, api_key: str, router_endpoint: str, router_model: str
) -> bool:
    """Use router model to determine if thinking mode is needed"""
    try:
        prompt = f"""Analyze this user message and determine if it requires deep reasoning, complex problem-solving, or step-by-step thinking.

User message: "{user_message}"

Respond with ONLY "YES" if thinking mode is needed for:
- Complex math problems
- Multi-step reasoning
- Coding challenges
- Analysis requiring careful consideration
- Problems that benefit from showing work

Respond with ONLY "NO" for:
- Simple questions
- Casual conversation
- Basic information requests
- Straightforward tasks

Response:"""

        payload = {
            "model": router_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            router_endpoint, headers=headers, json=payload, timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            text = extract_text_from_response(result).strip().upper()
            return "YES" in text
    except Exception as e:
        logger.error(f"Router model error: {e}")

    return False


def should_search_web(user_message: str) -> bool:
    """Simple heuristic to determine if web search is needed"""
    search_indicators = [
        "latest",
        "recent",
        "current",
        "today",
        "news",
        "weather",
        "what's happening",
        "update",
        "2024",
        "2025",
        "now",
        "search",
        "find",
        "look up",
    ]

    message_lower = user_message.lower()
    return any(indicator in message_lower for indicator in search_indicators)


# --- File processing functions ---
async def process_document(file_path: Path, file_type: str) -> str:
    """Process uploaded documents and extract text content"""
    try:
        if file_type.lower() == "pdf":
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()

        elif file_type.lower() == "txt":
            return file_path.read_text(encoding="utf-8")

        elif file_type.lower() in ["jpg", "jpeg", "png", "webp"]:
            return f"[IMAGE: {file_path.name}]"

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return f"[Error processing file: {e}]"

    return "[Unsupported file type]"


# --- Main Logic ---


# --- Helper to extract assistant text ---
def extract_text_from_response(resp_json: dict) -> str:
    # This function remains the same
    try:
        if "output" in resp_json:
            out = resp_json["output"]
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if "content" in first and isinstance(first["content"], list):
                    for item in first["content"]:
                        if item.get("type") in ("output_text", "message"):
                            return item.get("text") or item.get("content") or str(item)
                    return "\n".join(
                        [c.get("text", str(c)) for c in first.get("content", [])]
                    )
    except Exception:
        pass

    try:
        if (
            "choices" in resp_json
            and isinstance(resp_json["choices"], list)
            and len(resp_json["choices"]) > 0
        ):
            ch = resp_json["choices"][0]
            if "message" in ch and "content" in ch["message"]:
                if isinstance(ch["message"]["content"], dict):
                    return ch["message"]["content"].get(
                        "text", json.dumps(ch["message"]["content"])
                    )
                return ch["message"]["content"]
            if "text" in ch:
                return ch["text"]
    except Exception:
        pass

    return json.dumps(resp_json)[:2000]


# --- Web search function ---
def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search web using DuckDuckGo API"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1",
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = []

            if data.get("AbstractText"):
                results.append(
                    {
                        "title": data.get("AbstractText", "")[:100],
                        "snippet": data.get("AbstractText", ""),
                        "url": data.get("AbstractURL", ""),
                    }
                )

            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(
                        {
                            "title": topic.get("Text", "")[:50],
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", ""),
                        }
                    )

            return results[:num_results]
    except Exception as e:
        logger.error(f"Web search error: {e}")

    return []


def should_use_thinking_mode(
    user_message: str, api_key: str, router_endpoint: str, router_model: str
) -> bool:
    """Use router model to determine if thinking mode is needed"""
    try:
        prompt = f"""Analyze this user message and determine if it requires deep reasoning, complex problem-solving, or step-by-step thinking.

User message: "{user_message}"

Respond with ONLY "YES" if thinking mode is needed for:
- Complex math problems
- Multi-step reasoning
- Coding challenges
- Analysis requiring careful consideration
- Problems that benefit from showing work

Respond with ONLY "NO" for:
- Simple questions
- Casual conversation
- Basic information requests
- Straightforward tasks

Response:"""

        payload = {
            "model": router_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            router_endpoint, headers=headers, json=payload, timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            text = extract_text_from_response(result).strip().upper()
            return "YES" in text
    except Exception as e:
        logger.error(f"Router model error: {e}")

    return False


def should_search_web(user_message: str) -> bool:
    """Simple heuristic to determine if web search is needed"""
    search_indicators = [
        "latest",
        "recent",
        "current",
        "today",
        "news",
        "weather",
        "what's happening",
        "update",
        "2024",
        "2025",
        "now",
        "search",
        "find",
        "look up",
    ]

    message_lower = user_message.lower()
    return any(indicator in message_lower for indicator in search_indicators)


# --- File processing functions ---
async def process_document(file_path: Path, file_type: str) -> str:
    """Process uploaded documents and extract text content"""
    try:
        if file_type.lower() == "pdf":
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()

        elif file_type.lower() == "txt":
            return file_path.read_text(encoding="utf-8")

        elif file_type.lower() in ["jpg", "jpeg", "png", "webp"]:
            return f"[IMAGE: {file_path.name}]"

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return f"[Error processing file: {e}]"

    return "[Unsupported file type]"


# --- Message Handlers (Multi-User Aware) ---


async def main_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main handler for text, documents, and photos, with robust error handling."""
    chat_id = update.effective_chat.id
    try:
        user_settings = get_user_settings(chat_id)

        if not user_settings or not user_settings.get("api_key"):
            await update.message.reply_text(
                "Пожалуйста, сначала настройте свой аккаунт с помощью /start."
            )
            return

        user_text = update.message.text
        if user_text:
            add_message(chat_id, "user", user_text)

        # File handling logic
        if update.message.document:
            await handle_document(update, context, user_settings)
            if not user_text:
                return
        if update.message.photo:
            await handle_photo(update, context, user_settings)
            if not user_text:
                return

        if not user_text:
            return

        # --- Main LLM call logic ---
        status_message = await update.message.reply_text("🤔 Анализирую запрос...")

        api_key = user_settings["api_key"]
        endpoint = user_settings["endpoint"]
        model = user_settings["model"]
        reasoning_model = user_settings["reasoning_model"]
        router_model = user_settings["router_model"]
        system_prompt = user_settings["system_prompt"]
        context_messages_count = user_settings["context_messages_count"]
        auto_thinking = user_settings["auto_thinking"]
        search_enabled = user_settings["search_enabled"]

        search_results = []
        if search_enabled and should_search_web(user_text):
            await status_message.edit_text("🔍 Ищу информацию в интернете...")
            search_results = search_web(user_text)

        use_thinking = False
        if auto_thinking and reasoning_model:
            use_thinking = await asyncio.to_thread(
                should_use_thinking_mode, user_text, api_key, endpoint, router_model
            )

        mode_text = "🧠 Thinking режим" if use_thinking else "💬 Обычный режим"
        search_text = " + 🔍 поиск" if search_results else ""
        await status_message.edit_text(f"{mode_text}{search_text} - обрабатываю...")

        messages = []
        enhanced_system_prompt = system_prompt

        file_contexts = get_file_contexts(chat_id)
        if file_contexts:
            enhanced_system_prompt += "\n\nДоступный контекст файлов:\n"
            for fc in file_contexts:
                enhanced_system_prompt += (
                    f"\n=== {fc['filename']} ===\n{fc['content'][:1000]}...\n"
                )

        if search_results:
            enhanced_system_prompt += "\n\nРезультаты поиска в интернете:\n"
            for i, result in enumerate(search_results, 1):
                enhanced_system_prompt += (
                    f"{i}. {result['title']}\n{result['snippet'][:200]}...\n"
                )

        messages.append({"role": "system", "content": enhanced_system_prompt})

        first_message = get_first_message(chat_id)
        history = get_recent_messages(chat_id, limit=context_messages_count)

        if first_message and first_message not in history:
            messages.append({"role": first_message[0], "content": first_message[1]})

        for role, content in history:
            messages.append({"role": role, "content": content})

        if not any(m["role"] == "user" and m["content"] == user_text for m in messages):
            messages.append({"role": "user", "content": user_text})

        selected_model = reasoning_model if use_thinking else model
        payload = {"model": selected_model, "messages": messages}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        timeout = 180 if use_thinking else 60
        response = await asyncio.to_thread(
            requests.post, endpoint, headers=headers, json=payload, timeout=timeout
        )
        response.raise_for_status()

        resp_json = response.json()
        assistant_text = extract_text_from_response(resp_json)
        add_message(chat_id, "assistant", assistant_text)

        await status_message.delete()
        await update.message.reply_text(assistant_text)

    except (NetworkError, TimedOut) as e:
        logger.error(f"Telegram API network error for chat_id {chat_id}: {e}")
        try:
            await context.bot.send_message(
                chat_id,
                f"❌ Ошибка сети при общении с Telegram: {e}. Пожалуйста, попробуйте еще раз позже.",
            )
        except Exception as inner_e:
            logger.error(
                f"Failed to even send a network error message to chat_id {chat_id}: {inner_e}"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API request error for chat_id {chat_id}: {e}")
        await status_message.edit_text(f"❌ Ошибка сети при обращении к LLM API: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred for chat_id {chat_id}: {e}", exc_info=True
        )
        await context.bot.send_message(
            chat_id, f"❌ Произошла непредвиденная ошибка. Администратор был уведомлен."
        )


async def handle_document(
    update: Update, context: ContextTypes.DEFAULT_TYPE, user_settings: Dict[str, Any]
):
    """Handle uploaded documents for a specific user."""
    document = update.message.document
    chat_id = update.effective_chat.id

    file_extension = document.file_name.split(".")[-1].lower()

    # Check file contexts limit
    if len(get_file_contexts(chat_id)) >= 5:  # Hardcoded limit for now
        await update.message.reply_text(
            "📂 Достигнут лимит файлов в контексте (5). Используйте /context для управления."
        )
        return

    await update.message.reply_text("📥 Обрабатываю файл...")

    try:
        file = await context.bot.get_file(document.file_id)
        file_path = UPLOADS_DIR / f"{chat_id}_{int(time.time())}_{document.file_name}"
        await file.download_to_drive(file_path)

        content = await process_document(file_path, file_extension)
        add_file_context(
            chat_id, document.file_name, file_extension, content, str(file_path)
        )

        await update.message.reply_text(
            f"✅ Файл '{document.file_name}' добавлен в контекст!"
        )

    except Exception as e:
        logger.error(f"Error processing document for chat_id {chat_id}: {e}")
        await update.message.reply_text(f"❌ Ошибка обработки файла: {e}")


async def handle_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE, user_settings: Dict[str, Any]
):
    """Handle uploaded photos for a specific user."""
    photo = update.message.photo[-1]
    chat_id = update.effective_chat.id

    await update.message.reply_text("🖼 Обрабатываю изображение...")

    try:
        file = await context.bot.get_file(photo.file_id)
        file_path = UPLOADS_DIR / f"{chat_id}_{int(time.time())}_image.jpg"
        await file.download_to_drive(file_path)

        content = f"[IMAGE: {file_path.name} - Vision API not implemented]"
        add_file_context(chat_id, file_path.name, "jpg", content, str(file_path))

        await update.message.reply_text("✅ Изображение добавлено в контекст!")

    except Exception as e:
        logger.error(f"Error processing photo for chat_id {chat_id}: {e}")
        await update.message.reply_text(f"❌ Ошибка обработки изображения: {e}")


# --- Context management commands ---
async def context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show context management options"""
    chat_id = update.effective_chat.id
    file_contexts = get_file_contexts(chat_id)

    if not file_contexts:
        await update.message.reply_text(
            "📂 Контекст пуст. Загрузите файлы для добавления в контекст."
        )
        return

    context_info = "📂 Текущий контекст файлов:\n\n"
    for i, fc in enumerate(file_contexts, 1):
        context_info += f"{i}. 📄 {fc['filename']} ({fc['file_type']})\n"

    keyboard = [
        [InlineKeyboardButton("🗑 Очистить контекст", callback_data="clear_context")]
    ]
    await update.message.reply_text(
        context_info, reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def context_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle context management callbacks"""
    query = update.callback_query
    await query.answer()

    if query.data == "clear_context":
        chat_id = update.effective_chat.id
        clear_file_contexts(chat_id)
        await query.edit_message_text("✅ Контекст файлов очищен!")


# --- Help Command ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Команды бота**\n\n"
        "/start - Начало работы и регистрация\n"
        "/settings - Настройка вашего аккаунта\n"
        "/context - Управление файлами в контексте\n"
        "/help - Эта справка"
    )


# --- Entry point ---
def main():
    init_db()
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set in environment")

    app = ApplicationBuilder().token(token).build()

    # Setup conversation
    setup_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            ASK_ENDPOINT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_endpoint)
            ],
            ASK_MODEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_model)],
            ASK_APIKEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_apikey)],
        },
        fallbacks=[CommandHandler("cancel", cancel_setup)],
    )

    # Settings conversation
    settings_handler = ConversationHandler(
        entry_points=[CommandHandler("settings", settings_command)],
        states={
            SELECTING_SETTING: [
                CallbackQueryHandler(settings_callback_handler, pattern="^settings_")
            ],
            UPDATING_SETTING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_setting_value)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_setup)],  # Can reuse cancel
    )

    app.add_handler(setup_handler)
    app.add_handler(settings_handler)
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("context", context_command))
    app.add_handler(
        CallbackQueryHandler(context_callback_handler, pattern="^clear_context")
    )

    # Generic message handler for text, photos, and documents (with lower priority)
    app.add_handler(
        MessageHandler(
            filters.TEXT | filters.PHOTO | filters.Document.ALL, main_message_handler
        ),
        group=1,
    )

    logger.info("🚀 Multi-user bot started successfully!")
    return app


if __name__ == "__main__":
    try:
        application = main()
        application.run_polling()
    except Exception as e:
        logger.critical(f"Bot failed to start: {e}")
