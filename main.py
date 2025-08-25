#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sqlite3
import time
import base64
import io
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse

import psycopg2
import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, Document, PhotoSize
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

# Files
CONFIG_FILE = Path("config.json")
ENV_FILE = Path(".env")
UPLOADS_DIR = Path("uploads")

# Create uploads directory
UPLOADS_DIR.mkdir(exist_ok=True)

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
    "reasoning_model": "deepseek/deepseek-r1-distill-llama-70b",  # Thinking model
    "router_model": "anthropic/claude-3.5-sonnet:beta",  # Model to choose between thinking/non-thinking
    "system_prompt": "You are a helpful assistant.",
    "context_messages_count": 10,
    "auto_thinking": True,
    "search_enabled": True,
    "max_context_files": 5,
    "supported_file_types": ["pdf", "txt", "jpg", "jpeg", "png", "webp"]
}


# --- Config helpers ---
def read_config() -> dict:
    if not CONFIG_FILE.exists():
        write_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        cfg = json.loads(CONFIG_FILE.read_text())
        # Ensure all default keys exist
        for key, value in DEFAULT_CONFIG.items():
            if key not in cfg:
                cfg[key] = value
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()


def write_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


# --- PostgreSQL Database helpers ---
def get_db_conn():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def init_db() -> None:
    """Initialize PostgreSQL database with retry logic"""
    retries = 5
    while retries > 0:
        try:
            conn = get_db_conn()
            c = conn.cursor()
            
            # Messages table
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
            
            # File contexts table
            c.execute(
                '''
                CREATE TABLE IF NOT EXISTS file_contexts (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    filename TEXT,
                    file_type TEXT,
                    content TEXT,
                    file_path TEXT,
                    created_at BIGINT
                )
                '''
            )
            
            conn.commit()
            c.close()
            conn.close()
            logger.info("PostgreSQL database initialized successfully.")
            return
        except psycopg2.OperationalError as e:
            logger.warning("DB not ready, retrying... (%s)", e)
            retries -= 1
            time.sleep(5)
    raise RuntimeError("Could not connect to PostgreSQL database.")


def add_message(chat_id: int, role: str, content: str) -> None:
    """Add message to PostgreSQL"""
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
    """Get recent messages from PostgreSQL"""
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
    """Get first message for long-term context"""
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


def add_file_context(chat_id: int, filename: str, file_type: str, content: str, file_path: str) -> None:
    """Add file context to PostgreSQL"""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO file_contexts (chat_id, filename, file_type, content, file_path, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
        (chat_id, filename, file_type, content, file_path, int(time.time())),
    )
    conn.commit()
    c.close()
    conn.close()


def get_file_contexts(chat_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Get file contexts from PostgreSQL"""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(
        "SELECT filename, file_type, content FROM file_contexts WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s",
        (chat_id, limit),
    )
    rows = c.fetchall()
    c.close()
    conn.close()
    return [{"filename": row[0], "file_type": row[1], "content": row[2]} for row in rows]


def clear_file_contexts(chat_id: int) -> None:
    """Clear file contexts for a chat"""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("DELETE FROM file_contexts WHERE chat_id = %s", (chat_id,))
    conn.commit()
    c.close()
    conn.close()


# --- Web search function ---
def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search web using DuckDuckGo API"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_redirect': '1',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Get instant answer if available
            if data.get('AbstractText'):
                results.append({
                    'title': data.get('AbstractText', '')[:100],
                    'snippet': data.get('AbstractText', ''),
                    'url': data.get('AbstractURL', '')
                })
            
            # Get related topics
            for topic in data.get('RelatedTopics', [])[:num_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '')[:50],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', '')
                    })
            
            return results[:num_results]
    except Exception as e:
        logger.error(f"Web search error: {e}")
    
    return []


def should_use_thinking_mode(user_message: str, api_key: str, router_endpoint: str, router_model: str) -> bool:
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
            "max_tokens": 10
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(router_endpoint, headers=headers, json=payload, timeout=15)
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
        "latest", "recent", "current", "today", "news", "weather",
        "what's happening", "update", "2024", "2025", "now",
        "search", "find", "look up"
    ]
    
    message_lower = user_message.lower()
    return any(indicator in message_lower for indicator in search_indicators)


# --- File processing functions ---
async def process_document(file_path: Path, file_type: str) -> str:
    """Process uploaded documents and extract text content"""
    try:
        if file_type.lower() == 'pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        
        elif file_type.lower() == 'txt':
            return file_path.read_text(encoding='utf-8')
        
        elif file_type.lower() in ['jpg', 'jpeg', 'png', 'webp']:
            # For images, return a placeholder - actual image processing
            # would require vision API integration
            return f"[IMAGE: {file_path.name}]"
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return f"[Error processing file: {e}]"
    
    return "[Unsupported file type]"


# --- Setup conversation (/start) ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = read_config()
    api_key = os.getenv("API_KEY")
    if cfg.get("endpoint") and cfg.get("model") and api_key:
        await update.message.reply_text(
            "🤖 Бот настроен! Доступные функции:\n"
            "💬 Обычный чат\n"
            "🧠 Автоматический thinking режим\n"
            "🔍 Поиск в интернете\n"
            "📎 Загрузка файлов (PDF, TXT, изображения)\n"
            "💾 PostgreSQL база данных\n\n"
            "Отправляйте сообщения или файлы. Используйте /settings для настроек, /context для управления контекстом."
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Добро пожаловать! Пожалуйста, пришлите URL endpoint'а API (например https://openrouter.ai/api/v1/chat/completions)."
    )
    return SET_ENDPOINT


async def set_endpoint(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["endpoint"] = update.message.text.strip()
    await update.message.reply_text("Укажите основную модель (пример: deepseek/deepseek-chat):")
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

    await update.message.reply_text(
        "✅ Конфигурация сохранена!\n\n"
        "🤖 Основные функции:\n"
        "• Автоматический выбор thinking/обычного режима\n"
        "• Поиск в интернете для актуальной информации\n"
        "• Поддержка файлов: PDF, TXT, изображения\n"
        "• PostgreSQL для надежного хранения\n\n"
        "Отправьте сообщение для начала работы!\n"
        "Используйте /settings для тонкой настройки."
    )
    return ConversationHandler.END


async def cancel_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Настройка отменена.")
    return ConversationHandler.END


# --- Enhanced settings ---
async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Enhanced settings with new options"""
    cfg = read_config()
    text = (
        f"🛠 Текущая конфигурация:\n"
        f"• Endpoint: `{cfg.get('endpoint')}`\n"
        f"• Основная модель: `{cfg.get('model')}`\n"
        f"• Thinking модель: `{cfg.get('reasoning_model')}`\n"
        f"• Роутер модель: `{cfg.get('router_model')}`\n"
        f"• Авто thinking: `{cfg.get('auto_thinking', True)}`\n"
        f"• Поиск включен: `{cfg.get('search_enabled', True)}`\n"
        f"• Контекст сообщений: `{cfg.get('context_messages_count')}`\n"
        f"• System prompt: `{cfg.get('system_prompt')}`\n\n"
        "Что хотите изменить?"
    )
    keyboard = [
        [
            InlineKeyboardButton("🔑 API Key", callback_data="settings_apikey"),
            InlineKeyboardButton("💭 System prompt", callback_data="settings_system_prompt")
        ],
        [
            InlineKeyboardButton("🌐 Endpoint", callback_data="settings_endpoint"),
            InlineKeyboardButton("🤖 Основная модель", callback_data="settings_model")
        ],
        [
            InlineKeyboardButton("🧠 Thinking модель", callback_data="settings_reasoning_model"),
            InlineKeyboardButton("🎯 Роутер модель", callback_data="settings_router_model")
        ],
        [
            InlineKeyboardButton("⚡ Авто thinking", callback_data="settings_auto_thinking"),
            InlineKeyboardButton("🔍 Поиск", callback_data="settings_search_enabled")
        ],
        [
            InlineKeyboardButton("📝 Контекст сообщений", callback_data="settings_context_messages_count"),
            InlineKeyboardButton("❌ Отмена", callback_data="settings_cancel")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        text, reply_markup=reply_markup, parse_mode="MarkdownV2"
    )
    return SELECTING_SETTING


async def settings_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handle enhanced settings button clicks"""
    query = update.callback_query
    await query.answer()
    choice = query.data

    if choice == "settings_cancel":
        await query.edit_message_text("❌ Изменение настроек отменено.")
        return ConversationHandler.END

    context.user_data["setting_to_change"] = choice

    prompts = {
        "settings_apikey": "🔑 Введите новый API Key:",
        "settings_system_prompt": "💭 Введите новый system prompt:",
        "settings_endpoint": "🌐 Введите новый URL endpoint'а:",
        "settings_model": "🤖 Введите новую основную модель:",
        "settings_reasoning_model": "🧠 Введите модель для thinking режима:",
        "settings_router_model": "🎯 Введите модель для выбора режима:",
        "settings_auto_thinking": "⚡ Автоматический thinking (true/false):",
        "settings_search_enabled": "🔍 Поиск в интернете (true/false):",
        "settings_context_messages_count": "📝 Количество сообщений в контексте (число):"
    }
    
    prompt = prompts.get(choice, "Введите новое значение:")
    await query.edit_message_text(text=prompt)
    return UPDATING_SETTING


async def set_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save enhanced setting values"""
    new_value = update.message.text.strip()
    setting_to_change = context.user_data.get("setting_to_change")

    if not setting_to_change:
        await update.message.reply_text("❌ Произошла ошибка. Попробуйте снова /settings.")
        return ConversationHandler.END

    cfg = read_config()
    
    if setting_to_change == "settings_apikey":
        with open(ENV_FILE, "a") as f:
            f.write(f"\nAPI_KEY={new_value}\n")
        os.environ["API_KEY"] = new_value
        await update.message.reply_text("✅ API key сохранён.")
    
    elif setting_to_change in ["settings_auto_thinking", "settings_search_enabled"]:
        bool_value = new_value.lower() in ['true', '1', 'yes', 'on', 'включен']
        setting_key = setting_to_change.replace("settings_", "")
        cfg[setting_key] = bool_value
        write_config(cfg)
        status = "включен" if bool_value else "выключен"
        await update.message.reply_text(f"✅ Настройка обновлена: {status}")
    
    elif setting_to_change == "settings_context_messages_count":
        try:
            cfg["context_messages_count"] = int(new_value)
            write_config(cfg)
            await update.message.reply_text("✅ Количество сообщений в контексте обновлено.")
        except ValueError:
            await update.message.reply_text("❌ Ошибка: введите число.")
    
    else:
        setting_key = setting_to_change.replace("settings_", "")
        cfg[setting_key] = new_value
        write_config(cfg)
        await update.message.reply_text(f"✅ {setting_key} обновлён.")

    context.user_data.clear()
    return ConversationHandler.END


async def cancel_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel settings conversation"""
    await update.message.reply_text("❌ Изменение настроек отменено.")
    context.user_data.clear()
    return ConversationHandler.END


# --- Context management commands ---
async def context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show context management options"""
    chat_id = update.effective_chat.id
    file_contexts = get_file_contexts(chat_id)
    
    if not file_contexts:
        await update.message.reply_text("📂 Контекст пуст. Загрузите файлы для добавления в контекст.")
        return
    
    context_info = "📂 Текущий контекст:\n\n"
    for i, fc in enumerate(file_contexts, 1):
        context_info += f"{i}. 📄 {fc['filename']} ({fc['file_type']})\n"
    
    context_info += f"\n📊 Всего файлов: {len(file_contexts)}"
    
    keyboard = [
        [InlineKeyboardButton("🗑 Очистить контекст", callback_data="clear_context")],
        [InlineKeyboardButton("❌ Отмена", callback_data="cancel_context")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(context_info, reply_markup=reply_markup)


async def context_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle context management callbacks"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "clear_context":
        chat_id = update.effective_chat.id
        clear_file_contexts(chat_id)
        await query.edit_message_text("✅ Контекст очищен!")
    elif query.data == "cancel_context":
        await query.edit_message_text("❌ Операция отменена.")


# --- File handling ---
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uploaded documents"""
    document = update.message.document
    chat_id = update.effective_chat.id
    
    # Check file type
    file_extension = document.file_name.split('.')[-1].lower()
    cfg = read_config()
    supported_types = cfg.get("supported_file_types", DEFAULT_CONFIG["supported_file_types"])
    
    if file_extension not in supported_types:
        await update.message.reply_text(
            f"❌ Неподдерживаемый тип файла: {file_extension}\n"
            f"Поддерживаемые: {', '.join(supported_types)}"
        )
        return
    
    # Check file contexts limit
    existing_contexts = get_file_contexts(chat_id)
    max_files = cfg.get("max_context_files", DEFAULT_CONFIG["max_context_files"])
    
    if len(existing_contexts) >= max_files:
        await update.message.reply_text(
            f"📂 Достигнут лимит файлов в контексте ({max_files}). "
            f"Используйте /context для управления файлами."
        )
        return
    
    await update.message.reply_text("📥 Обрабатываю файл...")
    
    try:
        # Download file
        file = await context.bot.get_file(document.file_id)
        file_path = UPLOADS_DIR / f"{chat_id}_{int(time.time())}_{document.file_name}"
        await file.download_to_drive(file_path)
        
        # Process file content
        content = await process_document(file_path, file_extension)
        
        # Save to context
        add_file_context(chat_id, document.file_name, file_extension, content, str(file_path))
        
        await update.message.reply_text(
            f"✅ Файл '{document.file_name}' добавлен в контекст!\n"
            f"📄 Извлечено символов: {len(content)}"
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        await update.message.reply_text(f"❌ Ошибка обработки файла: {e}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uploaded photos"""
    photo = update.message.photo[-1]  # Get highest resolution
    chat_id = update.effective_chat.id
    
    await update.message.reply_text("🖼 Обрабатываю изображение...")
    
    try:
        # Download photo
        file = await context.bot.get_file(photo.file_id)
        file_path = UPLOADS_DIR / f"{chat_id}_{int(time.time())}_image.jpg"
        await file.download_to_drive(file_path)
        
        # For now, just add placeholder - would need vision API for actual processing
        content = f"[IMAGE: uploaded at {time.strftime('%Y-%m-%d %H:%M:%S')}]"
        
        add_file_context(chat_id, "uploaded_image.jpg", "jpg", content, str(file_path))
        
        await update.message.reply_text("✅ Изображение добавлено в контекст!")
        
    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await update.message.reply_text(f"❌ Ошибка обработки изображения: {e}")


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


# --- Enhanced main message handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced message handler with search, thinking mode, context, and PostgreSQL"""
    if update.message and update.message.text and update.message.text.startswith("/"):
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    add_message(chat_id, "user", user_text)

    cfg = read_config()
    endpoint = cfg.get("endpoint")
    model = cfg.get("model")
    reasoning_model = cfg.get("reasoning_model")
    router_model = cfg.get("router_model")
    system_prompt = cfg.get("system_prompt") or DEFAULT_CONFIG["system_prompt"]
    api_key = os.getenv("API_KEY")
    context_messages_count = cfg.get("context_messages_count", 10)
    auto_thinking = cfg.get("auto_thinking", True)
    search_enabled = cfg.get("search_enabled", True)

    if not endpoint or not model or not api_key:
        await update.message.reply_text(
            "⚙️ Конфигурация неполная. Используйте /start для настройки."
        )
        return

    # Check for forced mode from /mode command
    forced_mode = context.chat_data.get(f"{chat_id}_force_mode")
    if forced_mode:
        context.chat_data.pop(f"{chat_id}_force_mode", None)  # Use once
    
    use_thinking = False
    search_results = []
    status_message = None

    # Web search logic
    if search_enabled and should_search_web(user_text):
        status_message = await update.message.reply_text("🔍 Ищу информацию в интернете...")
        search_results = search_web(user_text)
    
    # Thinking mode logic
    if forced_mode == "thinking":
        use_thinking = True
    elif forced_mode == "normal":
        use_thinking = False
    elif auto_thinking and reasoning_model:
        if not status_message:
            status_message = await update.message.reply_text("🤔 Анализирую запрос...")
        use_thinking = should_use_thinking_mode(user_text, api_key, endpoint, router_model)
    
    # Update status
    if status_message:
        mode_text = "🧠 Thinking режим" if use_thinking else "💬 Обычный режим"
        search_text = f" + 🔍 поиск" if search_results else ""
        forced_text = " (принудительно)" if forced_mode else ""
        await status_message.edit_text(f"{mode_text}{search_text}{forced_text} - обрабатываю...")

    # Prepare messages with enhanced context (PostgreSQL + files + search)
    messages = []
    
    # Enhanced system prompt with file context and search results
    enhanced_system_prompt = system_prompt
    
    # Add file contexts
    file_contexts = get_file_contexts(chat_id)
    if file_contexts:
        enhanced_system_prompt += "\n\nДоступный контекст файлов:\n"
        for fc in file_contexts:
            enhanced_system_prompt += f"\n=== {fc['filename']} ({fc['file_type']}) ===\n{fc['content'][:1000]}...\n"
    
    # Add search results
    if search_results:
        enhanced_system_prompt += "\n\nРезультаты поиска в интернете:\n"
        for i, result in enumerate(search_results, 1):
            enhanced_system_prompt += f"{i}. {result['title']}\n{result['snippet'][:200]}...\n"
    
    messages.append({"role": "system", "content": enhanced_system_prompt})

    # Add first message for long-term context (from PostgreSQL)
    first_message = get_first_message(chat_id)
    
    # Get recent messages (from PostgreSQL)
    history = get_recent_messages(chat_id, limit=context_messages_count)

    # Include first message if not in recent history
    if first_message and first_message not in history:
        messages.append({"role": first_message[0], "content": first_message[1]})

    # Add recent conversation history
    for role, content in history:
        messages.append({"role": role, "content": content})

    # Ensure the current message is included if history was full
    if not any(m["role"] == "user" and m["content"] == user_text for m in messages):
        messages.append({"role": "user", "content": user_text})

    # Choose model based on thinking mode
    selected_model = reasoning_model if use_thinking else model
    
    payload = {"model": selected_model, "messages": messages}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Enhanced typing worker with status updates
    async def typing_worker():
        try:
            counter = 0
            while True:
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                counter += 1
                # Update status every 15 seconds for long requests
                if counter % 5 == 0 and status_message:
                    mode_text = "🧠 Думаю..." if use_thinking else "💬 Отвечаю..."
                    try:
                        await status_message.edit_text(f"{mode_text} ({counter*3}с)")
                    except:
                        pass
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            return

    # Blocking request function with longer timeout for thinking mode
    def do_request():
        try:
            timeout = 180 if use_thinking else 60  # Longer timeout for thinking mode
            r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            return r.status_code, r.text, r.headers
        except Exception as e:
            return None, str(e), {}

    # Start typing indicator
    typing_task = asyncio.create_task(typing_worker())
    
    try:
        status, text, resp_headers = await asyncio.to_thread(do_request)
    finally:
        # Stop typing indicator and clean up status message
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass
        
        if status_message:
            try:
                await status_message.delete()
            except:
                pass

    # Error handling
    if status is None:
        await update.message.reply_text(f"❌ Ошибка запроса к API: {text}")
        logger.error("API request error: %s", text)
        return

    # Detect HTML response (wrong endpoint/key)
    text_start = (text or "")[:200].lstrip()
    content_type = (resp_headers or {}).get("Content-Type", "")
    if (
        text_start.startswith("<!DOCTYPE")
        or text_start.startswith("<html")
        or "text/html" in content_type
    ):
        await update.message.reply_text(
            "❌ Сервер вернул HTML вместо JSON. Проверьте endpoint и API-ключ."
        )
        logger.error("API returned HTML: %s", text_start)
        return

    if status != 200:
        await update.message.reply_text(f"❌ API вернул статус {status}: {text[:500]}")
        logger.error("API returned status %s: %s", status, text[:500])
        return

    try:
        resp_json = json.loads(text)
    except Exception:
        resp_json = {"raw": text}

    assistant_text = extract_text_from_response(resp_json)
    
    # Add mode indicators to response
    mode_indicator = ""
    if use_thinking:
        mode_indicator = "🧠 "
    if search_results:
        mode_indicator += "🔍 "
    if forced_mode:
        mode_indicator += "⚡ "
    
    # Handle long responses by splitting them
    if len(assistant_text) > 4000:
        chunks = [assistant_text[i:i+4000] for i in range(0, len(assistant_text), 4000)]
        for i, chunk in enumerate(chunks):
            prefix = f"{mode_indicator}[{i+1}/{len(chunks)}] " if len(chunks) > 1 else mode_indicator
            await update.message.reply_text(prefix + chunk)
    else:
        await update.message.reply_text(mode_indicator + assistant_text)
    
    # Save assistant response to PostgreSQL
    add_message(chat_id, "assistant", assistant_text)


# --- Additional command handlers ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information"""
    help_text = """
🤖 **Расширенный AI-бот с PostgreSQL**

**📋 Основные команды:**
/start - Первоначальная настройка
/settings - Управление настройками
/context - Управление контекстом файлов
/mode - Ручной выбор режима
/help - Эта справка

**🎯 Функции:**

**💬 Умный чат:**
• Автоматический выбор режима (обычный/thinking)
• Контекстная память разговора
• PostgreSQL для надежного хранения

**🧠 Thinking режим:**
• Автоматически активируется для сложных задач
• Пошаговые рассуждения
• Решение математических задач
• Анализ и программирование

**🔍 Поиск в интернете:**
• Автоматический поиск актуальной информации
• Триггеры: "последние новости", "сегодня", "текущий"
• Интеграция результатов в контекст

**📎 Работа с файлами:**
• PDF документы
• Текстовые файлы (.txt)
• Изображения (JPG, PNG, WEBP)
• До 5 файлов в контексте одновременно

**⚙️ Настраиваемые параметры:**
• Выбор основной и thinking модели
• Включение/отключение автопоиска
• Настройка system промпта
• Количество сообщений в контексте
• Управление лимитами файлов

**💾 База данных:**
• PostgreSQL для надежности
• Долгосрочное хранение истории
• Автоматическое восстановление подключения

Просто отправьте сообщение или файл для начала работы!
    """
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually toggle thinking mode for next response"""
    cfg = read_config()
    current_auto = cfg.get("auto_thinking", True)
    
    keyboard = [
        [
            InlineKeyboardButton("🧠 Thinking", callback_data="force_thinking"),
            InlineKeyboardButton("💬 Обычный", callback_data="force_normal")
        ],
        [
            InlineKeyboardButton("⚡ Авто", callback_data="force_auto"),
            InlineKeyboardButton("❌ Отмена", callback_data="mode_cancel")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    status = "включен" if current_auto else "выключен"
    await update.message.reply_text(
        f"🎛 Выбор режима для следующего ответа:\nТекущий авто-режим: {status}",
        reply_markup=reply_markup
    )


async def mode_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle mode selection callbacks"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "mode_cancel":
        await query.edit_message_text("❌ Выбор режима отменен.")
        return
    
    # Store user preference for next message
    chat_id = update.effective_chat.id
    if query.data == "force_thinking":
        context.chat_data[f"{chat_id}_force_mode"] = "thinking"
        await query.edit_message_text("🧠 Следующий ответ будет в thinking режиме.")
    elif query.data == "force_normal":
        context.chat_data[f"{chat_id}_force_mode"] = "normal"
        await query.edit_message_text("💬 Следующий ответ будет в обычном режиме.")
    elif query.data == "force_auto":
        context.chat_data.pop(f"{chat_id}_force_mode", None)
        await query.edit_message_text("⚡ Следующий ответ будет в автоматическом режиме.")


# --- Entry point ---
def main():
    init_db()  # Initialize PostgreSQL
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set in environment (.env or export)")

    app = ApplicationBuilder().token(token).build()

    # Setup conversation handler
    setup_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SET_ENDPOINT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_endpoint)
            ],
            SET_MODEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_model)],
            SET_APIKEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_apikey)],
        },
        fallbacks=[CommandHandler("cancel", cancel_setup)],
    )

    # Settings conversation handler
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

    # Add all handlers
    app.add_handler(setup_conv_handler)
    app.add_handler(settings_conv_handler)
    
    # Command handlers
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("context", context_command))
    app.add_handler(CommandHandler("mode", mode_command))
    
    # Callback handlers
    app.add_handler(CallbackQueryHandler(context_callback_handler, pattern="^(clear_context|cancel_context)$"))
    app.add_handler(CallbackQueryHandler(mode_callback_handler, pattern="^(force_thinking|force_normal|force_auto|mode_cancel)$"))
    
    # File handlers
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # Main message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Enhanced bot starting with PostgreSQL, thinking mode, web search, and file context")
    return app


if __name__ == "__main__":
    try:
        application = main()
        application.run_polling(close_loop=False)
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")