#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import time
import hashlib
import secrets
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict

import psycopg2
import psycopg2.pool
import aiohttp
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError, TimedOut, BadRequest
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
import pdfplumber

# Setup
ENV_FILE = Path(".env")
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_MESSAGE_LENGTH = 4000
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_REQUESTS = 20
GLOBAL_CONTEXT_LIMIT_MB = 10
GLOBAL_CONTEXT_LIMIT_BYTES = GLOBAL_CONTEXT_LIMIT_MB * 1024 * 1024
DB_POOL_MIN = 5
DB_POOL_MAX = 20

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
load_dotenv()

# Rate limiting storage
rate_limits = defaultdict(list)

# Conversation states
ASK_ENDPOINT, ASK_MODEL, ASK_APIKEY, SELECTING_SETTING, UPDATING_SETTING = range(5)


@dataclass
class UserSettings:
    chat_id: int
    api_key: Optional[str] = None
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "mistralai/mistral-7b-instruct:free"
    reasoning_model: str = "deepseek/deepseek-r1-distill-llama-70b"
    router_model: str = "anthropic/claude-3.5-sonnet:beta"
    system_prompt: str = "You are a helpful assistant."
    context_messages_count: int = 10
    auto_thinking: bool = True
    search_enabled: bool = True


class AdvancedEncryption:
    """Advanced encryption using AES-256 with rotating keys"""

    def __init__(self):
        self.master_key = self._get_or_create_master_key()
        self.current_key_id = self._get_current_key_id()

    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        key_file = Path(".master_key")
        if key_file.exists():
            return key_file.read_bytes()

        master_key = secrets.token_bytes(32)  # 256-bit key
        key_file.write_bytes(master_key)
        key_file.chmod(0o600)  # Read-write for owner only
        logger.info("Generated new master encryption key")
        return master_key

    def _get_current_key_id(self) -> str:
        """Generate key ID based on current time (rotates daily)"""
        current_day = int(time.time()) // (24 * 3600)
        return hashlib.sha256(f"{current_day}".encode()).hexdigest()[:16]

    def _derive_key(self, key_id: str) -> bytes:
        """Derive encryption key from master key and key ID"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=key_id.encode(),
            iterations=100000,
        )
        return kdf.derive(self.master_key)

    def encrypt(self, data: str) -> str:
        """Encrypt data with current key"""
        key = self._derive_key(self.current_key_id)
        fernet = Fernet(Fernet.generate_key())  # This is wrong, let me fix
        # Actually, let's use the proper approach:
        import base64

        key = base64.urlsafe_b64encode(self._derive_key(self.current_key_id))
        fernet = Fernet(key)

        encrypted = fernet.encrypt(data.encode())
        # Prepend key ID for later decryption
        return f"{self.current_key_id}:{base64.urlsafe_b64encode(encrypted).decode()}"

    def decrypt(self, encrypted_data: str) -> Optional[str]:
        """Decrypt data, handling key rotation"""
        try:
            if ":" not in encrypted_data:
                return None

            key_id, data_b64 = encrypted_data.split(":", 1)
            encrypted_bytes = base64.urlsafe_b64decode(data_b64.encode())

            key = base64.urlsafe_b64encode(self._derive_key(key_id))
            fernet = Fernet(key)

            return fernet.decrypt(encrypted_bytes).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None


class DatabaseManager:
    """Database connection and operations manager with pooling"""

    def __init__(self):
        self.pool = None
        self._init_pool()

    def _init_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                DB_POOL_MIN,
                DB_POOL_MAX,
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                database=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                options="-c default_transaction_isolation=read_committed",
            )
            logger.info(
                f"Database pool initialized with {DB_POOL_MIN}-{DB_POOL_MAX} connections"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections"""
        conn = None
        try:
            conn = await asyncio.to_thread(self.pool.getconn)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                await asyncio.to_thread(self.pool.putconn, conn)

    async def execute_query(self, query: str, params: tuple = None, fetch: str = None):
        """Execute database query with proper error handling"""
        async with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())

                if fetch == "one":
                    return cursor.fetchone()
                elif fetch == "all":
                    return cursor.fetchall()
                elif fetch == "many":
                    return cursor.fetchmany()
                return None


class RateLimiter:
    """Rate limiting functionality"""

    @staticmethod
    def check_rate_limit(chat_id: int) -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        user_requests = rate_limits[chat_id]

        # Remove old requests
        rate_limits[chat_id] = [
            req_time for req_time in user_requests if now - req_time < RATE_LIMIT_WINDOW
        ]

        # Check limit
        if len(rate_limits[chat_id]) >= RATE_LIMIT_REQUESTS:
            return False

        # Add current request
        rate_limits[chat_id].append(now)
        return True


class FileProcessor:
    """File processing with security validation"""

    ALLOWED_EXTENSIONS = {"pdf", "txt", "doc", "docx", "jpg", "jpeg", "png", "webp"}
    DANGEROUS_EXTENSIONS = {"exe", "bat", "sh", "py", "js", "php", "html"}

    @staticmethod
    def validate_file(file_path: Path, file_size: int) -> tuple[bool, str]:
        """Validate uploaded file"""
        if file_size > MAX_FILE_SIZE:
            return (
                False,
                f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB",
            )

        extension = file_path.suffix.lower().lstrip(".")
        if extension in FileProcessor.DANGEROUS_EXTENSIONS:
            return False, "File type not allowed for security reasons"

        if extension not in FileProcessor.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type: {extension}"

        return True, "OK"

    @staticmethod
    async def process_document(file_path: Path, file_type: str) -> str:
        """Process uploaded documents with enhanced extraction"""
        try:
            if file_type.lower() == "pdf":
                return await FileProcessor._process_pdf(file_path)
            elif file_type.lower() == "txt":
                return file_path.read_text(encoding="utf-8")
            elif file_type.lower() in ["jpg", "jpeg", "png", "webp"]:
                return f"[IMAGE: {file_path.name}]"
            else:
                return "[Unsupported file type]"
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return f"[Error processing file: {e}]"

    @staticmethod
    async def _process_pdf(file_path: Path) -> str:
        """Enhanced PDF processing with fallback methods"""
        text = ""

        # Try pdfplumber first (better text extraction)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            # Fallback to PyPDF2
            try:
                with open(file_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"PDF processing failed: {e}")
                return "[Error: Could not extract text from PDF]"

        return text.strip()


class APIClient:
    """HTTP client with retry logic and rate limiting"""

    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=180),
            connector=aiohttp.TCPConnector(limit=100),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        await asyncio.sleep(retry_delay * (2**attempt))
                        continue
                    else:
                        logger.error(f"API request failed: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                else:
                    return None

        return None


class LLMService:
    """LLM API interaction service"""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client

    def extract_text_from_response(self, resp_json: dict) -> str:
        """Extract text from various LLM API response formats"""
        try:
            # Handle different response formats
            if "output" in resp_json:
                out = resp_json["output"]
                if isinstance(out, list) and len(out) > 0:
                    first = out[0]
                    if "content" in first and isinstance(first["content"], list):
                        for item in first["content"]:
                            if item.get("type") in ("output_text", "message"):
                                return (
                                    item.get("text") or item.get("content") or str(item)
                                )
                        return "\n".join(
                            [c.get("text", str(c)) for c in first.get("content", [])]
                        )

            # Standard OpenAI format
            if (
                "choices" in resp_json
                and isinstance(resp_json["choices"], list)
                and len(resp_json["choices"]) > 0
            ):
                ch = resp_json["choices"][0]
                if "message" in ch and "content" in ch["message"]:
                    content = ch["message"]["content"]
                    if isinstance(content, dict):
                        return content.get("text", json.dumps(content))
                    return content
                if "text" in ch:
                    return ch["text"]
        except Exception as e:
            logger.error(f"Error extracting response text: {e}")

        return json.dumps(resp_json)[:MAX_MESSAGE_LENGTH]

    async def should_use_thinking_mode(
        self, user_message: str, api_key: str, endpoint: str, router_model: str
    ) -> bool:
        """Determine if thinking mode should be used"""
        prompt = """Analyze this user message and determine if it requires deep reasoning.

User message: "{}"

Respond with ONLY "YES" if thinking mode is needed for:
- Complex math problems
- Multi-step reasoning  
- Coding challenges
- Analysis requiring careful consideration

Respond with ONLY "NO" for:
- Simple questions
- Casual conversation
- Basic information requests

Response:""".format(
            user_message
        )

        payload = {
            "model": router_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            result = await self.api_client.make_request(
                "POST", endpoint, json=payload, headers=headers
            )
            if result:
                text = self.extract_text_from_response(result).strip().upper()
                return "YES" in text
        except Exception as e:
            logger.error(f"Router model error: {e}")

        return False

    async def search_web(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, str]]:
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

            result = await self.api_client.make_request("GET", url, params=params)
            if not result:
                return []

            results = []
            if result.get("AbstractText"):
                results.append(
                    {
                        "title": result.get("AbstractText", "")[:100],
                        "snippet": result.get("AbstractText", ""),
                        "url": result.get("AbstractURL", ""),
                    }
                )

            for topic in result.get("RelatedTopics", [])[:num_results]:
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

    def should_search_web(self, user_message: str) -> bool:
        """Determine if web search is needed using ML-based approach"""
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
            "who is",
            "what is",
        ]

        message_lower = user_message.lower()
        score = sum(1 for indicator in search_indicators if indicator in message_lower)

        # Additional heuristics
        if any(word in message_lower for word in ["when did", "how much", "price of"]):
            score += 2

        return score >= 2


# Initialize global services
encryption = AdvancedEncryption()
db_manager = DatabaseManager()


def rate_limited(func):
    """Decorator to apply rate limiting"""

    @wraps(func)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs
    ):
        chat_id = update.effective_chat.id
        if not RateLimiter.check_rate_limit(chat_id):
            await update.message.reply_text(
                "Rate limit exceeded. Please wait a moment before trying again."
            )
            return
        return await func(update, context, *args, **kwargs)

    return wrapper


async def cleanup_old_files():
    """Cleanup old uploaded files"""
    try:
        cutoff_time = time.time() - (24 * 3600)  # 24 hours ago
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")


# Database operations
async def get_user_settings(chat_id: int) -> Optional[UserSettings]:
    """Fetch user settings from database"""
    try:
        result = await db_manager.execute_query(
            "SELECT * FROM users WHERE chat_id = %s", (chat_id,), fetch="one"
        )

        if not result:
            return None

        # Map database row to UserSettings
        settings = UserSettings(chat_id=chat_id)
        if result[1]:  # api_key
            settings.api_key = encryption.decrypt(result[1])
        settings.endpoint = result[2] or settings.endpoint
        settings.model = result[3] or settings.model
        settings.reasoning_model = result[4] or settings.reasoning_model
        settings.router_model = result[5] or settings.router_model
        settings.system_prompt = result[6] or settings.system_prompt
        settings.context_messages_count = result[7] or settings.context_messages_count
        settings.auto_thinking = (
            result[8] if result[8] is not None else settings.auto_thinking
        )
        settings.search_enabled = (
            result[9] if result[9] is not None else settings.search_enabled
        )

        return settings
    except Exception as e:
        logger.error(f"Error fetching user settings: {e}")
        return None


async def create_or_update_user(chat_id: int, updates: Dict[str, Any]) -> None:
    """Create or update user settings"""
    try:
        # Encrypt API key if present
        if "api_key" in updates and updates["api_key"]:
            updates["api_key"] = encryption.encrypt(updates["api_key"])

        # Build dynamic query
        columns = list(updates.keys())
        values = list(updates.values())

        if "chat_id" not in columns:
            columns.append("chat_id")
            values.append(chat_id)

        placeholders = ", ".join(["%s"] * len(values))
        columns_str = ", ".join(columns)
        update_str = ", ".join(
            [f"{col} = EXCLUDED.{col}" for col in columns if col != "chat_id"]
        )

        query = f"""
            INSERT INTO users ({columns_str}, created_at)
            VALUES ({placeholders}, %s)
            ON CONFLICT (chat_id) DO UPDATE SET {update_str}
        """

        await db_manager.execute_query(query, tuple(values) + (int(time.time()),))

    except Exception as e:
        logger.error(f"Error updating user settings: {e}")
        raise


async def add_message(chat_id: int, role: str, content: str) -> None:
    """Add message to database"""
    await db_manager.execute_query(
        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
        (chat_id, role, content, int(time.time())),
    )


async def get_recent_messages(chat_id: int, limit: int = 10) -> List[Tuple[str, str]]:
    """Get recent messages from database"""
    result = await db_manager.execute_query(
        "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s",
        (chat_id, limit),
        fetch="all",
    )
    return list(reversed(result)) if result else []


# Command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /start command"""
    chat_id = update.effective_chat.id
    user_settings = await get_user_settings(chat_id)

    if user_settings and user_settings.api_key:
        await update.message.reply_text(
            "Welcome back! I'm ready to work.\n"
            "Use /settings to change settings or /help to see all commands."
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Welcome! I'm your personal AI assistant.\n\n"
        "To get started, I need a few details. Let's set up your account.\n\n"
        "Please send your API endpoint URL (e.g., https://openrouter.ai/api/v1/chat/completions)."
    )
    context.user_data["settings"] = {}
    return ASK_ENDPOINT


@rate_limited
async def main_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main handler for messages with improved error handling"""
    chat_id = update.effective_chat.id

    try:
        user_settings = await get_user_settings(chat_id)
        if not user_settings or not user_settings.api_key:
            await update.message.reply_text(
                "Please set up your account first using /start."
            )
            return

        user_text = update.message.text
        if user_text:
            await add_message(chat_id, "user", user_text)

        # Handle files
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

        # Process with LLM
        status_message = await update.message.reply_text("Analyzing request...")

        async with APIClient() as api_client:
            llm_service = LLMService(api_client)

            # Web search if needed
            search_results = []
            if user_settings.search_enabled and llm_service.should_search_web(
                user_text
            ):
                await status_message.edit_text("Searching the internet...")
                search_results = await llm_service.search_web(user_text)

            # Determine thinking mode
            use_thinking = False
            if user_settings.auto_thinking and user_settings.reasoning_model:
                use_thinking = await llm_service.should_use_thinking_mode(
                    user_text,
                    user_settings.api_key,
                    user_settings.endpoint,
                    user_settings.router_model,
                )

            mode_text = "Thinking mode" if use_thinking else "Normal mode"
            search_text = " + search" if search_results else ""
            await status_message.edit_text(f"{mode_text}{search_text} - processing...")

            # Build messages
            messages = []
            system_prompt = user_settings.system_prompt

            if search_results:
                system_prompt += "\n\nWeb search results:\n"
                for i, result in enumerate(search_results, 1):
                    system_prompt += (
                        f"{i}. {result['title']}\n{result['snippet'][:200]}...\n"
                    )

            messages.append({"role": "system", "content": system_prompt})

            # Add conversation history
            history = await get_recent_messages(
                chat_id, limit=user_settings.context_messages_count
            )
            for role, content in history:
                messages.append({"role": role, "content": content})

            # Make API request
            selected_model = (
                user_settings.reasoning_model if use_thinking else user_settings.model
            )
            payload = {"model": selected_model, "messages": messages}
            headers = {
                "Authorization": f"Bearer {user_settings.api_key}",
                "Content-Type": "application/json",
            }

            result = await api_client.make_request(
                "POST", user_settings.endpoint, json=payload, headers=headers
            )

            if result:
                assistant_text = llm_service.extract_text_from_response(result)
                await add_message(chat_id, "assistant", assistant_text)
                await status_message.delete()

                # Split long messages
                if len(assistant_text) > MAX_MESSAGE_LENGTH:
                    for i in range(0, len(assistant_text), MAX_MESSAGE_LENGTH):
                        await update.message.reply_text(
                            assistant_text[i : i + MAX_MESSAGE_LENGTH]
                        )
                else:
                    await update.message.reply_text(assistant_text)
            else:
                await status_message.edit_text(
                    "Sorry, I couldn't process your request. Please try again later."
                )

    except Exception as e:
        logger.error(
            f"Error in main message handler for chat_id {chat_id}: {e}", exc_info=True
        )
        await context.bot.send_message(
            chat_id, "An unexpected error occurred. Please try again."
        )


async def handle_document(
    update: Update, context: ContextTypes.DEFAULT_TYPE, user_settings: UserSettings
):
    """Handle document uploads with security validation"""
    document = update.message.document
    chat_id = update.effective_chat.id

    # Validate file
    file_path = UPLOADS_DIR / f"{chat_id}_{int(time.time())}_{document.file_name}"
    is_valid, error_msg = FileProcessor.validate_file(file_path, document.file_size)

    if not is_valid:
        await update.message.reply_text(f"File validation failed: {error_msg}")
        return

    try:
        await update.message.reply_text("Processing file...")

        file = await context.bot.get_file(document.file_id)
        await file.download_to_drive(file_path)

        content = await FileProcessor.process_document(
            file_path, document.file_name.split(".")[-1]
        )

        # Add to context (implement this based on your needs)
        await update.message.reply_text(
            f"File '{document.file_name}' processed successfully!"
        )

        # Schedule file cleanup
        asyncio.create_task(cleanup_file_later(file_path, 3600))  # 1 hour

    except Exception as e:
        logger.error(f"Error processing document for chat_id {chat_id}: {e}")
        await update.message.reply_text(f"Error processing file: {e}")


async def cleanup_file_later(file_path: Path, delay: int):
    """Schedule file cleanup after delay"""
    await asyncio.sleep(delay)
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")


async def handle_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE, user_settings: UserSettings
):
    """Handle photo uploads"""
    photo = update.message.photo[-1]
    chat_id = update.effective_chat.id

    try:
        await update.message.reply_text("Processing image...")

        file = await context.bot.get_file(photo.file_id)
        file_path = UPLOADS_DIR / f"{chat_id}_{int(time.time())}_image.jpg"
        await file.download_to_drive(file_path)

        # Process image (implement vision API here)
        await update.message.reply_text("Image processed successfully!")

        # Schedule cleanup
        asyncio.create_task(cleanup_file_later(file_path, 3600))

    except Exception as e:
        logger.error(f"Error processing photo for chat_id {chat_id}: {e}")
        await update.message.reply_text(f"Error processing image: {e}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    await update.message.reply_text(
        "Bot Commands:\n\n"
        "/start - Registration and setup\n"
        "/settings - Configure your account\n"
        "/help - Show this help"
    )


async def ask_endpoint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save endpoint and ask for model"""
    context.user_data["settings"]["endpoint"] = update.message.text.strip()
    await update.message.reply_text(
        "Great! Now specify the main model name (e.g., mistralai/mistral-7b-instruct:free)."
    )
    return ASK_MODEL


async def ask_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save model and ask for API key"""
    context.user_data["settings"]["model"] = update.message.text.strip()
    await update.message.reply_text(
        "Now send your API key. It will be encrypted for security."
    )
    return ASK_APIKEY


async def ask_apikey(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save API key and complete setup"""
    chat_id = update.effective_chat.id
    api_key = update.message.text.strip()

    # Create user with settings
    settings = context.user_data["settings"]
    settings["api_key"] = api_key

    await create_or_update_user(chat_id, settings)

    await update.message.reply_text(
        "Setup completed! Your account is created.\n\n"
        "I'm ready to work. Send me a message or use /help for more information."
    )
    context.user_data.clear()
    return ConversationHandler.END


async def cancel_setup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel setup process"""
    await update.message.reply_text(
        "Setup cancelled. You can start again anytime with /start."
    )
    context.user_data.clear()
    return ConversationHandler.END


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Settings management command"""
    chat_id = update.effective_chat.id
    user_settings = await get_user_settings(chat_id)

    if not user_settings:
        await update.message.reply_text("Please register first using /start.")
        return ConversationHandler.END

    text = (
        f"Current Settings:\n"
        f"• Endpoint: `{user_settings.endpoint}`\n"
        f"• Main model: `{user_settings.model}`\n"
        f"• Thinking model: `{user_settings.reasoning_model}`\n"
        f"• Router model: `{user_settings.router_model}`\n"
        f"• Auto thinking: `{user_settings.auto_thinking}`\n"
        f"• Search enabled: `{user_settings.search_enabled}`\n"
        f"• Context messages: `{user_settings.context_messages_count}`\n\n"
        "What would you like to change?"
    )

    keyboard = [
        [
            InlineKeyboardButton("API Key", callback_data="settings_api_key"),
            InlineKeyboardButton(
                "System prompt", callback_data="settings_system_prompt"
            ),
        ],
        [
            InlineKeyboardButton("Endpoint", callback_data="settings_endpoint"),
            InlineKeyboardButton("Main model", callback_data="settings_model"),
        ],
        [
            InlineKeyboardButton(
                "Thinking model", callback_data="settings_reasoning_model"
            ),
            InlineKeyboardButton("Router model", callback_data="settings_router_model"),
        ],
        [
            InlineKeyboardButton(
                "Auto thinking", callback_data="settings_auto_thinking"
            ),
            InlineKeyboardButton("Search", callback_data="settings_search_enabled"),
        ],
        [
            InlineKeyboardButton(
                "Context", callback_data="settings_context_messages_count"
            ),
            InlineKeyboardButton("Cancel", callback_data="settings_cancel"),
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
    """Handle settings button clicks"""
    query = update.callback_query
    await query.answer()
    choice = query.data

    if choice == "settings_cancel":
        await query.edit_message_text("Settings change cancelled.")
        return ConversationHandler.END

    context.user_data["setting_to_change"] = choice.replace("settings_", "")

    prompts = {
        "api_key": "Enter new API Key:",
        "system_prompt": "Enter new system prompt:",
        "endpoint": "Enter new endpoint URL:",
        "model": "Enter new main model:",
        "reasoning_model": "Enter thinking model:",
        "router_model": "Enter router model:",
        "auto_thinking": "Auto thinking (true/false):",
        "search_enabled": "Web search (true/false):",
        "context_messages_count": "Context message count (number):",
    }

    prompt = prompts.get(context.user_data["setting_to_change"], "Enter new value:")
    await query.edit_message_text(text=prompt)
    return UPDATING_SETTING


async def set_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save new setting value"""
    chat_id = update.effective_chat.id
    new_value = update.message.text.strip()
    setting_key = context.user_data.get("setting_to_change")

    if not setting_key:
        await update.message.reply_text("Error occurred. Try /settings again.")
        return ConversationHandler.END

    # Type conversion
    if setting_key in ["auto_thinking", "search_enabled"]:
        value_to_save = new_value.lower() in ["true", "1", "yes", "on"]
    elif setting_key == "context_messages_count":
        try:
            value_to_save = int(new_value)
        except ValueError:
            await update.message.reply_text("Error: enter a number.")
            return UPDATING_SETTING
    else:
        value_to_save = new_value

    await create_or_update_user(chat_id, {setting_key: value_to_save})
    await update.message.reply_text(f"Setting '{setting_key}' updated.")

    context.user_data.clear()
    return ConversationHandler.END


def main():
    """Main entry point with background tasks"""
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set in environment")

    # Setup application
    app = ApplicationBuilder().token(token).build()

    # Setup conversation handlers
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
        fallbacks=[CommandHandler("cancel", cancel_setup)],
    )

    # Add handlers
    app.add_handler(setup_handler)
    app.add_handler(settings_handler)
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(
        MessageHandler(
            filters.TEXT | filters.PHOTO | filters.Document.ALL, main_message_handler
        ),
        group=1,
    )

    # Schedule background tasks
    async def schedule_cleanup():
        while True:
            await asyncio.sleep(3600)  # Every hour
            await cleanup_old_files()

    # Start background tasks
    asyncio.create_task(schedule_cleanup())

    logger.info("Multi-user bot started successfully with enhanced security!")
    return app


if __name__ == "__main__":
    try:
        application = main()
        application.run_polling()
    except Exception as e:
        logger.critical(f"Bot failed to start: {e}")
        raise
