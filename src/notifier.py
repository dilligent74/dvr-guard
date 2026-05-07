"""
Telegram notifier module for DVR Guard.
Stubbed version: prints to console instead of sending Telegram messages.
Runs a separate thread with a queue for non-blocking alerts.
"""

import threading
import queue
import logging
import requests
import json
import os
import asyncio
from datetime import datetime
from state import Detection
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, ConversationHandler, MessageHandler, filters
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup

# Conversation states
PASSWORD = 1

logger = logging.getLogger(__name__)


# Path to subscriptions file (next to config.yaml)
SUBSCRIPTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "subscriptions.json"
)

_subscribers_lock = threading.Lock()


def load_subscribers():
    """Load subscribers from subscriptions.json, return empty dict if not exists."""
    if not os.path.exists(SUBSCRIPTIONS_PATH):
        # Create empty subscriptions file
        save_subscribers({})
        return {}
    try:
        with open(SUBSCRIPTIONS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load subscribers: {e}")
        return {}


def save_subscribers(subscribers):
    """Save subscribers dict to subscriptions.json."""
    try:
        with open(SUBSCRIPTIONS_PATH, "w") as f:
            json.dump(subscribers, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save subscribers: {e}")


class TelegramNotifier:
    """
    Telegram notifier with a dedicated queue/thread for non-blocking alerts.
    Uses Telegram Bot API to send text + photo alerts.
    """

    def __init__(self, token: str = None, chat_id: str = None, subscriber_password: str = None):
        self._token = token
        self._chat_id = chat_id
        self._subscriber_password = subscriber_password
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._loop = None  # Polling thread's event loop (set in _polling_worker)
        self._application = None  # PTB Application instance (set in _polling_worker)
        self._polling_lock = threading.Lock()  # Protects _application and _loop access
        
        # Initialize subscriptions file and auto-add admin if needed
        self._init_subscriptions()
        
        self._thread = threading.Thread(
            target=self._worker,
            name="TelegramNotifier",
            daemon=True
        )
        self._thread.start()
        # Start polling thread for bot commands
        self._polling_thread = threading.Thread(
            target=self._polling_worker,
            name="TelegramPolling",
            daemon=True
        )
        self._polling_thread.start()
        if token and chat_id:
            logger.info("TelegramNotifier initialized (real Telegram)")
        else:
            logger.warning("TelegramNotifier initialized WITHOUT credentials - alerts will be logged only")

    def send_alert(self, detection: Detection) -> None:
        """
        Non-blocking alert dispatch.
        Puts the detection into the queue for async processing.
        """
        if not self._token:
            logger.warning("Cannot send alert: Telegram token not configured")
            # Still queue it so it gets logged
        self._queue.put(detection)
        logger.debug(f"Queued alert for {detection.camera_name}")

    def _worker(self) -> None:
        """Worker thread that processes queued alerts."""
        while not self._stop_event.is_set():
            try:
                detection = self._queue.get(timeout=1.0)
                try:
                    self._process_alert(detection)
                except Exception as e:
                    logger.error(f"Error processing alert: {e}")
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    def _process_alert(self, detection: Detection) -> None:
        """
        Process alert: send Telegram message with optional photo.
        Falls back to console output if Telegram not configured.
        """
        timestamp_str = detection.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Build message text
        message = (
            f"🚨 Persoană Detectată!\n"
            f"Camera: {detection.camera_name}\n"
            f"Time: {timestamp_str}\n"
            f"Probabilitate: {detection.confidence:.1%}"
        )
        
        # Log to console always
        logger.info(f"Alert: {message.replace(chr(10), ' | ')}")
        
        # Send via Telegram if configured
        if self._token:
            with _subscribers_lock:
                subscribers = load_subscribers()
            active_subs = [chat_id for chat_id, sub in subscribers.items() if sub.get("active", False)]
            if not active_subs:
                logger.debug("No active subscribers to send alert to.")
            for chat_id in active_subs:
                try:
                    self._send_telegram_message(chat_id, message, detection.snapshot_path)
                except Exception as e:
                    logger.error(f"Failed to send Telegram alert to {chat_id}: {e}")
        else:
            # Fallback console output
            print("\n" + "=" * 50)
            print(f"🚨 PERSOANĂ DETECTATĂ 🚨")
            print(f"Camera: {detection.camera_name} (ID: {detection.camera_id})")
            print(f"Time: {timestamp_str}")
            print(f"Probabilitate: {detection.confidence:.2%}")
            print(f"Bounding Box: {detection.bbox}")
            if detection.snapshot_path:
                print(f"Snapshot: {detection.snapshot_path}")
            print("=" * 50 + "\n")

    def _send_telegram_message(self, chat_id: str, text: str, photo_path: str = None) -> None:
        """Send message (and optional photo) via Telegram Bot API to a specific chat."""
        base_url = f"https://api.telegram.org/bot{self._token}"
        
        if photo_path and os.path.exists(photo_path):
            # Send photo with caption
            with open(photo_path, "rb") as photo:
                response = requests.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text},
                    files={"photo": photo},
                    timeout=10,
                )
        else:
            if photo_path and not os.path.exists(photo_path):
                logger.warning(f"Snapshot file no longer exists, sending text-only alert: {photo_path}")
            # Send text message only
            response = requests.post(
                f"{base_url}/sendMessage",
                data={"chat_id": chat_id, "text": text},
                timeout=10,
            )
        
        if response.status_code != 200:
            raise Exception(
                f"Telegram API error: {response.status_code} - {response.text}"
            )
        
        logger.debug("Telegram alert sent successfully")

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command: begin conversation flow."""
        chat_id = str(update.effective_chat.id)
        
        # Check if subscriber password is configured
        if not self._subscriber_password:
            logger.error("Subscriber password not configured in config.yaml!")
            await update.message.reply_text("❌ Configuration error.")
            return ConversationHandler.END
        
        with _subscribers_lock:
            subscribers = load_subscribers()

        if chat_id in subscribers:
            # Returning user: show welcome back and control message
            await update.message.reply_text("Bine ai revenit!")
            await self._show_control_message(update, context, chat_id, subscribers)
            return ConversationHandler.END
        else:
            # New user: ask for password
            await update.message.reply_text("Bine ai venit! Introdu parola:")
            return PASSWORD
    
    async def _password_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle password input from user."""
        chat_id = str(update.effective_chat.id)
        provided_password = update.message.text
        
        if provided_password != self._subscriber_password:
            await update.message.reply_text("❌ Parolă invalidă.")
            return ConversationHandler.END
        
        # Password correct: authorize user
        with _subscribers_lock:
            subscribers = load_subscribers()
            subscribers[chat_id] = {
                "active": True,
                "control_message_id": None
            }
            save_subscribers(subscribers)
        
        await update.message.reply_text("✅ Access permis")
        await self._show_control_message(update, context, chat_id, subscribers)
        return ConversationHandler.END
    
    async def _show_control_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id, subscribers):
        """Show or recreate control message for user."""
        # Always recreate control message: first clean up any previous one using saved ID
        old_msg_id = subscribers[chat_id].get("control_message_id")
        if old_msg_id is not None:
            try:
                await context.bot.unpin_chat_message(chat_id=int(chat_id), message_id=old_msg_id)
            except Exception:
                pass  # Message may have been manually deleted or chat cleared
            try:
                await context.bot.delete_message(chat_id=int(chat_id), message_id=old_msg_id)
            except Exception:
                pass  # Message may have been manually deleted or chat cleared
        
        # Send control message with inline button based on current active state
        active = subscribers[chat_id].get("active", True)
        if active:
            msg_text = "🔔 Notificările sunt PORNITE"
            button_text = "🔕 OPREȘTE"
        else:
            msg_text = "🔕 Notificările sunt OPRITE"
            button_text = "🔔 PORNEȘTE"
        
        keyboard = [[InlineKeyboardButton(button_text, callback_data="toggle")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        sent_message = await update.message.reply_text(
            msg_text,
            reply_markup=reply_markup
        )
        
        # Pin the control message
        await context.bot.pin_chat_message(
            chat_id=int(chat_id),
            message_id=sent_message.message_id
        )
        
        # Save control message ID
        with _subscribers_lock:
            subscribers[chat_id]["control_message_id"] = sent_message.message_id
            save_subscribers(subscribers)

    def _init_subscriptions(self):
        """Initialize subscriptions file and auto-add admin if needed."""
        with _subscribers_lock:
            subscribers = load_subscribers()
            if self._chat_id and self._chat_id not in subscribers:
                subscribers[self._chat_id] = {
                    "active": True,
                    "control_message_id": None
                }
                save_subscribers(subscribers)
                logger.info(f"Auto-added admin chat_id {self._chat_id} to subscribers")
    
    async def _toggle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle toggle callback to enable/disable notifications."""
        query = update.callback_query
        await query.answer()  # Answer the query first
        chat_id = str(query.message.chat_id)
        with _subscribers_lock:
            subscribers = load_subscribers()
            if chat_id not in subscribers:
                logger.warning(f"Toggle request from non-subscribed user {chat_id}")
                return
            # Flip active status
            subscribers[chat_id]["active"] = not subscribers[chat_id]["active"]
            save_subscribers(subscribers)
        
        # Prepare new message content
        active = subscribers[chat_id]["active"]
        if active:
            new_text = "🔔 Notificările sunt PORNITE"
            button_text = "🔕 OPREȘTE"
        else:
            new_text = "🔕 Notificările sunt OPRITE"
            button_text = "🔔 PORNEȘTE"
        
        # Edit the pinned control message, or send fresh if needed
        control_msg_id = subscribers[chat_id].get("control_message_id")
        if control_msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=int(chat_id),
                    message_id=control_msg_id,
                    text=new_text,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(button_text, callback_data="toggle")]])
                )
            except Exception as e:
                logger.warning(f"Failed to edit control message for {chat_id}, sending fresh: {e}")
                # Message was probably deleted, send a fresh one
                await self._send_fresh_control(context.bot, chat_id, active, new_text, button_text, subscribers)
        else:
            # No control message ID, send a fresh one
            await self._send_fresh_control(context.bot, chat_id, active, new_text, button_text, subscribers)
    
    async def _send_fresh_control(self, bot, chat_id, active, text, button_text, subscribers):
        """Send a fresh control message, pin it, and save the message ID."""
        try:
            keyboard = [[InlineKeyboardButton(button_text, callback_data="toggle")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            sent_message = await bot.send_message(
                chat_id=int(chat_id),
                text=text,
                reply_markup=reply_markup
            )
            # Pin the control message
            await bot.pin_chat_message(
                chat_id=int(chat_id),
                message_id=sent_message.message_id
            )
            # Save control message ID
            with _subscribers_lock:
                subscribers[chat_id]["control_message_id"] = sent_message.message_id
                save_subscribers(subscribers)
        except Exception as e:
            logger.error(f"Failed to send fresh control message for {chat_id}: {e}")

    def _polling_worker(self):
        """Worker thread for Telegram bot polling (command handling)."""
        if not self._token:
            logger.warning("No Telegram token provided, polling thread not started")
            return
        
        try:
            logger.info("Starting Telegram polling thread...")
            
            # Create and set event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Build Application and store it for external stop signaling
            application = Application.builder().token(self._token).build()
            
            # Store references with lock protection
            with self._polling_lock:
                self._loop = loop
                self._application = application
            
            # Create conversation handler for /start flow
            conv_handler = ConversationHandler(
                entry_points=[CommandHandler("start", self._start_command)],
                states={
                    PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, self._password_handler)]
                },
                fallbacks=[]
            )
            application.add_handler(conv_handler)
            application.add_handler(CallbackQueryHandler(self._toggle_callback, pattern="^toggle$"))
            
            # Run polling until application.stop() is called
            # run_until_complete will unblock once run_polling() exits
            loop.run_until_complete(application.run_polling(stop_signals=None))
            
        except Exception as e:
            logger.error(f"Polling thread error: {e}")
        finally:
            # Cleanup: close the event loop (only after run_until_complete returns)
            with self._polling_lock:
                self._application = None
                loop = self._loop
                self._loop = None
            if loop:
                try:
                    loop.close()
                except Exception as e:
                    logger.error(f"Error closing polling event loop: {e}")
            logger.info("Telegram polling thread stopped")

    def stop(self) -> None:
        """Stop the notifier worker thread and polling thread."""
        logger.info("Stopping TelegramNotifier...")

        # Drain the alert queue before signalling stop so queued detections are sent.
        # Run join() in a daemon thread so we can cap the wait at 5 s.
        drain_thread = threading.Thread(target=self._queue.join, daemon=True)
        drain_thread.start()
        drain_thread.join(timeout=5.0)
        if drain_thread.is_alive():
            logger.warning("Alert queue drain timed out after 5s — some alerts may not have been sent")

        self._stop_event.set()
        
        # Signal polling application to stop from main thread (cross-thread async call)
        with self._polling_lock:
            application = self._application
            loop = self._loop
        
        if application and loop:
            try:
                asyncio.run_coroutine_threadsafe(application.stop(), loop)
                logger.debug("Signaled polling application to stop")
            except Exception as e:
                logger.error(f"Failed to signal polling stop: {e}")
        
        # Wait for threads to exit (polling thread will exit once run_until_complete returns)
        self._thread.join(timeout=5.0)
        self._polling_thread.join(timeout=5.0)
        logger.info("TelegramNotifier stopped")
