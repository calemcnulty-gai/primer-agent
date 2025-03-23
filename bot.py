#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import aiohttp
import websockets
import asyncio
import json
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecatcloud.agent import DailySessionArguments

# Load environment variables
load_dotenv(override=True)

async def generate_image(text: str) -> str:
    """Generate a small image using Replicate's Flux.1 Schnell model, silently ignoring errors."""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.replicate.com/v1/predictions"
            headers = {
                "Authorization": f"Token {os.getenv('REPLICATE_API_TOKEN')}",
                "Content-Type": "application/json",
            }
            payload = {
                "version": "lucataco/flux.1schnell-uncensored-rasch3",
                "input": {
                    "prompt": f"A simple, colorful illustration of: {text}",
                    "width": 256,
                    "height": 256,
                    "num_outputs": 1,
                    "num_inference_steps": 4
                }
            }
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 201:
                        return ""  # Silently fail
                    data = await resp.json()
                    prediction_id = data["id"]

                while True:
                    async with session.get(f"{url}/{prediction_id}", headers=headers) as resp:
                        result = await resp.json()
                        if result["status"] in ["succeeded", "failed"]:
                            return result.get("output", [""])[0] if result["status"] == "succeeded" else ""
            except Exception:
                return ""  # Silently fail inner exceptions
    except Exception:
        return ""  # Silently fail outer exceptions

class CustomTTService(CartesiaTTSService):
    """Custom TTS service that emits text events before generating audio."""
    def __init__(self, *args, callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback

    async def run_tts(self, text: str):
        """Override to capture text and yield audio frames."""
        logger.debug("CustomTTSService processing text: {}", text)
        if self._callback:
            logger.info("Emitting text event: {}", text)
            await self._callback(text)
        async for frame in super().run_tts(text):
            yield frame  # Yield each audio frame downstream

async def websocket_listener(ws_url: str, text_queue: asyncio.Queue):
    """Listen for text events and send them to WebSocket."""
    ws = None
    last_reconnect_attempt = 0
    reconnect_interval = 5  # seconds

    logger.info("Starting WebSocket listener for {}", ws_url)

    async def connect():
        nonlocal ws, last_reconnect_attempt
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info("Connecting to WebSocket at {} (attempt {}/{})", ws_url, attempt, max_attempts)
                ws = await websockets.connect(ws_url)
                await ws.send(json.dumps({"type": "ping", "data": "Connection test from bot"}))
                logger.info("WebSocket connected successfully to {}", ws_url)
                return True
            except Exception as e:
                logger.error("WebSocket connection attempt {}/{} failed: {}", attempt, max_attempts, e)
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        logger.error("Failed to connect to WebSocket after {} attempts", max_attempts)
        last_reconnect_attempt = asyncio.get_event_loop().time()
        return False

    await connect()

    while True:
        try:
            text = await text_queue.get()
            if ws:
                try:
                    await ws.send(json.dumps({"type": "llm-text", "data": text}))
                    logger.info("Sent to WebSocket: {}", text)
                    image_url = await generate_image(text)
                    if image_url:
                        await ws.send(json.dumps({"type": "llm-image", "data": image_url}))
                        logger.info("Sent image to WebSocket: {}", image_url)
                except Exception as e:
                    logger.error("WebSocket send failed: {}", e)
                    ws = None
            else:
                current_time = asyncio.get_event_loop().time()
                if current_time - last_reconnect_attempt >= reconnect_interval:
                    if await connect():
                        continue
            text_queue.task_done()
        except Exception as e:
            logger.error("Error in websocket_listener: {}", e)
            await asyncio.sleep(1)

async def main(room_url: str, token: str):
    logger.info("Starting bot in room: {}", room_url)

    room_name = room_url.split("/")[-1]

    transport = DailyTransport(
        room_url,
        token,
        "bot",
        DailyParams(
            audio_out_enabled=True,
            transcription_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    text_queue = asyncio.Queue()

    async def on_tts_text(text):
        await text_queue.put(text)

    tts = CustomTTService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
        callback=on_tts_text
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model="gpt-4o"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Young Lady's Illustrated Primer, a friendly and wise companion designed to educate and inspire through voice. "
                "Your goals are: "
                "1. Craft short, engaging stories (2-3 sentences) based on the user's interests or learning needs, embedding educational content. "
                "2. Ask interactive questions or present challenges within stories, adjusting based on responses. "
                "3. Guide the user conversationally with encouragement and explanations. "
                "4. The user should be challenged but not overwhelmed. Don't give too much help, but give them hints if they need it. "
                "When you receive 'User has joined', say 'Hello! I'm the Young Lady's Illustrated Primer, here to share stories and adventures. What's your name, and how old are you?' "
                "Tailor stories to their age and interests. "
                "Remember previous interactions in this session to maintain context and personalize responses. "
                "Keep output simple for audio, avoiding special characters."
            )
        }
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    logger.info("Creating pipeline with components")
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    logger.info("Pipeline created successfully")

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    asyncio.create_task(websocket_listener("ws://primer.calemcnulty.com:8081", text_queue))

    bot_id = None
    last_speech_time = None

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        nonlocal bot_id
        logger.info("on_joined data: {}", data)
        if "participants" in data and "local" in data["participants"]:
            bot_id = data["participants"]["local"]["id"]
            logger.info("Bot joined with ID: {}", bot_id)
        else:
            logger.error("Unable to find bot participant ID in on_joined data")

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        if participant["id"] != bot_id:
            logger.info("Client joined: {}", participant["id"])
            await transport.capture_participant_transcription(participant["id"])
            initial_message = {"role": "user", "content": "User has joined"}
            messages.append(initial_message)
            logger.info("Queuing initial greeting message: {}", initial_message)
            await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info("Participant left: {}", participant["id"])
        logger.info("Pipeline remains active")

    @transport.event_handler("on_transcription_message")
    async def on_transcription_message(transport, message):
        nonlocal last_speech_time
        logger.info("Transcription received: {}", message)
        if "text" in message and message.get("participantId") != bot_id:
            # Update last speech time when user speaks
            last_speech_time = asyncio.get_event_loop().time()
            user_message = {"role": "user", "content": message["text"]}
            messages.append(user_message)
            logger.info("Queuing user message: {}", message["text"])
            # Wait 0.5 seconds after last speech to ensure user is done
            while True:
                current_time = asyncio.get_event_loop().time()
                if current_time - last_speech_time >= 0.5:
                    await task.queue_frames([LLMMessagesFrame(messages)])
                    break
                await asyncio.sleep(0.1)  # Check every 0.1s

    runner = PipelineRunner()
    logger.info("Starting pipeline runner")
    await runner.run(task)

async def bot(args: DailySessionArguments):
    logger.info(f"Bot process initialized with room {args.room_url} and token {args.token}")
    try:
        await main(args.room_url, args.token)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    args = DailySessionArguments(
        room_url="https://your-daily-room-url.daily.co/room",
        token="your-token-here",
        session_id="test-session"
    )
    asyncio.run(bot(args))