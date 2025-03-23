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
from pipecat.frames.frames import LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
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

class WebSocketSender(FrameProcessor):
    """Processor to send data via WebSocket."""
    def __init__(self, ws_url: str):
        super().__init__()
        self._ws_url = ws_url
        self._websocket = None

    async def start(self):
        """Establish WebSocket connection."""
        try:
            self._websocket = await websockets.connect(self._ws_url)
            logger.info("WebSocket connected to {}", self._ws_url)
        except Exception as e:
            logger.error("Failed to connect to WebSocket: {}", e)

    async def process_frame(self, frame, direction):
        if hasattr(frame, "text") and self._websocket:
            text_message = json.dumps({"type": "llm-text", "data": frame.text})
            try:
                await self._websocket.send(text_message)
                logger.debug("Sent LLM text via WebSocket: {}", frame.text)
            except Exception as e:
                logger.error("WebSocket text send failed: {}", e)

            image_url = await generate_image(frame.text)
            if image_url:
                image_message = json.dumps({"type": "llm-image", "data": image_url})
                try:
                    await self._websocket.send(image_message)
                    logger.debug("Sent LLM image via WebSocket: {}", image_url)
                except Exception as e:
                    logger.error("WebSocket image send failed: {}", e)

        return await super().process_frame(frame, direction)

    async def cleanup(self):
        """Close WebSocket connection."""
        if self._websocket:
            await self._websocket.close()
            logger.info("WebSocket closed")

async def clear_room_except_bot(room_name: str, bot_id: str):
    """Clear all participants except the bot from the room using Daily API."""
    api_key = os.getenv("DAILY_API_KEY")
    if not api_key:
        logger.error("DAILY_API_KEY not set in environment variables")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"https://api.daily.co/v1/rooms/{room_name}/participants",
                headers=headers
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to list participants: {await resp.text()}")
                    return
                participants = await resp.json()
        except Exception as e:
            logger.error(f"Error listing participants: {str(e)}")
            return

        for participant in participants.get("data", []):
            participant_id = participant.get("id")
            if participant_id != bot_id:
                try:
                    async with session.post(
                        f"https://api.daily.co/v1/rooms/{room_name}/participants/{participant_id}/eject",
                        headers=headers
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"Ejected participant: {participant_id}")
                        else:
                            logger.error(f"Failed to eject {participant_id}: {await resp.text()}")
                except Exception as e:
                    logger.error(f"Error ejecting participant {participant_id}: {str(e)}")

async def main(room_url: str, token: str):
    """Main pipeline setup and execution function."""
    logger.debug("Starting bot in room: {}", room_url)

    # Extract room name from URL
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

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"), voice_id=os.getenv("CARTESIA_VOICE_ID")
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

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
                "When you receive 'User has joined', say 'Hello! I’m the Young Lady’s Illustrated Primer, here to share stories and adventures. What’s your name, and how old are you?' "
                "Tailor stories to their age and interests. "
                "Remember previous interactions in this session to maintain context and personalize responses. "
                "Keep output simple for audio, avoiding special characters."
            )
        }
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    ws_sender = WebSocketSender("ws://primer.calemcnulty.com:8081")
    await ws_sender.start()

    # Set up event handlers
    bot_id = None
    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        nonlocal bot_id
        logger.info("on_joined data: {}", data)
        if "participants" in data and "local" in data["participants"]:
            bot_id = data["participants"]["local"]["id"]
            logger.info("Bot joined with ID: {}", bot_id)
            await clear_room_except_bot(room_name, bot_id)
        else:
            logger.error("Unable to find bot participant ID in on_joined data")

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        if participant["id"] != bot_id:
            logger.info("Client joined: {}", participant["id"])
            await transport.capture_participant_transcription(participant["id"])  # Ensure transcription is captured
            initial_message = {"role": "user", "content": "User has joined"}
            messages.append(initial_message)
            await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info("Participant left: {}", participant["id"])
        logger.debug("Pipeline remains active")

    @transport.event_handler("on_transcription_message")
    async def on_transcription_message(transport, message):
        logger.debug("Transcription received: {}", message)
        if "text" in message and message["participant_id"] != bot_id:
            user_message = {"role": "user", "content": message["text"]}
            messages.append(user_message)
            await task.queue_frames([LLMMessagesFrame(messages)])

    # Create and start the pipeline immediately
    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    runner = PipelineRunner()
    await runner.run(task)

async def bot(args: DailySessionArguments):
    """Main bot entry point compatible with the FastAPI route handler."""
    logger.info(f"Bot process initialized with room {args.room_url} and token {args.token}")
    try:
        await main(args.room_url, args.token)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise