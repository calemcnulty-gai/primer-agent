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
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.functions.schema import FunctionSchema, ParameterSchema

# Load environment variables
load_dotenv(override=True)

async def generate_image(text: str) -> str:
    """Generate a small image using OpenAI's DALL-E model optimized for speed."""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.openai.com/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "dall-e-2",  # Using DALL-E 2 for faster generation
                "prompt": f"A simple, colorful illustration of: {text}",
                "n": 1,
                "size": "512x512",
            }
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        logger.error("DALL-E generation failed with status: {}", resp.status)
                        return ""  # Silently fail
                    data = await resp.json()
                    return data["data"][0]["url"] if data.get("data") else ""
            except Exception as e:
                logger.error("DALL-E inner exception: {}", e)
                return ""  # Silently fail inner exceptions
    except Exception as e:
        logger.error("DALL-E outer exception: {}", e)
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

async def websocket_listener(ws_url: str, text_queue: asyncio.Queue, image_queue: asyncio.Queue):
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
            # Handle text messages
            if not text_queue.empty():
                text = await text_queue.get()
                if ws:
                    try:
                        await ws.send(json.dumps({"type": "llm-text", "data": text}))
                        logger.info("Sent to WebSocket: {}", text)
                    except Exception as e:
                        logger.error("WebSocket text send failed: {}", e)
                        ws = None
                text_queue.task_done()
            
            # Handle image messages
            if not image_queue.empty():
                image_prompt = await image_queue.get()
                if ws:
                    try:
                        logger.info("Generating image for prompt: {}", image_prompt)
                        image_url = await generate_image(image_prompt)
                        if image_url:
                            await ws.send(json.dumps({"type": "llm-image", "data": image_url}))
                            logger.info("Sent image to WebSocket: {}", image_url)
                    except Exception as e:
                        logger.error("WebSocket image send failed: {}", e)
                        ws = None
                image_queue.task_done()
            
            # Check connection state
            if ws is None:
                current_time = asyncio.get_event_loop().time()
                if current_time - last_reconnect_attempt >= reconnect_interval:
                    await connect()
            
            # Small sleep to prevent busy-waiting
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("Error in websocket_listener: {}", e)
            await asyncio.sleep(1)

async def main(room_url: str, token: str):
    logger.info("Starting bot in room: {}", room_url)

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
    image_queue = asyncio.Queue()

    async def on_tts_text(text):
        await text_queue.put(text)

    tts = CustomTTService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
        callback=on_tts_text
    )

    # Define function schema for image generation
    generate_image_schema = FunctionSchema(
        name="generate_image",
        description="Generate an image based on a text description",
        parameters=[
            ParameterSchema(
                name="prompt",
                description="A detailed description of the image to generate",
                type="string",
                required=True
            )
        ]
    )

    # Define function handler for image generation
    async def handle_generate_image(prompt: str):
        logger.info("Function called: generate_image with prompt: {}", prompt)
        await image_queue.put(prompt)
        return {
            "success": True,
            "message": "Image generation has been requested and will be displayed shortly."
        }

    # Create LLM service with function calling capability
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model="gpt-4o",
        functions=[generate_image_schema],
        function_handlers={"generate_image": handle_generate_image}
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
                "End every response with the exact string ' {}'. So if your response was, 'how old are you?' you would return 'How old are you? {}'"
                "\n\nYou have the ability to generate images to illustrate your stories and explanations. "
                "When appropriate, use the generate_image function to create visual aids that complement your narrative. "
                "This is especially useful when describing new concepts, characters in stories, or complex ideas."
            )
        }
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    logger.info("Creating pipeline with components")
    pipeline = Pipeline([
        transport.input(),
        # LoggingProcessor(name="Input"),  # Log frames after input
        context_aggregator.user(),
        # LoggingProcessor(name="Pre-LLM"),  # Log what's going into the LLM
        llm,
        # LoggingProcessor(name="Post-LLM"),  # Log what's coming out of the LLM
        tts,
        # LoggingProcessor(name="Pre-Output"),  # Log what's about to be output
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
    logger.info("Pipeline task created with allow_interruptions=True")
    logger.info("Note: If experiencing unwanted interruptions, you may want to set allow_interruptions=False")

    asyncio.create_task(websocket_listener("ws://primer.calemcnulty.com:8081", text_queue, image_queue))

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
            logger.info("Added user message to context: {}", message["text"])
            logger.info("Current message history length: {}", len(messages))
            
            # Log the complete conversation context for debugging
            logger.debug("Complete conversation context:")
            for idx, msg in enumerate(messages):
                logger.debug("[{}] {}: {}", idx, msg["role"], msg["content"])
            
            # Wait 1.5 seconds after last speech to ensure user is done (increased from 0.5)
            logger.info("Waiting for user to complete speaking...")
            while True:
                current_time = asyncio.get_event_loop().time()
                wait_time = current_time - last_speech_time
                if wait_time >= 1.5:  # Increased from 0.5 to give more time
                    logger.info("User speech pause detected after {:.2f}s, queuing message for LLM", wait_time)
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

class LoggingProcessor(FrameProcessor):
    """Custom processor for detailed logging of frames flowing through the pipeline."""
    
    def __init__(self, name="Pipeline"):
        super().__init__()
        self.name = name
        
    async def process_frame(self, frame, direction):
        """Log detailed information about frames moving through the pipeline."""
        logger.info("{} Processor: received frame of type {} in direction {}", 
                   self.name, type(frame).__name__, direction)
        
        # Log frame content based on type
        if hasattr(frame, "text"):
            logger.info("{} Frame text content: {}", self.name, frame.text)
        elif hasattr(frame, "messages"):
            logger.info("{} Frame contains {} messages", self.name, len(frame.messages))
            # Log the messages in the frame
            for i, msg in enumerate(frame.messages):
                logger.info("{} Message {}: {} - {}", 
                           self.name, i, msg.get("role", "unknown"), 
                           msg.get("content", "")[:50] + ("..." if len(msg.get("content", "")) > 50 else ""))
        else:
            logger.info("{} Unknown frame format", self.name)
            
        # Continue processing the frame
        return await super().process_frame(frame, direction)

if __name__ == "__main__":
    import asyncio
    args = DailySessionArguments(
        room_url="https://your-daily-room-url.daily.co/room",
        token="your-token-here",
        session_id="test-session"
    )
    asyncio.run(bot(args))