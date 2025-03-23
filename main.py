import os
import asyncio
from dotenv import load_dotenv
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.network.websocket_server import WebsocketServerTransport, WebsocketServerParams
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.processors.frame_processor import FrameProcessor
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = [
    'GEMINI_API_KEY',
    'DEEPGRAM_API_KEY',
    'CARTESIA_API_KEY',
    'CARTESIA_MODEL_ID',
    'CARTESIA_DEFAULT_VOICE_ID'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class CustomFrameSerializer(FrameSerializer):
    def serialize(self, frame):
        if hasattr(frame, 'text'):
            return {'type': 'text', 'content': frame.text}
        elif hasattr(frame, 'audio'):
            return {'type': 'audio', 'content': frame.audio}
        return {'type': 'unknown', 'content': str(frame)}

    def deserialize(self, data):
        if data['type'] == 'text':
            return self.create_frame(text=data['content'])
        elif data['type'] == 'audio':
            return self.create_frame(audio=data['content'])
        return self.create_frame(text=str(data['content']))

    def type(self):
        return 'custom'

def get_current_time():
    """Get the current time"""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

# Custom LangChain Agent Processor
class LangChainAgent(FrameProcessor):
    def __init__(self):
        super().__init__()
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Define tools
        tools = [
            Tool(
                name="get_current_time",
                func=get_current_time,
                description="Get the current time"
            )
        ]
        
        self.agent = initialize_agent(
            llm=llm,
            tools=tools,
            agent_type="zero-shot-react-description"
        )

    async def process_frame(self, frame):
        if hasattr(frame, 'text'):  # STT output
            try:
                response = self.agent.run(frame.text)
                return self.create_frame(text=response)
            except Exception as e:
                print(f"Error processing frame: {e}")
                return self.create_frame(text="I apologize, but I encountered an error processing your request.")
        return frame

async def main():
    try:
        # Initialize services with configuration
        params = WebsocketServerParams(
            add_wav_header=True,
            serializer=CustomFrameSerializer(),
            session_timeout=None
        )
        transport = WebsocketServerTransport(params)
        stt = DeepgramSTTService(api_key=os.getenv('DEEPGRAM_API_KEY'))
        agent = LangChainAgent()
        tts = CartesiaTTSService(
            api_key=os.getenv('CARTESIA_API_KEY'),
            model_id=os.getenv('CARTESIA_MODEL_ID'),
            voice_id=os.getenv('CARTESIA_DEFAULT_VOICE_ID')
        )
        
        # Create pipeline
        pipeline = Pipeline([transport.input(), stt, agent, tts, transport.output()])
        print("Starting voice AI assistant on ws://localhost:8765...")
        
        # Run the pipeline by processing frames indefinitely
        while True:
            await asyncio.sleep(0.1)  # Small delay to prevent CPU hogging
            
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())