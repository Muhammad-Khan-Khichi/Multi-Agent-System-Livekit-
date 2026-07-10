import logging

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, cli, inference

from agents.checkout import Checkout
from agents.greeter import Greeter
from agents.reservation import Reservation
from agents.takeaway import Takeaway
from config import MENU
from userData import UserData

logger = logging.getLogger("restaurant-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(MENU),
            "reservation": Reservation(),
            "takeaway": Takeaway(MENU),
            "checkout": Checkout(MENU),
        }
    )
    session = AgentSession[UserData](
        userdata=userdata,
        stt=inference.STT(model="deepgram/nova-3"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(model="cartesia/sonic-3"),
        max_tool_steps=5,
    )

    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
    )

    # await agent.say("Welcome to our restaurant! How may I assist you today?")


if __name__ == "__main__":
    cli.run_app(server)