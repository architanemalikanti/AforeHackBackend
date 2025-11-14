from dotenv import load_dotenv
import os, asyncio, json, logging
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from database.db import SessionLocal
from database.models import User
from agent import Agent
from prompt_manager import set_prompt
from redis_client import r

load_dotenv()

logger = logging.getLogger(__name__)

# Get absolute path to conversations database
DB_PATH = str(Path(__file__).parent / "conversations.db")

# --- Tools ---
from tools import (
    create_redis_session,
    set_username,
    set_password,
    confirm_password,
    get_email,
    get_user_birthday,
    get_user_gender,
    get_user_sexuality,
    get_user_ethnicity,
    get_user_pronouns,
    get_user_first_name,
    get_user_university,
    get_user_occupation,
    get_user_college_major,
    generate_verification_code,
    resend_verification_code,
    log_in_user
)
# Use the finalize_user version for background tasks
from finalize_user import test_verification_code

tool = TavilySearchResults(max_results=2)

# Collect all tools
all_tools = [
    tool,
    create_redis_session,
    set_username,
    set_password,
    confirm_password,
    get_email,
    get_user_birthday,
    get_user_gender,
    get_user_sexuality,
    get_user_ethnicity,
    get_user_pronouns,
    get_user_first_name,
    get_user_university,
    get_user_occupation,
    get_user_college_major,
    generate_verification_code,
    resend_verification_code,
    test_verification_code,  # From finalize_user - returns "verified"/"incorrect"
    log_in_user
]

# --- Model ---
model = ChatAnthropic(model="claude-sonnet-4-20250514")


# # --- Instantiate agent (sync) ---
# with SqliteSaver.from_conn_string(":memory:") as memory:
#     abot = Agent(model, [tool], system=prompt, checkpointer=memory)

#     messages = [HumanMessage(content="What's the weather in San Francisco?")]
#     thread = {"configurable": {"thread_id": "sync-1"}}

#     for event in abot.graph.stream({"messages": messages}, thread):
#         for v in event.values(): 
#             print(v["messages"])

# --- FastAPI app + SSE streaming endpoint ---
app = FastAPI()

@app.get("/chat/stream") 
async def chat_stream(q: str = Query(""), session_id: str = Query(...)):
    """
    SSE streaming endpoint for chat. 
    Query params:
    - q: user message
    - session_id: unique session identifier (required)
    """
    async def event_gen():
        import threading
        import time
        from finalize_user import finalize_user_background
        
        # Generate dynamic prompt based on current Redis state
        dynamic_prompt = set_prompt(session_id)
        
        messages = [HumanMessage(content=q)]
        thread = {"configurable": {"thread_id": session_id}}
        
        verification_succeeded = False
        background_task_started = False

        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as async_memory:
            async_abot = Agent(model, all_tools, system=dynamic_prompt, checkpointer=async_memory)
            async for ev in async_abot.graph.astream_events({"messages": messages}, thread, version="v1"):
                # Check if verification tool was called and succeeded
                if ev["event"] == "on_tool_end":
                    tool_name = ev.get("name", "")
                    tool_output = ev.get("data", {}).get("output", "")

                    if tool_name == "test_verification_code" and tool_output == "verified":
                        verification_succeeded = True
                        # Start background task immediately
                        if not background_task_started:
                            threading.Thread(
                                target=finalize_user_background,
                                args=(session_id,),
                                daemon=True
                            ).start()
                            background_task_started = True
                            logger.info(f"🚀 Started background finalization for {session_id}")
                
                # Stream LLM tokens
                if ev["event"] == "on_chat_model_stream":
                    content = ev["data"]["chunk"].content
                    if content:
                        yield f"event: token\ndata: {json.dumps({'content': content})}\n\n"

        # If verification succeeded, send onboarding_complete immediately
        if verification_succeeded:
            logger.info(f"✅ Sending onboarding_complete to iOS for session {session_id}")
            # Send onboarding complete signal with session_id
            # iOS will poll Redis for user_id using this session_id
            yield f"event: onboarding_complete\ndata: {json.dumps({'session_id': session_id})}\n\n"
            logger.info(f"✅ Sent onboarding_complete with session_id to iOS")

        yield "event: done\ndata: {}\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)



@app.get("/createRedisKey")
async def create_redis_key():
    """Generate a new unique session_id."""
    import uuid
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.get("/poll/{session_id}")
async def poll_user_id(session_id: str):
    """
    Poll endpoint for iOS to check if user_id is ready after verification.
    Returns user_id if available, or status if still processing.
    """
    try:
        redis_key = f"session:{session_id}"
        session_data_str = r.get(redis_key)

        if not session_data_str:
            return {"status": "not_found", "message": "Session not found"}

        session_data = json.loads(session_data_str)
        user_id = session_data.get("user_id")

        if user_id:
            # User ID is ready! Return user_id and JWT tokens
            return {
                "status": "ready",
                "user_id": user_id,
                "access_token": session_data.get("access_token"),
                "refresh_token": session_data.get("refresh_token"),
                "session_id": session_id
            }
        else:
            # Still processing
            return {
                "status": "processing",
                "message": "User verification in progress"
            }

    except Exception as e:
        logger.error(f"Error polling for user_id: {str(e)}")
        return {"status": "error", "error": str(e)}

@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """
    Delete Redis session AND SQLite checkpoints after iOS has retrieved the user_id.
    Called by iOS after polling and getting user_id.
    """
    import sqlite3
    from pathlib import Path

    try:
        # 1. Check if Redis session exists and get metadata
        redis_key = f"session:{session_id}"
        session_data_str = r.get(redis_key)

        if not session_data_str:
            logger.warning(f"⚠️  Session {session_id} not found in Redis")
            return {"status": "not_found", "session_id": session_id}

        session_data = json.loads(session_data_str)
        conversations_saved = session_data.get('conversations_saved', False)

        # 2. Delete Redis session
        r.delete(redis_key)
        logger.info(f"🗑️  Deleted Redis session {session_id}")

        # 3. Delete SQLite checkpoints (if conversations were saved)
        if conversations_saved:
            db_path = str(Path(__file__).parent / "conversations.db")
            try:
                sqlite_conn = sqlite3.connect(db_path)
                cursor = sqlite_conn.cursor()
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                cursor.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
                deleted_checkpoints = cursor.rowcount
                sqlite_conn.commit()
                sqlite_conn.close()
                logger.info(f"🗑️  Deleted {deleted_checkpoints} SQLite checkpoints for session {session_id}")
            except Exception as sqlite_error:
                logger.warning(f"Failed to delete SQLite checkpoints: {sqlite_error}")

        return {
            "status": "deleted",
            "session_id": session_id,
            "redis_deleted": True,
            "sqlite_deleted": conversations_saved
        }

    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint - verifies Redis connection."""
    try:
        redis_ok = r.ping()
        return {
            "status": "healthy" if redis_ok else "unhealthy",
            "redis": "connected" if redis_ok else "disconnected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "redis": "disconnected",
            "error": str(e)
        }

@app.get("/debug/latest-session")
async def get_latest_session():
    """
    Debug endpoint: Get the most recent Redis session and its contents.
    Useful for Postman testing.
    """
    try:
        # Get all session keys
        keys = r.keys("session:*")

        if not keys:
            return {
                "status": "no_sessions",
                "message": "No sessions found in Redis"
            }

        # Get the most recent one (first in list)
        latest_key = keys[0].decode() if isinstance(keys[0], bytes) else keys[0]
        session_data_str = r.get(latest_key)

        if not session_data_str:
            return {
                "status": "error",
                "message": "Session key exists but no data found"
            }

        # Parse the session data
        session_data = json.loads(session_data_str)

        return {
            "status": "success",
            "session_id": latest_key.replace("session:", ""),
            "full_key": latest_key,
            "data": session_data,
            "analyze_button_pressed": session_data.get("analyze_button_pressed", False),
            "total_sessions": len(keys)
        }

    except Exception as e:
        logger.error(f"Error fetching latest session: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/user/{user_id}/name")
async def get_user_name(user_id: str):
    """
    Get a user's name in lowercase from Postgres.

    Args:
        user_id: The user's ID in the database

    Returns:
        User's name in lowercase
    """
    from database.db import SessionLocal
    from database.models import User

    db = SessionLocal()
    try:
        # Query user by ID
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return {
                "status": "error",
                "message": "User not found"
            }

        return {
            "status": "success",
            "user_id": user_id,
            "name": user.name.lower() if user.name else ""
        }

    except Exception as e:
        logger.error(f"Error fetching user name for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/user/{user_id}/introduction")
async def generate_user_introduction(user_id: str):
    """
    Generate a short, chic, third-person introduction for a user.
    Uses their name and gender from Postgres to create a personalized intro.

    Args:
        user_id: The user's ID in the database

    Returns:
        A chic third-person introduction
    """

    db = SessionLocal()
    try:
        # Query user by ID
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return {
                "status": "error",
                "message": "User not found"
            }

        if not user.name:
            return {
                "status": "error",
                "message": "User name not available"
            }

        # Get user's name and gender
        name = user.name
        gender = user.gender if user.gender else "person"

        # Create prompt for Claude to generate introduction
        prompt = f"""Generate a short, chic, third-person introduction for a user who is a {gender} and whose name is {name}.

                Requirements:
                - Start with "introducing {name.lower()}"
                - Keep it SHORT (1 sentence max)
                - Make it chic, stylish, and entertaining
                - Third-person only
                - Lowercase letters throughout
                - Gen-z/fun vibe

                Example style: "introducing mademoiselle archita🌸", 
                "presenting miss archita", 
                "introducing the one and only archita", 
                "presenting the divine archita"

                are a few examples for women. make it end with the name of the user. 


                Now generate one for {name} ({gender}):"""

        # Call Claude API
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        introduction = response.content[0].text.strip()

        return {
            "status": "success",
            "user_id": user_id,
            "name": name,
            "gender": gender,
            "introduction": introduction
        }

    except Exception as e:
        logger.error(f"Error generating introduction for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/user/{user_id}/bio")
async def generate_user_bio(user_id: str):
    """
    Generate a short Instagram-style bio line for a user.
    Uses their university, occupation, conversations, and other data from Postgres.

    Args:
        user_id: The user's ID in the database

    Returns:
        A short bio line like "cornell ECE '26 • SF" or "wharton | NYC"
    """

    db = SessionLocal()
    try:
        # Query user by ID
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return {
                "status": "error",
                "message": "User not found"
            }

        # Gather all available user data
        name = user.name if user.name else ""
        university = user.university if user.university else ""
        college_major = user.college_major if user.college_major else ""
        occupation = user.occupation if user.occupation else ""
        conversations = user.conversations if user.conversations else []

        # Create prompt for Claude to generate bio
        prompt = f"""Generate a short, Instagram-style bio line for a user based on their information.

User Information:
- Name: {name}
- University: {university}
- Major: {college_major}
- Occupation: {occupation}
- Conversations: {json.dumps(conversations)}

Requirements:
- ONE line only (like an Instagram bio)
- Include school/university with major/year if available (e.g., "cornell ECE '26")
- Include city or location if mentioned in conversations
- Use symbols like • or | to separate elements
- Keep it SHORT and chic
- Lowercase preferred
- Examples: "cornell ECE '26 • SF", "wharton | NYC", "stanford cs • building in SF", "google | brooklyn"

Based on the user's info and conversations, generate a short bio line:"""

        # Call Claude API
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        bio = response.content[0].text.strip()

        return {
            "status": "success",
            "user_id": user_id,
            "bio": bio
        }

    except Exception as e:
        logger.error(f"Error generating bio for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/user/{user_id}/twoCaptions")
async def generate_user_captions(user_id: str):
    """
    Generate two strong, bold captions about the user for their profile.
    These will be the biggest/boldest text on their profile.
    Uses conversations and user data from Postgres.

    Args:
        user_id: The user's ID in the database

    Returns:
        Two short, chic, bold captions
    """

    db = SessionLocal()
    try:
        # Query user by ID
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return {
                "status": "error",
                "message": "User not found"
            }

        # Gather all available user data
        name = user.name if user.name else ""
        gender = user.gender if user.gender else ""
        university = user.university if user.university else ""
        college_major = user.college_major if user.college_major else ""
        occupation = user.occupation if user.occupation else ""
        conversations = user.conversations if user.conversations else []

        # Create prompt for Claude to generate captions
        prompt = f"""Generate TWO strong, bold captions about this user for their profile. These will be the BIGGEST and BOLDEST text on their profile.

User Information:
- Name: {name}
- Gender: {gender}
- University: {university}
- Major: {college_major}
- Occupation: {occupation}
- Conversations: {json.dumps(conversations)}

Requirements:
- Generate EXACTLY 2 captions
- Each caption should be SHORT (3-7 words max)
- Bold, confident, attention-grabbing
- Chic and entertaining
- Based on their personality from conversations
- Lowercase preferred
- Third person.

Analyze their conversations and info to capture their vibe. Generate 2 strong captions:"""

        # Call Claude API
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Parse response to extract two captions
        response_text = response.content[0].text.strip()

        # Split by newlines or numbers to get two captions
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        # Clean up any numbering (1., 2., etc)
        captions = []
        for line in lines:
            cleaned = line.lstrip('12345678.-) ').strip('"\'')
            if cleaned:
                captions.append(cleaned)

        # Ensure we have exactly 2 captions
        if len(captions) < 2:
            captions = ["chic and mysterious", "living my best life"]

        caption1 = captions[0]
        caption2 = captions[1] if len(captions) > 1 else captions[0]

        return {
            "status": "success",
            "user_id": user_id,
            "caption1": caption1,
            "caption2": caption2
        }

    except Exception as e:
        logger.error(f"Error generating captions for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/user/{user_id}/eightCaptions")
async def generate_eight_captions(user_id: str):
    """
    Generate 8 captions to describe the user for their profile.
    Main character energy, third person.
    Uses conversations and user data from Postgres.

    Args:
        user_id: The user's ID in the database

    Returns:
        Eight captions describing the user
    """

    db = SessionLocal()
    try:
        # Query user by ID
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return {
                "status": "error",
                "message": "User not found"
            }

        # Gather all available user data
        name = user.name if user.name else ""
        gender = user.gender if user.gender else ""
        university = user.university if user.university else ""
        college_major = user.college_major if user.college_major else ""
        occupation = user.occupation if user.occupation else ""
        sexuality = user.sexuality if user.sexuality else ""
        ethnicity = user.ethnicity if user.ethnicity else ""
        pronouns = user.pronouns if user.pronouns else ""
        conversations = user.conversations if user.conversations else []

        # Create prompt for Claude to generate 8 captions
        prompt = f"""Generate EIGHT captions to describe this user for their profile. Main character energy, third person.

User Information:
- Name: {name}
- Gender: {gender}
- Pronouns: {pronouns}
- University: {university}
- Major: {college_major}
- Occupation: {occupation}
- Sexuality: {sexuality}
- Ethnicity: {ethnicity}
- Conversations: {json.dumps(conversations)}

Requirements:
- Generate EXACTLY 8 captions, make sure they are specific to the user's personality, and capture specfic aspects of who they are. 
- Each caption should be SHORT (3-8 words max)
- Third person only
- Main character energy
- Chic, fun, entertaining
- Based on their personality from conversations
- Lowercase preferred
- Capture different aspects of who they are

Analyze their conversations and info deeply. Generate 8 captions that paint a full picture of who they are:"""

        # Call Claude API
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Parse response to extract 8 captions
        response_text = response.content[0].text.strip()

        # Split by newlines to get captions
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        # Clean up any numbering (1., 2., etc)
        captions = []
        for line in lines:
            cleaned = line.lstrip('12345678.-) ').strip('"\'')
            if cleaned:
                captions.append(cleaned)

        # Ensure we have exactly 8 captions
        default_captions = [
            "living their best life",
            "chic and mysterious",
            "main character energy",
            "your new favorite person",
            "vibes immaculate",
            "certified trendsetter",
            "story worth hearing",
            "effortlessly cool"
        ]

        while len(captions) < 8:
            captions.append(default_captions[len(captions)])

        # Return exactly 8 captions
        return {
            "status": "success",
            "user_id": user_id,
            "captions": captions[:8]
        }

    except Exception as e:
        logger.error(f"Error generating 8 captions for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/debug/all-sessions")
async def get_all_sessions():
    """
    Debug endpoint: Get ALL Redis session keys and their contents.
    Returns a list of all sessions with their data.
    """
    try:
        # Get all session keys
        keys = r.keys("session:*")

        if not keys:
            return {
                "status": "no_sessions",
                "message": "No sessions found in Redis",
                "total": 0,
                "sessions": []
            }

        # Get data for all sessions
        all_sessions = []
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            session_data_str = r.get(key_str)

            if session_data_str:
                try:
                    session_data = json.loads(session_data_str)
                    all_sessions.append({
                        "session_id": key_str.replace("session:", ""),
                        "full_key": key_str,
                        "analyze_button_pressed": session_data.get("analyze_button_pressed", False),
                        "has_user_id": "user_id" in session_data,
                        "signup_data": session_data.get("signup_data", {}),
                        "full_data": session_data
                    })
                except json.JSONDecodeError:
                    all_sessions.append({
                        "session_id": key_str.replace("session:", ""),
                        "full_key": key_str,
                        "error": "Failed to parse JSON"
                    })

        return {
            "status": "success",
            "total": len(all_sessions),
            "sessions": all_sessions
        }

    except Exception as e:
        logger.error(f"Error fetching all sessions: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stream:app", host="0.0.0.0", port=8000, reload=True)

