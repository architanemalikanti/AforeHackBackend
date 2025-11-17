from dotenv import load_dotenv
import os, asyncio, json, logging
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from database.db import SessionLocal
from database.models import User, Design, Follow, FollowRequest
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
    log_in_user,
    # Login tools
    switch_to_login_mode,
    get_login_username,
    get_login_password,
    verify_login_credentials,
    finalize_login
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
    log_in_user,
    # Login tools
    switch_to_login_mode,
    get_login_username,
    get_login_password,
    verify_login_credentials,
    finalize_login
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
        login_succeeded = False
        background_task_started = False

        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as async_memory:
            async_abot = Agent(model, all_tools, system=dynamic_prompt, checkpointer=async_memory)
            async for ev in async_abot.graph.astream_events({"messages": messages}, thread, version="v1"):
                # Check if verification tool was called and succeeded
                if ev["event"] == "on_tool_end":
                    tool_name = ev.get("name", "")
                    tool_output = ev.get("data", {}).get("output", "")

                    # Signup verification
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

                    # Login verification
                    if tool_name == "finalize_login" and tool_output == "verified":
                        login_succeeded = True
                        logger.info(f"🚀 Login completed for {session_id}")

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

        # If login succeeded, send onboarding_complete (same event as signup)
        if login_succeeded:
            logger.info(f"✅ Sending onboarding_complete to iOS for session {session_id}")
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

                Example style: "introducing mademoiselle archita", 
                "presenting miss archita", 
                "introducing the one and only archita", 
                "presenting the divine archita"

                are a few examples for women. make it end with the name of the user (no emojis allowed). 


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

IMPORTANT: Return ONLY the two captions, one per line. NO explanatory text, NO introductions, NO symbols like ** or bullets. Just the captions themselves.

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

        # Clean up any numbering, symbols, and explanatory text
        captions = []
        for line in lines:
            # Skip lines that look like explanatory text
            lower_line = line.lower()
            if any(phrase in lower_line for phrase in ['based on', 'here are', 'these captions', 'analyzing', 'capturing']):
                continue

            # Remove numbering, quotes, asterisks, bullets, etc.
            cleaned = line.lstrip('12345678.-) ').strip('"\'*•–—')
            # Remove any remaining asterisks in the middle
            cleaned = cleaned.replace('**', '').replace('*', '')
            if cleaned and len(cleaned) > 2:  # Filter out very short fragments
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

IMPORTANT: Return ONLY the eight captions, one per line. NO explanatory text, NO introductions, NO symbols like ** or bullets. Just the captions themselves.

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

        # Clean up any numbering, symbols, and explanatory text
        captions = []
        for line in lines:
            # Skip lines that look like explanatory text
            lower_line = line.lower()
            if any(phrase in lower_line for phrase in ['based on', 'here are', 'these captions', 'analyzing', 'capturing', 'paint a', 'who they are']):
                continue

            # Remove numbering, quotes, asterisks, bullets, etc.
            cleaned = line.lstrip('12345678.-) ').strip('"\'*•–—')
            # Remove any remaining asterisks in the middle
            cleaned = cleaned.replace('**', '').replace('*', '')
            if cleaned and len(cleaned) > 2:  # Filter out very short fragments
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

# Pydantic model for design creation request
class DesignCreate(BaseModel):
    user_id: str
    two_captions: List[str]
    intro_caption: str
    eight_captions: List[str]
    design_name: str
    song: str

@app.get("/user/{user_id}/topQuestions")
async def generate_top_questions(user_id: str):
    """
    Generate the top 2 questions this user might ask based on their conversation history.
    Uses conversations from Postgres.

    Args:
        user_id: The user's ID in the database

    Returns:
        Two questions the user might ask
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

        # Get user data
        name = user.name if user.name else ""
        conversations = user.conversations if user.conversations else []

        if not conversations:
            return {
                "status": "error",
                "message": "No conversations found for this user"
            }

        # Create prompt for Claude to analyze conversations and generate questions
        prompt = f"""Analyze this user's conversation history and generate the top 2 questions / things they'd they're most likely talk about / things they are most likely to ask next.

User Information:
- Name: {name}
- Conversations: {json.dumps(conversations)}

Based on their conversation patterns, interests, and personality, what are the top 2 questions they would most likely ask?

Requirements:
- Generate EXACTLY 2 questions
- Questions should feel natural and aligned with their interests/personality
- Each question should be SHORT (5-15 words)
- Based on what they've talked about in conversations
- Write questions in girly, genz, human tone. you can be humorous. 

IMPORTANT: Return ONLY the two questions, one per line. NO explanatory text, NO introductions, NO numbering, NO symbols like ** or bullets. Just the questions themselves.

Generate 2 questions:"""

        # Call Claude API
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Parse response to extract two questions
        response_text = response.content[0].text.strip()

        # Split by newlines to get questions
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        # Clean up any numbering, symbols, and explanatory text
        questions = []
        for line in lines:
            # Skip lines that look like explanatory text
            lower_line = line.lower()
            if any(phrase in lower_line for phrase in ['based on', 'here are', 'these questions', 'analyzing', 'likely to ask']):
                continue

            # Remove numbering, quotes, asterisks, bullets, etc.
            cleaned = line.lstrip('12345678.-) ').strip('"\'*•–—')
            # Remove any remaining asterisks in the middle
            cleaned = cleaned.replace('**', '').replace('*', '')
            if cleaned and len(cleaned) > 5:  # Filter out very short fragments
                questions.append(cleaned)

        # Ensure we have exactly 2 questions
        if len(questions) < 2:
            questions = [
                "what's your favorite thing to do on weekends?",
                "any fun plans coming up?"
            ]

        question1 = questions[0]
        question2 = questions[1] if len(questions) > 1 else questions[0]

        return {
            "status": "success",
            "user_id": user_id,
            "question1": question1,
            "question2": question2
        }

    except Exception as e:
        logger.error(f"Error generating questions for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/searchUsers")
async def search_users(query: str = Query(..., min_length=1)):
    """
    Search for users by username or name.
    Returns top 5 matching results.

    Args:
        query: Search string (username or name)

    Returns:
        List of up to 5 matching users with their basic info
    """
    db = SessionLocal()
    try:
        # Normalize query to lowercase for case-insensitive search
        search_term = query.lower().strip()

        if not search_term:
            return {
                "status": "error",
                "message": "Search query cannot be empty"
            }

        # Query database for users matching username or name
        # Using ilike for case-insensitive partial matching
        matching_users = db.query(User).filter(
            (User.username.ilike(f"%{search_term}%")) |
            (User.name.ilike(f"%{search_term}%"))
        ).limit(5).all()

        # Format results
        results = []
        for user in matching_users:
            results.append({
                "user_id": user.id,
                "username": user.username,
                "name": user.name,
                "university": user.university if user.university else None,
                "occupation": user.occupation if user.occupation else None
            })

        return {
            "status": "success",
            "query": query,
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error searching users with query '{query}': {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/user/{user_id}/currentEra")
async def get_current_era(user_id: str):
    """
    Analyze the user's conversations and return a cinematic description of their current "era".
    Very Gen-Z, very girly, very main character energy.

    Args:
        user_id: The user's ID in the database

    Returns:
        A cinematic 1-3 sentence description of what era they're in right now
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

        # Get user data
        name = user.name if user.name else ""
        conversations = user.conversations if user.conversations else []

        if not conversations:
            return {
                "status": "error",
                "message": "No conversations found for this user"
            }

        # Create prompt for Claude to analyze and describe their current era
        prompt = f"""Analyze this user's recent conversations and describe what "era" they're currently in.

User Information:
- Name: {name}
- Conversations: {json.dumps(conversations)}

Based on their recent conversations, what's happening in their life right now? What era are they entering or living through?

Requirements:
- Write 1-3 sentences MAX
- Cinematic and dramatic tone
- Very Gen-Z, very girly. 
- Third person (e.g., "{name} is entering her law school era")
- Lowercase preferred
- Focus on what's currently happening or about to happen in their life
- Make it feel like a movie narration
- three sentences max. 

IMPORTANT: Return ONLY the era description. NO explanatory text, NO introductions. Just the cinematic description itself.

Analyze their conversations and describe their current era:"""

        # Call Claude API
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        era_description = response.content[0].text.strip()

        # Clean up any unwanted formatting
        era_description = era_description.replace('**', '').replace('*', '')

        return {
            "status": "success",
            "user_id": user_id,
            "era": era_description
        }

    except Exception as e:
        logger.error(f"Error generating era description for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.post("/design/create")
async def create_design(design_data: DesignCreate):
    """
    Create a new design for a user and save it to the designs table.

    Request body:
    {
        "user_id": "uuid-string",
        "two_captions": ["caption1", "caption2"],
        "intro_caption": "introducing mademoiselle archita 🌸",
        "eight_captions": ["caption1", "caption2", ..., "caption8"],
        "design_name": "design name",
        "song": "song name or URL"
    }

    Returns:
        Design ID and success status
    """
    db = SessionLocal()
    try:
        # Verify user exists
        user = db.query(User).filter(User.id == design_data.user_id).first()
        if not user:
            return {
                "status": "error",
                "message": f"User with ID {design_data.user_id} not found"
            }

        # Create new design
        new_design = Design(
            user_id=design_data.user_id,
            two_captions=design_data.two_captions,
            intro_caption=design_data.intro_caption,
            eight_captions=design_data.eight_captions,
            design_name=design_data.design_name,
            song=design_data.song
        )

        db.add(new_design)
        db.commit()
        db.refresh(new_design)

        logger.info(f"✅ Created design {new_design.id} for user {design_data.user_id}")

        return {
            "status": "success",
            "design_id": new_design.id,
            "user_id": design_data.user_id,
            "message": "Design created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating design: {e}")
        db.rollback()
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

# ===== PUSH NOTIFICATION ROUTES =====

class DeviceTokenUpdate(BaseModel):
    user_id: str
    device_token: str

@app.post("/user/device-token")
async def update_device_token(token_data: DeviceTokenUpdate):
    """
    Register or update a user's APNs device token for push notifications.

    Request body:
    {
        "user_id": "uuid-string",
        "device_token": "apns-device-token"
    }
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == token_data.user_id).first()

        if not user:
            return {
                "status": "error",
                "message": "User not found"
            }

        user.device_token = token_data.device_token
        db.commit()

        logger.info(f"✅ Updated device token for user {token_data.user_id}")

        return {
            "status": "success",
            "message": "Device token updated successfully",
            "user_id": token_data.user_id
        }

    except Exception as e:
        logger.error(f"Error updating device token: {e}")
        db.rollback()
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

# ===== FOLLOW SYSTEM ROUTES =====

# Pydantic models for follow requests
class FollowRequestCreate(BaseModel):
    requester_id: str
    requested_id: str

class FollowActionRequest(BaseModel):
    requester_id: str
    requested_id: str

@app.post("/follow/request")
async def send_follow_request(request_data: FollowRequestCreate):
    """
    User A sends a follow request to User B.
    Since all profiles are private, this creates a pending request.

    Request body:
    {
        "requester_id": "user_a_id",
        "requested_id": "user_b_id"
    }
    """
    from push_notifications import send_follow_request_notification

    db = SessionLocal()
    try:
        # Check if both users exist
        requester = db.query(User).filter(User.id == request_data.requester_id).first()
        requested = db.query(User).filter(User.id == request_data.requested_id).first()

        if not requester or not requested:
            return {
                "status": "error",
                "message": "One or both users not found"
            }

        # Check if already following
        existing_follow = db.query(Follow).filter(
            Follow.follower_id == request_data.requester_id,
            Follow.following_id == request_data.requested_id
        ).first()

        if existing_follow:
            return {
                "status": "error",
                "message": "Already following this user"
            }

        # Check if request already exists
        existing_request = db.query(FollowRequest).filter(
            FollowRequest.requester_id == request_data.requester_id,
            FollowRequest.requested_id == request_data.requested_id
        ).first()

        if existing_request:
            return {
                "status": "error",
                "message": "Follow request already sent"
            }

        # Create new follow request
        new_request = FollowRequest(
            requester_id=request_data.requester_id,
            requested_id=request_data.requested_id
        )

        db.add(new_request)
        db.commit()
        db.refresh(new_request)

        logger.info(f"✅ User {request_data.requester_id} sent follow request to {request_data.requested_id}")

        # Send push notification to the requested user (User B)
        if requested.device_token:
            requester_name = requester.name if requester.name else requester.username
            await send_follow_request_notification(
                device_token=requested.device_token,
                requester_name=requester_name
            )
        else:
            logger.info(f"⚠️  No device token for user {request_data.requested_id}, skipping push notification")

        return {
            "status": "success",
            "message": "Follow request sent",
            "request_id": new_request.id
        }

    except Exception as e:
        logger.error(f"Error sending follow request: {e}")
        db.rollback()
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/follow/requests/{user_id}")
async def get_follow_requests(user_id: str):
    """
    Get all pending follow requests for a user.
    Shows who wants to follow them.

    Args:
        user_id: The user's ID

    Returns:
        List of pending follow requests with requester info
    """
    db = SessionLocal()
    try:
        # Get all pending requests for this user
        requests = db.query(FollowRequest).filter(
            FollowRequest.requested_id == user_id
        ).order_by(FollowRequest.created_at.desc()).all()

        # Format results with requester info
        results = []
        for req in requests:
            requester = db.query(User).filter(User.id == req.requester_id).first()
            if requester:
                results.append({
                    "request_id": req.id,
                    "requester_id": requester.id,
                    "username": requester.username,
                    "name": requester.name,
                    "university": requester.university,
                    "occupation": requester.occupation,
                    "created_at": req.created_at.isoformat()
                })

        return {
            "status": "success",
            "user_id": user_id,
            "count": len(results),
            "requests": results
        }

    except Exception as e:
        logger.error(f"Error fetching follow requests for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.post("/follow/accept")
async def accept_follow_request(request_data: FollowActionRequest):
    """
    User B accepts User A's follow request.
    Creates the actual follow relationship and deletes the pending request.

    Request body:
    {
        "requester_id": "user_a_id",
        "requested_id": "user_b_id"
    }
    """
    from push_notifications import send_follow_accepted_notification

    db = SessionLocal()
    try:
        # Find the pending request
        pending_request = db.query(FollowRequest).filter(
            FollowRequest.requester_id == request_data.requester_id,
            FollowRequest.requested_id == request_data.requested_id
        ).first()

        if not pending_request:
            return {
                "status": "error",
                "message": "Follow request not found"
            }

        # Get both users for push notification
        requester = db.query(User).filter(User.id == request_data.requester_id).first()
        accepter = db.query(User).filter(User.id == request_data.requested_id).first()

        # Create the actual follow relationship
        new_follow = Follow(
            follower_id=request_data.requester_id,
            following_id=request_data.requested_id
        )

        db.add(new_follow)

        # Delete the pending request
        db.delete(pending_request)

        db.commit()

        logger.info(f"✅ User {request_data.requested_id} accepted follow from {request_data.requester_id}")

        # Send push notification to the requester (User A) that their request was accepted
        if requester and requester.device_token:
            accepter_name = accepter.name if accepter.name else accepter.username
            accepter_conversations = accepter.conversations if accepter.conversations else []
            await send_follow_accepted_notification(
                device_token=requester.device_token,
                accepter_name=accepter_name,
                accepter_conversations=accepter_conversations
            )
        else:
            logger.info(f"⚠️  No device token for user {request_data.requester_id}, skipping push notification")

        return {
            "status": "success",
            "message": "Follow request accepted"
        }

    except Exception as e:
        logger.error(f"Error accepting follow request: {e}")
        db.rollback()
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.post("/follow/decline")
async def decline_follow_request(request_data: FollowActionRequest):
    """
    User B declines User A's follow request.
    Simply deletes the pending request.

    Request body:
    {
        "requester_id": "user_a_id",
        "requested_id": "user_b_id"
    }
    """
    db = SessionLocal()
    try:
        # Find the pending request
        pending_request = db.query(FollowRequest).filter(
            FollowRequest.requester_id == request_data.requester_id,
            FollowRequest.requested_id == request_data.requested_id
        ).first()

        if not pending_request:
            return {
                "status": "error",
                "message": "Follow request not found"
            }

        # Delete the pending request
        db.delete(pending_request)
        db.commit()

        logger.info(f"❌ User {request_data.requested_id} declined follow from {request_data.requester_id}")

        return {
            "status": "success",
            "message": "Follow request declined"
        }

    except Exception as e:
        logger.error(f"Error declining follow request: {e}")
        db.rollback()
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/user/{user_id}/followers")
async def get_followers(user_id: str):
    """
    Get all users who follow this user (User B's followers).

    Args:
        user_id: The user's ID

    Returns:
        List of followers with their info
    """
    db = SessionLocal()
    try:
        # Get all follows where this user is being followed
        follows = db.query(Follow).filter(
            Follow.following_id == user_id
        ).all()

        # Get follower info
        results = []
        for follow in follows:
            follower = db.query(User).filter(User.id == follow.follower_id).first()
            if follower:
                results.append({
                    "user_id": follower.id,
                    "username": follower.username,
                    "name": follower.name,
                    "university": follower.university,
                    "occupation": follower.occupation,
                    "followed_at": follow.created_at.isoformat()
                })

        return {
            "status": "success",
            "user_id": user_id,
            "count": len(results),
            "followers": results
        }

    except Exception as e:
        logger.error(f"Error fetching followers for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/user/{user_id}/following")
async def get_following(user_id: str):
    """
    Get all users that this user follows (who User B is following).

    Args:
        user_id: The user's ID

    Returns:
        List of users they're following with their info
    """
    db = SessionLocal()
    try:
        # Get all follows where this user is the follower
        follows = db.query(Follow).filter(
            Follow.follower_id == user_id
        ).all()

        # Get following info
        results = []
        for follow in follows:
            following = db.query(User).filter(User.id == follow.following_id).first()
            if following:
                results.append({
                    "user_id": following.id,
                    "username": following.username,
                    "name": following.name,
                    "university": following.university,
                    "occupation": following.occupation,
                    "followed_at": follow.created_at.isoformat()
                })

        return {
            "status": "success",
            "user_id": user_id,
            "count": len(results),
            "following": results
        }

    except Exception as e:
        logger.error(f"Error fetching following for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/user/{user_id}/follower-count")
async def get_follower_count(user_id: str):
    """
    Get the follower and following counts for a user.

    Args:
        user_id: The user's ID

    Returns:
        Follower count and following count
    """
    db = SessionLocal()
    try:
        # Count followers
        follower_count = db.query(Follow).filter(
            Follow.following_id == user_id
        ).count()

        # Count following
        following_count = db.query(Follow).filter(
            Follow.follower_id == user_id
        ).count()

        return {
            "status": "success",
            "user_id": user_id,
            "follower_count": follower_count,
            "following_count": following_count
        }

    except Exception as e:
        logger.error(f"Error fetching counts for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stream:app", host="0.0.0.0", port=8000, reload=True)

