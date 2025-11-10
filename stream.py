from dotenv import load_dotenv
import os, asyncio, json, logging
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

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
            # User ID is ready!
            return {
                "status": "ready",
                "user_id": user_id,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stream:app", host="0.0.0.0", port=8000, reload=True)

