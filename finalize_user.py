"""
Finalization tools for user onboarding.
Handles verification, saving Redis data to Postgres, and conversation migration.
"""
import json
import logging
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from redis_client import r
from database.db import SessionLocal
from database.models import User
from jwt_utils import create_token_pair
from prompt_manager import set_prompt

logger = logging.getLogger(__name__)


def save_redis_to_postgres(session_id: str) -> int:
    """
    Save user data from Redis to Postgres users table.
    
    Args:
        session_id: The session identifier
        
    Returns:
        user_id: The new user's ID in Postgres
    """
    db = SessionLocal()
    try:
        # Get Redis data (using session: key format to match tools.py)
        redis_key = f"session:{session_id}"
        session_json = r.get(redis_key)
        
        if not session_json:
            logger.error(f"No Redis data found for session {session_id}")
            return 0
        
        session_data = json.loads(session_json)
        user_data = session_data.get("signup_data", {})
        logger.info(f"Retrieved Redis data for session {session_id}")

        # Generate and save the current dynamic prompt state
        current_prompt = set_prompt(session_id)
        logger.info(f"Generated prompt for user (length: {len(current_prompt)} chars)")

        # Create User object - Redis keys now match database column names
        new_user = User(
            session_id=session_id,
            name=user_data.get("name"),
            username=user_data.get("username"),
            password=user_data.get("password"),  # Already hashed by set_password tool
            email=user_data.get("email"),
            birthday=user_data.get("birthday"),
            gender=user_data.get("gender"),
            sexuality=user_data.get("sexuality"),
            ethnicity=user_data.get("ethnicity"),
            pronouns=user_data.get("pronouns"),
            university=user_data.get("university"),
            college_major=user_data.get("college_major"),
            occupation=user_data.get("occupation"),
            prompt=current_prompt  # Save the prompt state
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        user_id = new_user.id
        
        logger.info(f"✅ Saved user to Postgres with ID: {user_id}")
        return user_id
            
    except Exception as e:
        logger.error(f"Error saving Redis to Postgres for session {session_id}: {str(e)}")
        db.rollback()
        return 0
    finally:
        db.close()


def save_conversations_to_postgres(session_id: str, user_id: int) -> bool:
    """
    Save conversation history from SQLite checkpointer to user's conversations column in Postgres.

    Args:
        session_id: The session identifier (thread_id in checkpointer)
        user_id: The user's ID in Postgres

    Returns:
        bool: Success status
    """
    import sqlite3
    import pickle
    from pathlib import Path

    db = SessionLocal()
    try:
        # Get path to conversations.db
        db_path = str(Path(__file__).parent / "conversations.db")

        # Connect to SQLite and get the latest checkpoint for this thread
        sqlite_conn = sqlite3.connect(db_path)
        cursor = sqlite_conn.cursor()

        # Get the most recent checkpoint (has the full conversation state)
        cursor.execute("""
            SELECT checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
        """, (session_id,))

        row = cursor.fetchone()
        sqlite_conn.close()

        if not row:
            logger.warning(f"No checkpoint found for session {session_id}")
            return False

        checkpoint_blob, metadata_blob = row

        # Decode the checkpoint (LangGraph uses msgpack)
        try:
            import msgpack
            # Try msgpack first (newer LangGraph versions)
            checkpoint_data = msgpack.unpackb(checkpoint_blob, raw=False)
        except Exception as e1:
            try:
                # Fallback to pickle (older versions)
                checkpoint_data = pickle.loads(checkpoint_blob)
            except Exception as e2:
                logger.error(f"Failed to decode checkpoint for session {session_id}: msgpack={e1}, pickle={e2}")
                return False

        # Extract messages from checkpoint
        # LangGraph stores state in checkpoint['channel_values']
        conversations = []
        if isinstance(checkpoint_data, dict):
            channel_values = checkpoint_data.get('channel_values', {})
            messages_data = channel_values.get('messages', [])

            logger.info(f"📊 Checkpoint has {len(messages_data)} messages to process")

            # Parse each message (msgpack ExtType objects)
            for msg in messages_data:
                msg_type = None
                msg_content = None

                # Messages are msgpack ExtType objects that need to be deserialized
                if hasattr(msg, 'code') and hasattr(msg, 'data'):
                    # It's an ExtType - deserialize the data
                    try:
                        # The data is msgpack-encoded message info
                        msg_dict = msgpack.unpackb(msg.data, raw=False)
                        # msg_dict is an array: [module_path, class_name, message_data]
                        if isinstance(msg_dict, list) and len(msg_dict) >= 3:
                            class_name = msg_dict[1]  # e.g., 'HumanMessage', 'AIMessage', 'ToolMessage'
                            message_data = msg_dict[2]  # The actual message fields

                            if isinstance(message_data, dict):
                                msg_type = message_data.get('type', class_name.lower())
                                msg_content = message_data.get('content', '')
                    except Exception as e:
                        logger.warning(f"Failed to deserialize message ExtType: {e}")
                        continue

                # Fallback: check if it's already a deserialized object
                elif hasattr(msg, 'type'):
                    # It's a LangChain message object
                    msg_type = msg.type if hasattr(msg, 'type') else msg.__class__.__name__
                    msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                elif isinstance(msg, dict):
                    # It's a dictionary representation
                    msg_type = msg.get('type', '')
                    msg_content = msg.get('content', '')
                else:
                    continue

                # Only save user and assistant messages (skip tool messages)
                if msg_type and 'human' in msg_type.lower():
                    sender = 'user'
                elif msg_type and 'ai' in msg_type.lower():
                    sender = 'assistant'
                else:
                    continue  # Skip tool messages, system messages, etc.

                # Skip empty messages
                if msg_content:
                    conversations.append({
                        'sender': sender,
                        'message': msg_content,
                        'timestamp': datetime.utcnow().isoformat()
                    })

        logger.info(f"📝 Extracted {len(conversations)} user/assistant messages from checkpoint")

        # Update user's conversations column in Postgres
        if conversations:
            # Get the user and update their conversations column
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user.conversations = conversations
                db.commit()
                logger.info(f"✅ Saved {len(conversations)} messages to user {user_id}'s conversations column")
                return True
            else:
                logger.error(f"User {user_id} not found in database")
                return False
        else:
            logger.warning(f"No messages found in checkpoint for session {session_id}")
            return False

    except Exception as e:
        logger.error(f"Error saving conversations for session {session_id}: {str(e)}")
        db.rollback()
        return False
    finally:
        db.close()


@tool
def test_verification_code(session_id: str, user_input_verification_code: int) -> str:
    """
    Verify the email verification code.
    
    If verification fails: Returns "incorrect" and the prompt will ask user to try again.
    If verification succeeds: Returns "verified" and triggers background save process.
    
    Args:
        session_id: The session identifier
        user_input_verification_code: The code the user entered
        
    Returns:
        str: "incorrect" if wrong code, "verified" if correct (background tasks will handle the rest)
    """
    try:
        # Step 1: Get stored verification code from Redis (using session: key format)
        redis_key = f"session:{session_id}"
        session_json = r.get(redis_key)
        
        if not session_json:
            logger.error(f"No session found for {session_id}")
            return "incorrect"
        
        session_data = json.loads(session_json)
        user_data = session_data.get("signup_data", {})
        stored_code = user_data.get("verificationCodeGenerated")  # tools.py uses this field name
        
        if not stored_code:
            logger.error(f"No verification code found for session {session_id}")
            return "incorrect"
        
        # Step 2: Verify the code
        if int(stored_code) != int(user_input_verification_code):
            logger.info(f"❌ Verification failed for session {session_id}")
            # Mark that verification was attempted
            session_data["signup_data"]["last_verification_attempt"] = "failed"
            r.set(redis_key, json.dumps(session_data))
            return "incorrect"
        
        logger.info(f"✅ Verification code matched for session {session_id}")
        
        # Step 3: Mark as verified (background tasks will be triggered)
        session_data["signup_data"]["verification_status"] = "verified"
        r.set(redis_key, json.dumps(session_data))
        
        # Return verified - background tasks will be triggered by the stream endpoint
        return "verified"
        
    except Exception as e:
        logger.error(f"Error in test_verification_code for session {session_id}: {str(e)}")
        return "incorrect"


def finalize_user_background(session_id: str) -> int:
    """
    Background task: Save Redis data and conversations to Postgres.
    This runs AFTER the LLM has responded with the welcome message.
    
    Args:
        session_id: The session identifier
        
    Returns:
        int: user_id from Postgres (0 if failed)
    """
    try:
        logger.info(f"🔄 Starting background finalization for session {session_id}")
        
        # Step 1: Save Redis data to Postgres
        user_id = save_redis_to_postgres(session_id)
        
        if user_id == 0:
            logger.error(f"Failed to save user to Postgres for session {session_id}")
            return 0
        
        # Step 2: Save conversations to Postgres
        conversations_saved = save_conversations_to_postgres(session_id, user_id)

        if not conversations_saved:
            logger.warning(f"⚠️  Failed to save conversations for session {session_id}, but continuing...")

        # Step 3: Generate JWT tokens
        access_token, refresh_token = create_token_pair(user_id)
        logger.info(f"🔑 Generated JWT tokens for user {user_id}")

        # Step 4: Store user_id and tokens in the SAME Redis session key
        # iOS will poll this key to get the user_id and tokens, then delete it (along with SQLite)
        redis_key = f"session:{session_id}"
        session_data = json.loads(r.get(redis_key) or '{}')
        session_data['user_id'] = user_id
        session_data['access_token'] = access_token
        session_data['refresh_token'] = refresh_token
        session_data['conversations_saved'] = conversations_saved  # Track if we should clean SQLite
        r.setex(redis_key, 300, json.dumps(session_data))  # 5 min TTL for iOS to poll
        logger.info(f"💾 Stored user_id {user_id} and tokens in Redis session {session_id}")

        # Note: SQLite cleanup will happen when iOS calls /cleanup endpoint
        
        logger.info(f"🎉 Background finalization complete! User ID: {user_id}")
        return user_id
        
    except Exception as e:
        logger.error(f"Error in finalize_user_background for session {session_id}: {str(e)}")
        return 0

