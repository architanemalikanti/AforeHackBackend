"""
Push notification service for iOS using APNs
"""

import os
import logging
from typing import Optional
from aioapns import APNs, NotificationRequest, PushType
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# APNs Configuration
# You'll need to set these in your .env file:
# APNS_KEY_PATH=path/to/your/AuthKey_XXXXXXXXXX.p8
# APNS_KEY_ID=your-key-id (10 characters)
# APNS_TEAM_ID=your-team-id (10 characters)
# APNS_TOPIC=your.app.bundle.id (e.g., com.yourcompany.yourapp)
# APNS_USE_SANDBOX=True (for development) or False (for production)

APNS_KEY_PATH = os.getenv("APNS_KEY_PATH")
APNS_KEY_ID = os.getenv("APNS_KEY_ID")
APNS_TEAM_ID = os.getenv("APNS_TEAM_ID")
APNS_TOPIC = os.getenv("APNS_TOPIC")
APNS_USE_SANDBOX = os.getenv("APNS_USE_SANDBOX", "True").lower() == "true"

# Global APNs client (initialized on first use)
_apns_client: Optional[APNs] = None


async def get_apns_client() -> Optional[APNs]:
    """Get or initialize the APNs client"""
    global _apns_client

    if _apns_client is not None:
        return _apns_client

    # Check if all required credentials are configured
    if not all([APNS_KEY_PATH, APNS_KEY_ID, APNS_TEAM_ID, APNS_TOPIC]):
        logger.warning("⚠️  APNs credentials not configured. Push notifications will be disabled.")
        logger.warning("Please set APNS_KEY_PATH, APNS_KEY_ID, APNS_TEAM_ID, and APNS_TOPIC in .env")
        return None

    try:
        # Read the .p8 key file
        with open(APNS_KEY_PATH, 'r') as f:
            apns_key = f.read()

        # Initialize APNs client
        _apns_client = APNs(
            key=apns_key,
            key_id=APNS_KEY_ID,
            team_id=APNS_TEAM_ID,
            topic=APNS_TOPIC,
            use_sandbox=APNS_USE_SANDBOX
        )

        logger.info(f"✅ APNs client initialized (sandbox={APNS_USE_SANDBOX})")
        return _apns_client

    except Exception as e:
        logger.error(f"❌ Failed to initialize APNs client: {e}")
        return None


async def send_push_notification(
    device_token: str,
    title: str,
    body: str,
    badge: Optional[int] = None,
    sound: str = "default",
    data: Optional[dict] = None
) -> bool:
    """
    Send a push notification to an iOS device.

    Args:
        device_token: The APNs device token
        title: Notification title
        body: Notification body text
        badge: Badge count (optional)
        sound: Sound to play (default: "default")
        data: Additional custom data (optional)

    Returns:
        True if notification sent successfully, False otherwise
    """
    try:
        client = await get_apns_client()

        if client is None:
            logger.warning(f"⚠️  APNs not configured. Skipping notification: {title}")
            return False

        # Build notification payload
        alert = {
            "title": title,
            "body": body
        }

        aps = {
            "alert": alert,
            "sound": sound
        }

        if badge is not None:
            aps["badge"] = badge

        # Build full payload
        payload = {"aps": aps}

        # Add custom data if provided
        if data:
            payload.update(data)

        # Create notification request
        request = NotificationRequest(
            device_token=device_token,
            message=payload,
            push_type=PushType.ALERT
        )

        # Send the notification
        response = await client.send_notification(request)

        if response.is_successful:
            logger.info(f"✅ Push notification sent: {title}")
            return True
        else:
            logger.error(f"❌ Failed to send push notification: {response.description}")
            return False

    except Exception as e:
        logger.error(f"❌ Error sending push notification: {e}")
        return False


async def send_follow_request_notification(device_token: str, requester_name: str) -> bool:
    """
    Send a notification when someone sends a follow request.

    Args:
        device_token: The device token of the user receiving the request
        requester_name: Name of the user who sent the request

    Returns:
        True if notification sent successfully
    """
    return await send_push_notification(
        device_token=device_token,
        title=f"{requester_name} wants to follow u", 
        body=f"{requester_name} thinks your vibe matches hers. prove her right?",
        badge=1,
        data={
            "type": "follow_request",
            "requester_name": requester_name
        }
    )


async def send_follow_accepted_notification(device_token: str, accepter_name: str, accepter_conversations: list = None) -> bool:
    """
    Send a notification when someone accepts your follow request.

    Args:
        device_token: The device token of the user whose request was accepted
        accepter_name: Name of the user who accepted the request
        accepter_conversations: The accepter's conversation history

    Returns:
        True if notification sent successfully
    """

    #create a body for this notification, prompt anthropic api for a status on the accepter's life. 2 sentences max!
    import json
    from anthropic import Anthropic

    # Default fallback body
    notification_body = f"{accepter_name} accepted your follow request"

    # Generate personalized body if conversations available
    if accepter_conversations and len(accepter_conversations) > 0:
        try:
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            prompt = f"""Generate a fun, short notification for when someone accepts a follow request.

User who accepted: {accepter_name}
Their conversations: {json.dumps(accepter_conversations)}

Write a 2-sentence life update about {accepter_name} based on their conversations. Make it personal and exciting.

Requirements:
- TWO sentences max
- 15-25 words total
- Lowercase, casual, gen-z vibe
- Third person about {accepter_name}
- Give a cinemtic peek into what they're currently up to, what era they're life is in. 
- feel free to use emojis. 

Example: "dimple has accepted your request. currently in her girlboss era — interning at j.p. morgan sf living off iced lattes and late-night spreadsheets."

Return ONLY the notification text, no quotes or explanations."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )

            generated_text = response.content[0].text.strip().strip('"\'')

            # Use generated text if it fits iOS notification limit (178 chars)
            if len(generated_text) <= 178:
                notification_body = generated_text
                logger.info(f"✨ Generated notification: {notification_body}")
            else:
                logger.warning(f"⚠️  Generated text too long ({len(generated_text)} chars), using fallback")

        except Exception as e:
            logger.error(f"❌ Error generating notification body: {e}")
            # Keep fallback body

    return await send_push_notification(
        device_token=device_token,
        title=f"{accepter_name} has accepted ur request",
        body=notification_body,
        badge=1,
        data={
            "type": "follow_accepted",
            "accepter_name": accepter_name
        }
    )
