# Push Notifications Setup Guide

This guide explains how to set up iOS push notifications for the follow request system.

## What's Been Implemented

1. **Database**: Added `device_token` column to `users` table
2. **Models**: Updated User model with `device_token` field
3. **Endpoint**: POST `/user/device-token` - Register/update device tokens
4. **Push Service**: `push_notifications.py` - Handles APNs integration
5. **Follow Notifications**:
   - When User A sends a follow request → User B gets a push notification
   - When User B accepts the request → User A gets a push notification

## How It Works

### Flow Overview

1. **User registers device token** (iOS app startup):
   ```
   POST /user/device-token
   {
     "user_id": "abc-123",
     "device_token": "apns-token-from-ios"
   }
   ```

2. **User A sends follow request**:
   - POST `/follow/request` creates the request
   - Backend sends push notification to User B's device
   - User B sees: "New Follow Request - [User A's name] wants to follow you"

3. **User B accepts request**:
   - POST `/follow/accept` creates the follow relationship
   - Backend sends push notification to User A's device
   - User A sees: "Follow Request Accepted - [User B's name] accepted your follow request"

## APNs Configuration Required

To enable push notifications, you need to configure Apple Push Notification service (APNs):

### Step 1: Get APNs Credentials from Apple Developer Portal

1. Go to [Apple Developer Portal](https://developer.apple.com/account)
2. Navigate to **Certificates, Identifiers & Profiles**
3. Click **Keys** in the sidebar
4. Click the **+** button to create a new key
5. Give it a name (e.g., "Push Notification Key")
6. Check **Apple Push Notifications service (APNs)**
7. Click **Continue** and then **Register**
8. **Download the .p8 file** (you can only download it once!)
9. Note your **Key ID** (10 characters, e.g., ABC1234DEF)
10. Note your **Team ID** (found in top right of developer portal, 10 characters)

### Step 2: Add Credentials to .env File

Add these variables to your `.env` file:

```bash
# APNs Configuration
APNS_KEY_PATH=/path/to/your/AuthKey_ABC1234DEF.p8
APNS_KEY_ID=ABC1234DEF
APNS_TEAM_ID=XYZ9876WVU
APNS_TOPIC=com.yourcompany.yourapp
APNS_USE_SANDBOX=True
```

**Configuration Details:**
- `APNS_KEY_PATH`: Absolute path to your downloaded .p8 key file
- `APNS_KEY_ID`: Your 10-character key ID
- `APNS_TEAM_ID`: Your 10-character team ID
- `APNS_TOPIC`: Your app's bundle identifier (e.g., `com.glow.app`)
- `APNS_USE_SANDBOX`: Set to `True` for development, `False` for production

### Step 3: iOS App Configuration

Your iOS app needs to:

1. **Request notification permissions** on app launch
2. **Register for remote notifications** with APNs
3. **Receive device token** from iOS
4. **Send token to backend**:
   ```swift
   func application(_ application: UIApplication,
                    didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
       let token = deviceToken.map { String(format: "%02.2hhx", $0) }.joined()

       // Send to backend
       let url = URL(string: "https://your-backend.com/user/device-token")!
       var request = URLRequest(url: url)
       request.httpMethod = "POST"
       request.setValue("application/json", forHTTPHeaderField: "Content-Type")

       let body = ["user_id": currentUserId, "device_token": token]
       request.httpBody = try? JSONSerialization.data(withJSONObject: body)

       URLSession.shared.dataTask(with: request).resume()
   }
   ```

## Testing Push Notifications

### Without Full APNs Setup (Development)

If APNs credentials are not configured, the system will:
- Log warnings but continue working
- Not crash or fail follow requests
- Log: `⚠️  APNs not configured. Skipping notification: [title]`

This allows you to develop and test the follow system without APNs.

### With APNs Setup

1. Ensure `.env` has all APNs credentials
2. Restart your FastAPI server
3. Register a device token via POST `/user/device-token`
4. Test follow request flow:
   - User A sends request → User B should receive push notification
   - User B accepts → User A should receive push notification

### Check Logs

The server logs will show:
- `✅ APNs client initialized (sandbox=True)` - APNs is configured
- `✅ Push notification sent: [title]` - Notification sent successfully
- `❌ Failed to send push notification: [reason]` - Error occurred

## Notification Payload Structure

### Follow Request Notification
```json
{
  "aps": {
    "alert": {
      "title": "New Follow Request",
      "body": "[Requester Name] wants to follow you"
    },
    "badge": 1,
    "sound": "default"
  },
  "type": "follow_request",
  "requester_name": "[Requester Name]"
}
```

### Follow Accepted Notification
```json
{
  "aps": {
    "alert": {
      "title": "Follow Request Accepted",
      "body": "[Accepter Name] accepted your follow request"
    },
    "badge": 1,
    "sound": "default"
  },
  "type": "follow_accepted",
  "accepter_name": "[Accepter Name]"
}
```

## API Reference

### POST /user/device-token
Register or update a user's device token.

**Request:**
```json
{
  "user_id": "user-uuid",
  "device_token": "apns-device-token-string"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Device token updated successfully",
  "user_id": "user-uuid"
}
```

## Security Notes

1. **Keep your .p8 key file secure** - Never commit it to git
2. **Add to .gitignore**: Add `*.p8` to your .gitignore
3. **Use environment variables** for all sensitive credentials
4. **Production vs Development**:
   - Use `APNS_USE_SANDBOX=True` for TestFlight and development
   - Use `APNS_USE_SANDBOX=False` for production App Store builds

## Troubleshooting

### "APNs credentials not configured"
- Check that all 5 environment variables are set in `.env`
- Verify the `.p8` file path is correct and accessible
- Restart the FastAPI server after updating `.env`

### "No device token for user, skipping push notification"
- User hasn't registered their device token yet
- iOS app needs to send device token via POST `/user/device-token`
- Check iOS app has notification permissions enabled

### "Failed to send push notification"
- Check Key ID and Team ID are correct
- Verify `.p8` file is valid
- Ensure bundle ID matches your iOS app
- Check if using correct sandbox setting (dev vs production)

## Next Steps

1. **Get APNs credentials** from Apple Developer Portal
2. **Add credentials to .env**
3. **Update iOS app** to register device tokens
4. **Test** the full flow with real devices or simulator
