# Firebase Authentication Setup Guide

This guide will help you set up Firebase Authentication for the AI Forecast Intelligence Platform.

## Prerequisites

- A Google account
- Basic knowledge of Firebase Console

## Step 1: Create a Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" or "Add project"
3. Enter a project name (e.g., "ai-forecast-platform")
4. Choose whether to enable Google Analytics (optional)
5. Click "Create project"

## Step 2: Enable Authentication

1. In your Firebase project, click on "Authentication" in the left sidebar
2. Click "Get started"
3. Go to the "Sign-in method" tab
4. Enable the following providers:

### Google Sign-in
1. Click on "Google" provider
2. Click "Enable"
3. Add your authorized domain (your deployment domain)
4. Click "Save"

### Email/Password Sign-in (Optional)
1. Click on "Email/Password" provider
2. Click "Enable"
3. Click "Save"

## Step 3: Get Firebase Configuration

1. In your Firebase project, click on the gear icon (⚙️) next to "Project Overview"
2. Select "Project settings"
3. Scroll down to "Your apps" section
4. Click the web icon (</>)
5. Register your app with a nickname (e.g., "AI Forecast Web App")
6. Copy the Firebase configuration object

## Step 4: Update Frontend Configuration

1. Open `index.html`
2. Find the `firebaseConfig` object in the JavaScript section
3. Replace the placeholder values with your actual Firebase configuration:

```javascript
const firebaseConfig = {
  apiKey: "your-actual-api-key",
  authDomain: "your-project-id.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project-id.appspot.com",
  messagingSenderId: "your-sender-id",
  appId: "your-app-id"
};
```

## Step 5: Set Up Backend Authentication

### Option A: Using Service Account (Recommended for Production)

1. In Firebase Console, go to Project Settings
2. Go to "Service accounts" tab
3. Click "Generate new private key"
4. Download the JSON file
5. Rename it to `firebase_config.json` and place it in your project root
6. Update the configuration in the JSON file with your actual values

### Option B: Using Environment Variables (Recommended for Deployment)

1. In your deployment platform (Vercel, Railway, etc.), set the following environment variables:

```
FIREBASE_CONFIG={"type":"service_account","project_id":"your-project-id",...}
JWT_SECRET=your-secret-key-here
```

2. Copy the entire content of your service account JSON file as the value for `FIREBASE_CONFIG`

## Step 6: Install Dependencies

Make sure you have the required Python packages installed:

```bash
pip install firebase-admin PyJWT
```

Or update your `requirements.txt`:

```
firebase-admin==6.2.0
PyJWT==2.8.0
```

## Step 7: Test the Setup

1. Start your application
2. Click the "Sign In" button
3. Try signing in with Google
4. Check the browser console and server logs for any errors

## Security Considerations

1. **Never commit your Firebase config files to version control**
2. **Use environment variables for sensitive data**
3. **Set up proper CORS rules in Firebase Console**
4. **Configure authorized domains in Firebase Authentication**

## Troubleshooting

### Common Issues

1. **"Firebase config not found" error**
   - Make sure `firebase_config.json` exists in your project root
   - Or set the `FIREBASE_CONFIG` environment variable

2. **"Authentication not available" error**
   - Check that Firebase Admin SDK is properly installed
   - Verify your service account configuration

3. **"Invalid ID token" error**
   - Ensure your Firebase project ID matches in both frontend and backend
   - Check that the user is properly authenticated in Firebase

4. **CORS errors**
   - Add your domain to authorized domains in Firebase Console
   - Check that your deployment URL is properly configured

### Debug Mode

To enable debug logging, add this to your Python code:

```python
import logging
logging.getLogger('firebase_admin').setLevel(logging.DEBUG)
```

## Production Deployment

For production deployment:

1. **Use environment variables** for all sensitive configuration
2. **Set up proper CORS** in Firebase Console
3. **Configure authorized domains** for your production URL
4. **Use a strong JWT secret** (generate a random 32+ character string)
5. **Enable HTTPS** for your production domain
6. **Set up proper error monitoring** (e.g., Sentry)

## Example Environment Variables

```bash
# Firebase Configuration (entire JSON as string)
FIREBASE_CONFIG={"type":"service_account","project_id":"your-project","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"...","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"..."}

# JWT Secret (generate a random string)
JWT_SECRET=your-super-secret-jwt-key-here-32-chars-minimum
```

## Support

If you encounter issues:

1. Check the browser console for JavaScript errors
2. Check the server logs for Python errors
3. Verify your Firebase configuration
4. Ensure all environment variables are set correctly
5. Test with a simple Firebase project first 