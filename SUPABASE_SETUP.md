# Supabase Authentication Setup Guide

This guide will help you set up Supabase Authentication for the AI Forecast Intelligence Platform.

## Prerequisites

- A Supabase account
- Basic knowledge of Supabase Dashboard

## Step 1: Create a Supabase Project

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Click "New Project"
3. Choose your organization
4. Enter a project name (e.g., "ai-forecast-platform")
5. Enter a database password
6. Choose a region close to your users
7. Click "Create new project"

## Step 2: Enable Authentication

1. In your Supabase project, go to "Authentication" in the left sidebar
2. Go to "Settings" tab
3. Configure the following:

### Email Authentication
1. Enable "Enable email confirmations" (optional)
2. Enable "Enable email change confirmations" (optional)
3. Configure your site URL (your deployment domain)

### Email Templates (Optional)
1. Customize email templates for better user experience
2. Set your site name and logo

## Step 3: Get Supabase Configuration

1. In your Supabase project, go to "Settings" in the left sidebar
2. Click on "API"
3. Copy the following values:
   - **Project URL** (e.g., `https://your-project.supabase.co`)
   - **Anon public key** (starts with `eyJ...`)

## Step 4: Update Backend Configuration

### Option A: Using Environment Variables (Recommended for Production)

1. In your deployment platform (Vercel, Railway, etc.), set the following environment variables:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
JWT_SECRET=your-secret-key-here
```

### Option B: Using Default Configuration (Development)

The application comes with a default Supabase configuration for development. For production, always use environment variables.

## Step 5: Install Dependencies

Make sure you have the required Python packages installed:

```bash
pip install supabase PyJWT
```

Or update your `requirements.txt`:

```
supabase==2.3.4
PyJWT==2.8.0
```

## Step 6: Test the Setup

1. Start your application
2. Run the authentication test:
   ```bash
   python test_auth.py
   ```
3. Try signing in/up through the web interface
4. Check the browser console and server logs for any errors

## Security Considerations

1. **Never commit your Supabase keys to version control**
2. **Use environment variables for sensitive data**
3. **Set up proper CORS rules in Supabase Dashboard**
4. **Configure authorized domains in Supabase Authentication**

## Troubleshooting

### Common Issues

1. **"Supabase client not initialized" error**
   - Make sure SUPABASE_URL and SUPABASE_ANON_KEY are set correctly
   - Check that your Supabase project is active

2. **"Authentication not available" error**
   - Check that Supabase client is properly installed
   - Verify your Supabase configuration

3. **"Invalid email or password" error**
   - Ensure your Supabase project ID matches in both frontend and backend
   - Check that the user is properly created in Supabase

4. **CORS errors**
   - Add your domain to authorized domains in Supabase Dashboard
   - Check that your deployment URL is properly configured

### Debug Mode

To enable debug logging, add this to your Python code:

```python
import logging
logging.getLogger('supabase').setLevel(logging.DEBUG)
```

## Production Deployment

For production deployment:

1. **Use environment variables** for all sensitive configuration
2. **Set up proper CORS** in Supabase Dashboard
3. **Configure authorized domains** for your production URL
4. **Use a strong JWT secret** (generate a random 32+ character string)
5. **Enable HTTPS** for your production domain
6. **Set up proper error monitoring** (e.g., Sentry)

## Example Environment Variables

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# JWT Secret (generate a random string)
JWT_SECRET=your-super-secret-jwt-key-here-32-chars-minimum
```

## Frontend Integration

The frontend is already configured to work with Supabase. The configuration is in `index.html`:

```javascript
const SUPABASE_URL = 'https://your-project.supabase.co';
const SUPABASE_ANON_KEY = 'your-anon-key-here';
```

## Support

If you encounter issues:

1. Check the browser console for JavaScript errors
2. Check the server logs for Python errors
3. Verify your Supabase configuration
4. Ensure all environment variables are set correctly
5. Test with a simple Supabase project first

## Migration from Firebase

If you're migrating from Firebase:

1. **Export user data** from Firebase (if needed)
2. **Create new users** in Supabase
3. **Update environment variables** to use Supabase
4. **Test authentication flow** thoroughly
5. **Update any Firebase-specific code** to use Supabase 