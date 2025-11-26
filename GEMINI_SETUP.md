# Gemini API Setup Guide

## Getting Your Free API Key

1. Visit https://aistudio.google.com/
2. Sign in with your Google account
3. Click "Get API Key" button
4. Create a new API key (it's FREE!)
5. Copy the API key

## Setup

1. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

3. Set environment variable before running the server:

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
uvicorn main:app --reload
```

**Or permanently in PowerShell:**
```powershell
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your_api_key_here', 'User')
```

## Free Tier Limits

- **Gemini 1.5 Flash**: 15 requests per minute
- **1 million tokens per day**
- No credit card required

## What It Adds

The Gemini API enhances your predictions with:
- **Analysis Summary**: One-sentence explanation
- **Reasoning Points**: 3-5 bullet points explaining the decision
- **Key Indicators**: Specific elements detected
- All presented seamlessly as part of your ML model's output!

## Running Without Gemini

If you don't set the API key, the system works fine - it just won't show the enhanced reasoning section.
