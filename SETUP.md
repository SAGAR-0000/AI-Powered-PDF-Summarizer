# Quick Setup Guide

Follow these steps to get the AI-Powered PDF Summarizer running in under 5 minutes!

## Step 1: Get Your Free Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

## Step 2: Configure API Key

1. Open the file `.streamlit/secrets.toml` in this project
2. Replace `your-api-key-here` with your actual API key:
   ```toml
   GEMINI_API_KEY = "AIza...your-actual-key-here"
   ```
3. Save the file

## Step 3: Install Dependencies

Open terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

## Step 4: Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Step 5: Test It Out

1. Upload a PDF (or use files from `demos/` folder)
2. Click "Generate Summary"
3. Get your AI-powered summary in seconds!

## Troubleshooting

### "Module not found" error
- Make sure you installed requirements: `pip install -r requirements.txt`
- Try: `pip install streamlit google-generativeai PyPDF2 python-dotenv`

### "API key not configured" error
- Check that `.streamlit/secrets.toml` exists
- Verify your API key is correct
- Make sure there are no extra spaces or quotes

### PDF won't upload
- Check file size (must be <5MB)
- Verify it's a valid PDF file
- Try a different PDF

## Next Steps

- **Deploy to Streamlit Cloud**: See README.md for deployment instructions
- **Customize**: Modify colors, prompts, features in `app.py`
- **Add demos**: Create your own sample PDFs in the `demos/` folder

## Getting Help

- Check the main [README.md](README.md) for full documentation
- Review [Streamlit docs](https://docs.streamlit.io)
- Visit [Google AI Studio](https://ai.google.dev) for API help

---

**Time to first summary: <5 minutes!** ⚡
