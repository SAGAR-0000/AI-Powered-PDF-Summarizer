# 📄 AI-Powered PDF Summarizer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![Gemini](https://img.shields.io/badge/Google-Gemini_2.5_Flash_Lite-4285F4.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Cost](https://img.shields.io/badge/Cost-$0-success.svg)

**An intelligent PDF document summarizer powered by Google Gemini AI, optimized for speed and efficiency within free tier constraints.**

**Personal Project by SAGAR**

[Live Demo](https://ai-powered-pdf-summarizer.streamlit.app/) • [Features](#-features) • [Quick Start](#-quick-start)

</div>

---

## Overview

A production-ready document intelligence tool I built to demonstrate full-stack AI integration. This application processes PDF documents into concise AI-generated summaries using Google's Gemini 2.5 Flash Lite model, with intelligent optimizations to operate entirely within the free tier.

**Key Achievements:**
-  **1 API call per PDF** - Optimized from 8+ calls through intelligent caching
-  **1,500 PDFs/day capacity** - Efficient quota management
-  **$0 infrastructure cost** - Free tier optimization
-  **Secure deployment** - API key protection best practices

**Use Cases:**
-  Quick summarization of research papers and textbooks
-  Business document analysis
-  Study material condensation
-  Document intelligence workflows

##  Features

### Core Functionality
- ** Smart PDF Upload** - 5MB max, automatic format validation
- ** AI Summarization** - Powered by Google Gemini 2.5 Flash Lite
- ** Lightning Fast** - Process documents in <15 seconds
- ** Intelligent Caching** - Same PDF = 0 API calls on reload
- ** Document Statistics** - Word count, page count, metadata

### Free Tier Optimizations
- **✅ 10-Page Limit** - Processes first 10 pages to minimize token usage
- **✅ Rate Limiting** - 4-second delays ensure <15 requests/min
- **✅ Quota Tracking** - Visual display of daily request usage
- **✅ One-Shot Summarization** - Single API call per document
- **✅ Smart Warnings** - Alerts at 50+ requests

### Bonus Features (No API Calls)
-  Extracted text preview
-  Download summary as TXT
-  Copy to clipboard
-  Real-time usage statistics

##  Architecture

```
User uploads PDF (5MB max)
    ↓
Extract first 10 pages (PyPDF2)
    ↓
Limit to 50K characters
    ↓
Check cache (st.session_state)
    ↓
[Cache Hit] → Return cached summary (0 API calls)
    ↓
[Cache Miss] → Rate limit (4s delay)
    ↓
Generate summary (Gemini API)
    ↓
Cache result + Display
```

##  Usage

1. **Upload** a PDF document (max 5MB)
2. **Review** document statistics (pages, word count)
3. **Click** "Generate Summary" button
4. **Get** a concise 3-point AI summary
5. **Download** or copy the summary

### Example Output

For a 15-page business report:
```
• The document analyzes Q4 revenue trends showing 23% growth in digital services
• Key recommendations include expanding customer support teams and investing in automation
• Main challenges identified: supply chain delays and increased competition in emerging markets
```

##  Technology Stack

| Technology | Purpose | Why This Choice |
|------------|---------|-----------------|
| **Streamlit** | Web UI framework | Rapid development, Python-native, free hosting |
| **Google Gemini 2.5 Flash Lite** | AI summarization | Free tier (1500/day), fastest model, optimized for speed |
| **PyPDF2** | PDF text extraction | Lightweight, reliable, no dependencies |
| **python-dotenv** | Environment management | Secure API key handling |

##  Free Tier Limits & Solutions

| Limitation | Strategy Implemented |
|------------|---------------------|
| 15 requests/min | 4-second delay between calls |
| 1,500 requests/day | Visual quota tracker + warnings |
| No persistent storage | Streamlit session state caching |
| API key exposure | `.gitignore` + `secrets.toml` |


## 👨‍💻 Author

**SAGAR**
- GitHub: [@SAGAR-0000](https://github.com/SAGAR-0000)
- Project: [AI-Powered PDF Summarizer](https://github.com/SAGAR-0000/AI-Powered-PDF-Summarizer)

##  Contributing

This is a personal portfolio project. If you find it useful:
- ⭐ Star the repository
-  Report bugs via Issues
-  Suggest features
-  Fork for your own projects (MIT License)

##  License

MIT License - Copyright (c) 2025 SAGAR

Feel free to use this project as inspiration for your own portfolio!

##  Links

- [Live Demo](https://ai-powered-pdf-summarizer.streamlit.app/)
- [Google Gemini API](https://makersuite.google.com/app/apikey)

##  Acknowledgments

- Google for the free Gemini API tier
- Streamlit for free cloud hosting
- The open-source community

---

<div align="center">

**Built with ❤️ by SAGAR using Streamlit & Google Gemini 2.5 Flash Lite**

Personal Portfolio Project | 2025

⭐ Star this repo if you find it useful! ⭐

</div>
