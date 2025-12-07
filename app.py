"""
AI-Powered PDF Summarizer
Secure, optimized version with improved error handling and performance
"""

import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import time
import hashlib
import re
from datetime import datetime
from collections import Counter
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

# Constants
MAX_PAGES = 10
MAX_TEXT_LENGTH = 35000
MAX_FILE_SIZE_MB = 5
MAX_DAILY_REQUESTS = 1500
RATE_LIMIT_DELAY_SECONDS = 4
CHUNK_SIZE_CHARS = 2500
MAX_CHUNKS = 3

# Pre-compile regex patterns for performance
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s.,;:()\-\n]')
URL_PATTERN = re.compile(r'http[s]?://\S+')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
WHITESPACE_PATTERN = re.compile(r'\s+')
LONG_WORD_PATTERN = re.compile(r'\b\w{30,}\b')

class SummaryType(Enum):
    """Enumeration of summary generation methods"""
    AI = "ai"
    KEYWORD = "keyword"

class ErrorType(Enum):
    """Categorized error types for better handling"""
    QUOTA_EXHAUSTED = "quota_exhausted"
    INVALID_API_KEY = "invalid_api_key"
    NETWORK_ERROR = "network_error"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"

@dataclass
class ProcessingResult:
    """Data class for PDF processing results"""
    text: str
    total_pages: int
    pages_processed: int
    summary: Optional[str] = None
    api_calls_used: int = 0
    summary_type: SummaryType = SummaryType.KEYWORD
    error_type: Optional[ErrorType] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Page configuration
st.set_page_config(
    page_title="PDF Summarizer - Secure Edition",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with defaults
def initialize_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        'request_count': 0,
        'last_request_time': None,
        'processed_pdfs': {},
        'last_error': None,
        'error_type': None,
        'gemini_model': None,
        'model_initialized': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

def categorize_error(error_message: str) -> ErrorType:
    """Categorize error messages for appropriate handling"""
    error_lower = error_message.lower()
    
    if any(keyword in error_lower for keyword in 
           ['quota', '429', 'resource', 'exhausted', 'resourceexhausted']):
        return ErrorType.QUOTA_EXHAUSTED
    elif any(keyword in error_lower for keyword in 
             ['api key', 'invalid', '401', '403', 'unauthorized']):
        return ErrorType.INVALID_API_KEY
    elif any(keyword in error_lower for keyword in 
             ['network', 'connection', 'timeout', 'unreachable']):
        return ErrorType.NETWORK_ERROR
    elif any(keyword in error_lower for keyword in 
             ['candidate', 'finish_reason', 'safety', 'filter']):
        return ErrorType.CONTENT_FILTER
    else:
        return ErrorType.UNKNOWN

def get_error_message(error_type: ErrorType) -> Tuple[str, str]:
    """Get user-friendly error message and suggestion"""
    messages = {
        ErrorType.QUOTA_EXHAUSTED: (
            "🚫 API quota exhausted. Switching to keyword-only mode.",
            "💡 Quota resets at midnight UTC. Enable 'Skip AI (Keyword Mode)' checkbox."
        ),
        ErrorType.INVALID_API_KEY: (
            "⚠️ Invalid API key. Please check your configuration.",
            "💡 Get a free API key from: https://makersuite.google.com/app/apikey"
        ),
        ErrorType.NETWORK_ERROR: (
            "🌐 Network error occurred.",
            "💡 Check your internet connection and retry."
        ),
        ErrorType.CONTENT_FILTER: (
            "⚠️ Content was filtered by AI safety systems.",
            "💡 Using keyword-based analysis instead."
        ),
        ErrorType.UNKNOWN: (
            "❌ An unexpected error occurred.",
            "💡 Using keyword-based analysis as fallback."
        )
    }
    return messages.get(error_type, messages[ErrorType.UNKNOWN])

def validate_file_upload(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validate uploaded file meets requirements"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large: {file_size_mb:.2f}MB (max {MAX_FILE_SIZE_MB}MB)"
    
    # Check file type
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Invalid file type. Only PDF files are supported."
    
    # Basic content validation
    if len(uploaded_file.getvalue()) < 100:
        return False, "File appears to be empty or corrupted"
    
    return True, None

def generate_file_hash(file_content: bytes) -> str:
    """Generate secure hash for file content"""
    return hashlib.sha256(file_content).hexdigest()



def init_gemini() -> Optional[genai.GenerativeModel]:
    """
    Initialize Gemini API with error handling and fallback logic.
    Returns cached model if already initialized.
    """
    # Return cached model if available
    if st.session_state.model_initialized and st.session_state.gemini_model:
        return st.session_state.gemini_model
    
    try:
        # Get API key from secrets
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        
        if not api_key or api_key == "your-api-key-here":
            st.session_state.error_type = ErrorType.INVALID_API_KEY
            error_msg, suggestion = get_error_message(ErrorType.INVALID_API_KEY)
            st.error(error_msg)
            st.info(suggestion)
            return None
        
        genai.configure(api_key=api_key)
        
        # Try latest model first (no test calls - quota errors caught during actual use)
        model_names = ['gemini-2.5-flash-lite', 'gemini-2.5-flash']
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                
                # Cache and return model (no test call to save quota)
                st.session_state.gemini_model = model
                st.session_state.model_initialized = True
                return model
                    
            except Exception as model_error:
                error_type = categorize_error(str(model_error))
                
                # Only stop on quota errors
                if error_type == ErrorType.QUOTA_EXHAUSTED:
                    st.session_state.error_type = error_type
                    error_msg, suggestion = get_error_message(error_type)
                    st.error(error_msg)
                    st.info(suggestion)
                    return None
                
                # Try next model if current one fails
                continue
        
        # All models failed
        st.session_state.error_type = ErrorType.UNKNOWN
        st.warning("⚠️ Could not initialize AI model. Using keyword-only mode.")
        return None
                
    except Exception as e:
        error_type = categorize_error(str(e))
        st.session_state.error_type = error_type
        error_msg, suggestion = get_error_message(error_type)
        st.error(error_msg)
        st.info(suggestion)
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = MAX_PAGES) -> Tuple[Optional[str], int, int]:
    """
    Extract text from PDF with memory-efficient processing.
    Cached for 1 hour to improve performance.
    """
    try:
        # Use BytesIO to avoid file system operations
        from io import BytesIO
        pdf_file = BytesIO(pdf_bytes)
        
        pdf_reader = PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        
        if total_pages == 0:
            st.error("❌ PDF appears to be empty or corrupted.")
            return None, 0, 0
        
        pages_to_process = min(total_pages, max_pages)
        text_parts = []
        
        # Process pages with progress indicator
        for page_num in range(pages_to_process):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:  # Only add non-empty text
                    text_parts.append(page_text)
            except Exception as page_error:
                st.warning(f"⚠️ Could not extract text from page {page_num + 1}")
                continue
        
        if not text_parts:
            st.error("❌ No text could be extracted from PDF.")
            return None, total_pages, 0
        
        text = "\n\n".join(text_parts)
        
        # Limit total text length
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            st.warning(f"⚠️ Text truncated to {MAX_TEXT_LENGTH:,} characters")
        
        return text, total_pages, pages_to_process
        
    except Exception as e:
        st.error(f"❌ Error extracting text: {type(e).__name__}")
        return None, 0, 0

def sanitize_text(text: str) -> str:
    """
    Clean and sanitize text to avoid potential issues.
    Uses pre-compiled regex patterns for performance.
    """
    # Remove special characters
    text = SPECIAL_CHARS_PATTERN.sub(' ', text)
    # Remove URLs
    text = URL_PATTERN.sub('', text)
    # Remove emails
    text = EMAIL_PATTERN.sub('', text)
    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(' ', text)
    # Remove very long words (likely corrupted)
    text = LONG_WORD_PATTERN.sub('', text)
    
    return text.strip()

def enforce_rate_limit():
    """Implement rate limiting between API calls"""
    if st.session_state.last_request_time:
        elapsed = time.time() - st.session_state.last_request_time
        if elapsed < RATE_LIMIT_DELAY_SECONDS:
            wait_time = RATE_LIMIT_DELAY_SECONDS - elapsed
            time.sleep(wait_time)
    st.session_state.last_request_time = time.time()

def extract_keywords(text: str, top_n: int = 15) -> List[Tuple[str, int]]:
    """Extract most common keywords using NLP techniques"""
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
        'those', 'it', 'its', 'from', 'by', 'as', 'which', 'who', 'what',
        'where', 'when', 'why', 'how', 'not', 'no', 'yes', 'also', 'into',
        'than', 'them', 'then', 'there', 'their', 'about', 'after', 'before',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'only', 'own', 'same', 'so', 'too', 'very', 'just', 'can',
        'chapter', 'figure', 'table', 'page', 'section', 'using', 'used'
    }
    
    words = text.lower().split()
    filtered_words = [
        word.strip('.,!?;:()[]{}""\'') 
        for word in words 
        if len(word) > 4 and word not in stop_words and word.isalpha()
    ]
    
    return Counter(filtered_words).most_common(top_n)

def generate_ai_summary(text: str, model: genai.GenerativeModel) -> Tuple[Optional[str], int]:
    """
    Generate AI summary using multiple fallback strategies.
    Returns (summary, api_calls_used)
    """
    enforce_rate_limit()
    
    # Sanitize text
    clean_text = sanitize_text(text)
    text_snippet = clean_text[:CHUNK_SIZE_CHARS]
    
    # Progressive prompt strategies
    prompts = [
        f"Summarize this document concisely:\n\n{text_snippet}",
        f"What are the main topics in this text?\n\n{text_snippet}",
        f"List the key points:\n\n{text_snippet}",
    ]
    
    for attempt, prompt in enumerate(prompts):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=600,
                    top_p=0.95,
                    top_k=50
                )
            )
            
            if hasattr(response, 'text') and response.text and len(response.text.strip()) > 15:
                st.session_state.request_count += 1
                if attempt > 0:
                    st.info(f"✅ Success with strategy {attempt + 1}")
                return response.text, 1
                
        except Exception as e:
            error_type = categorize_error(str(e))
            st.session_state.error_type = error_type
            
            # Stop immediately on quota errors
            if error_type == ErrorType.QUOTA_EXHAUSTED:
                error_msg, suggestion = get_error_message(error_type)
                st.error(error_msg)
                st.info(suggestion)
                return None, attempt + 1
            
            # Try chunked approach on content filter
            if error_type == ErrorType.CONTENT_FILTER and attempt == len(prompts) - 1:
                st.info("⚠️ Trying chunked processing strategy...")
                chunked_summary = generate_chunked_summary(clean_text, model)
                if chunked_summary:
                    return chunked_summary, attempt + 2
            
            # Continue to next strategy
            continue
    
    return None, len(prompts)

def generate_chunked_summary(text: str, model: genai.GenerativeModel) -> Optional[str]:
    """
    Process text in smaller chunks to avoid content filters.
    Combines individual chunk summaries.
    """
    try:
        chunk_size = len(text) // MAX_CHUNKS
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)][:MAX_CHUNKS]
        
        partial_summaries = []
        
        for i, chunk in enumerate(chunks):
            try:
                enforce_rate_limit()
                
                response = model.generate_content(
                    f"Extract key facts:\n\n{chunk[:2000]}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=200
                    )
                )
                
                if response.text:
                    partial_summaries.append(response.text)
                    st.session_state.request_count += 1
            except:
                continue
        
        if partial_summaries:
            combined = "\n\n".join([
                f"**Part {i+1}:** {summary}" 
                for i, summary in enumerate(partial_summaries)
            ])
            return combined
            
        return None
        
    except Exception as e:
        return None

def generate_keyword_summary(text: str) -> str:
    """Generate intelligent summary using NLP keyword extraction"""
    keywords = extract_keywords(text, top_n=25)
    
    # Extract meaningful sentences
    sentences = [
        s.strip() 
        for s in re.split(r'[.!?]+', text) 
        if len(s.strip()) > 40 and len(s.split()) > 5
    ]
    
    first_paragraph = ' '.join(sentences[:2]) if len(sentences) >= 2 else (sentences[0] if sentences else "")
    
    # Determine document type
    text_lower = text.lower()
    if any(term in text_lower for term in ['research', 'study', 'methodology']):
        doc_type = "Research Paper"
    elif any(term in text_lower for term in ['chapter', 'section', 'exercise']):
        doc_type = "Educational Material"
    elif any(term in text_lower for term in ['company', 'revenue', 'quarterly']):
        doc_type = "Business Document"
    else:
        doc_type = "General Document"
    
    # Build summary
    summary_parts = [
        f"**📚 {doc_type} Analysis**\n",
        "**Document Overview:**",
        f"{first_paragraph[:300]}{'...' if len(first_paragraph) > 300 else ''}\n",
        "**Core Topics:**"
    ]
    
    for i, (word, count) in enumerate(keywords[:6], 1):
        frequency = "frequently" if count > 15 else "moderately" if count > 8 else "occasionally"
        summary_parts.append(f"{i}. **{word.title()}** — {frequency} ({count}×)")
    
    summary_parts.append("\n**Related Concepts:**")
    related = ', '.join([w.title() for w, _ in keywords[6:15]])
    summary_parts.append(related)
    
    # Document statistics
    total_words = len(text.split())
    avg_word_length = sum(len(w) for w in text.split()) / max(total_words, 1)
    complexity = "Advanced" if avg_word_length > 6 else "Intermediate" if avg_word_length > 5 else "Basic"
    
    summary_parts.extend([
        "\n**Document Characteristics:**",
        f"• Complexity: {complexity}",
        f"• Word Count: {total_words:,}",
        f"• Avg Word Length: {avg_word_length:.1f} characters",
        "\n_Generated using keyword-based analysis_"
    ])
    
    return "\n".join(summary_parts)

def calculate_usage_percentage() -> float:
    """Calculate API usage as percentage of daily limit"""
    return min(100.0, (st.session_state.request_count / MAX_DAILY_REQUESTS) * 100)

def render_sidebar():
    """Render sidebar with information and controls"""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"""
        **Features:**
        - First {MAX_PAGES} pages processed
        - Smart caching system
        - Multiple fallback strategies
        - {MAX_FILE_SIZE_MB}MB file size limit
        - Secure error handling
        
        **Daily Limits:**
        - {MAX_DAILY_REQUESTS:,} API requests
        - 15 requests/minute
        """)
        
        st.divider()
        
        st.header("📊 Usage Statistics")
        requests_remaining = max(0, MAX_DAILY_REQUESTS - st.session_state.request_count)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Calls Today", st.session_state.request_count)
        with col2:
            st.metric("Remaining", requests_remaining)
        
        usage_percent = calculate_usage_percentage()
        st.progress(usage_percent / 100)
        
        # Usage warnings
        if usage_percent >= 97:
            st.error("🚨 Quota nearly exhausted! Use Keyword Mode.")
        elif usage_percent >= 93:
            st.warning("⚠️ Approaching limit (93%+)")
        elif usage_percent >= 80:
            st.info("💡 80% of daily quota used")
        
        st.divider()
        
        st.header("🛠️ Options")
        
        skip_ai = st.checkbox(
            "Skip AI (Keyword Mode)",
            value=False,
            help="Use intelligent keyword extraction (0 API calls)"
        )
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.processed_pdfs = {}
            st.session_state.model_initialized = False
            st.session_state.gemini_model = None
            st.success("✅ Cache cleared!")
            st.rerun()
        
        return skip_ai

def render_file_upload():
    """Render file upload section with validation"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help=f"Maximum {MAX_FILE_SIZE_MB}MB, first {MAX_PAGES} pages processed"
        )
    
    with col2:
        if uploaded_file:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("File Size", f"{file_size_mb:.2f} MB")
    
    return uploaded_file

def process_pdf(uploaded_file, skip_ai: bool = False) -> Optional[ProcessingResult]:
    """Main PDF processing pipeline"""
    # Validate file
    is_valid, error_message = validate_file_upload(uploaded_file)
    if not is_valid:
        st.error(f"❌ {error_message}")
        return None
    
    # Generate cache key
    file_content = uploaded_file.getvalue()
    content_hash = generate_file_hash(file_content)
    cache_key = f"{uploaded_file.name}_{content_hash[:16]}"
    
    # Check cache
    if cache_key in st.session_state.processed_pdfs:
        cached = st.session_state.processed_pdfs[cache_key]
        
        # If user wants AI but cache is keyword-only, allow retry
        if not skip_ai and cached.summary_type == SummaryType.KEYWORD:
            col1, col2 = st.columns([3, 1])
            with col1:
                error_msg = cached.error_type.value if cached.error_type else "unknown"
                st.warning(f"⚠️ Cached keyword analysis (AI failed: {error_msg})")
            with col2:
                if st.button("🔄 Retry AI"):
                    del st.session_state.processed_pdfs[cache_key]
                    st.rerun()
        else:
            st.success("✨ Using cached results (0 API calls)")
        
        return cached
    
    # Extract text
    with st.spinner("📖 Extracting text..."):
        text, total_pages, pages_processed = extract_text_from_pdf(file_content)
    
    if not text:
        return None
    
    if total_pages > MAX_PAGES:
        st.warning(f"📌 Processing first {pages_processed} of {total_pages} pages")
    else:
        st.success(f"✅ Extracted {pages_processed} pages")
    
    # Generate summary
    summary = None
    api_calls = 0
    summary_type = SummaryType.KEYWORD
    error_type = None
    
    if skip_ai:
        with st.spinner("🔍 Analyzing with NLP (0 API calls)..."):
            summary = generate_keyword_summary(text)
    else:
        model = init_gemini()
        
        if model is None:
            error_type = st.session_state.error_type
            summary = generate_keyword_summary(text)
        else:
            with st.spinner("🤖 Generating AI summary..."):
                summary, api_calls = generate_ai_summary(text, model)
            
            if summary:
                summary_type = SummaryType.AI
            else:
                error_type = st.session_state.error_type
                error_msg, _ = get_error_message(error_type)
                st.warning(f"⚠️ {error_msg}")
                summary = generate_keyword_summary(text)
    
    # Create result object
    result = ProcessingResult(
        text=text,
        total_pages=total_pages,
        pages_processed=pages_processed,
        summary=summary,
        api_calls_used=api_calls,
        summary_type=summary_type,
        error_type=error_type
    )
    
    # Cache result
    st.session_state.processed_pdfs[cache_key] = result
    
    return result

def render_document_stats(result: ProcessingResult):
    """Render document statistics"""
    st.subheader("📊 Document Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pages", result.total_pages)
    with col2:
        st.metric("Processed", result.pages_processed)
    with col3:
        word_count = len(result.text.split())
        st.metric("Words", f"{word_count:,}")
    with col4:
        delta_text = "Cached" if result.api_calls_used == 0 else None
        st.metric("API Calls", result.api_calls_used, delta=delta_text)

def render_summary(result: ProcessingResult, filename: str):
    """Render summary with download option"""
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📝 Summary")
    with col2:
        if result.api_calls_used > 0:
            st.success(f"✅ {result.api_calls_used} API call(s)")
        else:
            st.success("✅ 0 API calls")
    
    st.markdown(result.summary)
    
    # Generate download content
    summary_content = f"""PDF SUMMARY
{'='*50}

File: {filename}
Generated: {result.timestamp}
Pages: {result.pages_processed}/{result.total_pages}
Words: {len(result.text.split()):,}
API Calls: {result.api_calls_used}
Summary Type: {result.summary_type.value}

{'='*50}

{result.summary}

{'='*50}
AI-Powered PDF Summarizer
Secure Educational Content Analysis Tool
"""
    
    # Sanitize filename
    safe_filename = re.sub(r'[^\w\s-]', '', filename.replace('.pdf', ''))
    safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
    
    st.download_button(
        label="⬇️ Download Summary",
        data=summary_content.encode('utf-8'),
        file_name=f"{safe_filename}_summary.txt",
        mime="text/plain",
        use_container_width=True
    )

def render_text_preview(text: str, preview_length: int = 3000):
    """Render expandable text preview"""
    with st.expander("👁️ View Extracted Text", expanded=False):
        display_text = text[:preview_length]
        if len(text) > preview_length:
            display_text += "..."
        
        st.text_area(
            "Text Preview",
            value=display_text,
            height=300,
            disabled=True,
            key="text_preview"
        )
        
        if len(text) > preview_length:
            st.caption(f"Showing first {preview_length:,} of {len(text):,} characters")

def render_welcome_screen():
    """Render welcome screen when no file is uploaded"""
    st.info("👆 Upload a PDF to get started")
    
    st.markdown("""
    ### How It Works:
    
    1. **Upload** your PDF document (max 5MB)
    2. **Extract** text from the first 10 pages
    3. **Generate** AI summary or keyword analysis
    4. **Download** or copy the results
    
    ### Features:
    - 🚀 Smart caching for faster repeated processing
    - 🔄 Multiple AI processing strategies with fallbacks
    - 🔑 Zero-API-call keyword extraction mode
    - 🔒 Secure file handling and error management
    - 📊 Comprehensive usage tracking
    
    ### Best For:
    - 📚 Textbooks and course materials
    - 🔬 Research papers and academic articles
    - 📖 Technical documentation
    - 📝 Study guides and notes
    - 📄 Business reports and presentations
    
    ### Security & Privacy:
    - Files are processed in memory (not saved to disk)
    - API keys stored securely in Streamlit secrets
    - No data retention after session ends
    - Content sanitization before AI processing
    """)

# Main Application
def main():
    """Main application entry point"""
    # Header
    st.markdown('<h1 class="main-header">📄 AI PDF Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Secure & Optimized for Educational Content</p>', unsafe_allow_html=True)
    
    # Render sidebar and get options
    skip_ai = render_sidebar()
    
    # File upload
    uploaded_file = render_file_upload()
    
    if uploaded_file is None:
        render_welcome_screen()
    else:
        # Process PDF
        result = process_pdf(uploaded_file, skip_ai)
        
        if result:
            # Display statistics
            render_document_stats(result)
            
            # Generate button or display cached result
            if result.summary:
                render_summary(result, uploaded_file.name)
                render_text_preview(result.text)
            else:
                st.divider()
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                        st.rerun()
                
                with col2:
                    estimated_calls = 0 if skip_ai else 3
                    calls_after = st.session_state.request_count + estimated_calls
                    st.metric("Est. Calls", estimated_calls)
                    st.caption(f"After: {calls_after}/{MAX_DAILY_REQUESTS}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with Streamlit & Google Gemini AI | 
        <a href='https://makersuite.google.com/app/apikey' target='_blank' rel='noopener noreferrer'>Get API Key</a> | 
        Secure & Privacy-Focused</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()