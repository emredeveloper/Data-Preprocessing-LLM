import logging
import requests
import re
from bs4 import BeautifulSoup
import json
import ollama
import pandas as pd
from datetime import datetime
import os
import time
import glob
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import PyPDF2
import io
import tempfile
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


def _build_error_response(source, message, error_type, exception=None):
    """Create a structured error response for API consumers."""
    error_payload = {
        "error": True,
        "source": source,
        "type": error_type,
        "message": message,
    }
    if exception is not None:
        error_payload["details"] = str(exception)
    return error_payload


# Add these helper functions after imports section

def get_theme_for_category(category):
    """Generate a consistent color theme based on the paper category"""
    # Define base themes for common categories
    category_themes = {
        "stat.ML": {"primary": "#4361ee", "secondary": "#3f37c9", "accent": "#f72585"},
        "cs.LG": {"primary": "#2b9348", "secondary": "#007f5f", "accent": "#55a630"},
        "cs.AI": {"primary": "#7209b7", "secondary": "#560bad", "accent": "#480ca8"},
        "cs.CV": {"primary": "#f77f00", "secondary": "#e36414", "accent": "#d62828"},
        "cs.CL": {"primary": "#4cc9f0", "secondary": "#4895ef", "accent": "#4361ee"},
        "math.ST": {"primary": "#7678ed", "secondary": "#3d348b", "accent": "#7678ed"},
    }
    
    # Extract the first category if multiple are present
    primary_category = category.split(',')[0].strip()
    
    # Return the theme if it exists for this category
    if primary_category in category_themes:
        return category_themes[primary_category]
    
    # Generate a theme based on the hash of the category name for consistency
    import hashlib
    hash_val = int(hashlib.md5(primary_category.encode()).hexdigest(), 16)
    
    # Generate colors from the hash
    hue = hash_val % 360
    primary_hsl = f"hsl({hue}, 80%, 50%)"
    secondary_hsl = f"hsl({(hue + 30) % 360}, 80%, 40%)"
    accent_hsl = f"hsl({(hue + 180) % 360}, 80%, 60%)"
    
    return {
        "primary": primary_hsl,
        "secondary": secondary_hsl,
        "accent": accent_hsl,
        "generated": True  # Flag that this was auto-generated
    }

def estimate_reading_time(text, wpm=200):
    """Estimate reading time based on text length"""
    if not text:
        return "5 min"
    
    word_count = len(text.split())
    minutes = max(1, round(word_count / wpm))
    
    if minutes < 60:
        return f"{minutes} min"
    else:
        hours = minutes // 60
        remaining_mins = minutes % 60
        if remaining_mins > 0:
            return f"{hours}h {remaining_mins}m"
        return f"{hours}h"

def get_citation_estimate(arxiv_id):
    """Return citation count for paper (uses a placeholder range for demo)"""
    # In a real implementation, you might query a citation database
    # For demo purposes, we'll generate a consistent but fake citation count
    import hashlib
    hash_val = int(hashlib.md5(arxiv_id.encode()).hexdigest(), 16)
    
    # Generate numbers that seem plausible based on paper age
    year_part = arxiv_id.split('.')[0]
    if len(year_part) >= 2:
        year_prefix = year_part[:2]
        if year_prefix.isdigit():
            year = int(year_prefix)
            # Older papers tend to have more citations
            if year < 20:  # Pre-2020
                base = 50 + (hash_val % 200)
            else:  # 2020 or newer
                base = 5 + (hash_val % 50)
            
            # Add some variance
            variance = hash_val % 10
            return base + variance
    
    # Default fallback
    return hash_val % 50

def extract_keywords_from_text(text, max_keywords=5):
    """Extract keywords from text for tagging"""
    if not text:
        return []
    
    # Use NLTK for keyword extraction
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    # Common ML terms to keep even if they're stop words
    ml_terms = {'ai', 'ml', 'deep', 'learning', 'neural', 'network', 'model', 'algorithm'}
    # Filter tokens
    keywords = [
        word for word in tokens 
        if (word.isalnum() and len(word) > 2 and (word not in stop_words or word in ml_terms))
    ]
    # Count frequencies
    keyword_freq = Counter(keywords)
    # Return top keywords
    return [word for word, _ in keyword_freq.most_common(max_keywords)]

def check_has_code(text):
    """Check if the paper likely has associated code"""
    code_indicators = [
        'github', 'code', 'implementation', 'source code', 
        'pytorch', 'tensorflow', 'jax', 'repository', 
        'available at', 'open source', 'python'
    ]
    text_lower = text.lower()
    for indicator in code_indicators:
        if indicator in text_lower:
            return True
    
    return False

def generate_thumbnail_url(arxiv_id, category):
    """Generate a thumbnail image URL for the paper"""
    # For a real implementation, you might generate thumbnails from the PDF
    # For now, we'll return a placeholder image based on category
    # Placeholder images based on category
    category_images = {
        "stat.ML": "ml_paper.jpg",
        "cs.LG": "deep_learning.jpg",
        "cs.AI": "ai_paper.jpg",
        "cs.CV": "computer_vision.jpg",
        "cs.CL": "nlp_paper.jpg",
        "math.ST": "statistics_paper.jpg"
    }
    
    # Extract the primary category
    primary_category = category.split(',')[0].strip()
    # Get the appropriate image    
    if primary_category in category_images:
        image_name = category_images[primary_category]    
    else:
        # Default image for other categories
        image_name = "generic_paper.jpg"
    
    # Create a full URL (adjust the path to match your static files location)
    return f"/static/images/papers/{image_name}"

# Download necessary NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def fetch_arxiv_papers(url="https://arxiv.org/list/stat.ML/recent", num_papers=20):
    """Fetch recent papers from arXiv's stat.ML category"""
    logger.info(f"Fetching papers from {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        message = f"Request to {url} timed out."
        logger.error(message)
        return _build_error_response("fetch_arxiv_papers", message, "timeout", exc)
    except requests.exceptions.RequestException as exc:
        message = f"Unable to fetch papers from arXiv: {exc}"
        logger.error(message)
        return _build_error_response("fetch_arxiv_papers", message, "network", exc)

    soup = BeautifulSoup(response.text, 'html.parser')
    papers = []
    dt_elements = soup.find_all('dt')
    
    for i, dt in enumerate(dt_elements):
        if i >= num_papers:
            break
            
        dd = dt.find_next_sibling('dd')
        
        if not dd:
            continue
            
        arxiv_id_match = re.search(r'arXiv:(\d+\.\d+)', dt.text)
        if not arxiv_id_match:
            continue
        arxiv_id = arxiv_id_match.group(1)
        title_element = dd.find('div', class_='list-title')
        title = title_element.text.replace('Title:', '').strip() if title_element else "No title found"
        
        authors_element = dd.find('div', class_='list-authors')
        authors = authors_element.text.replace('Authors:', '').strip() if authors_element else "No authors found"
        
        abstract_element = dd.find('p', class_='mathjax')
        abstract = abstract_element.text.strip() if abstract_element else "No abstract found"
        
        pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        description = "No description available"
        if abstract and abstract != "No abstract found":
            sentences = abstract.split('. ')        
            if len(sentences) > 1:
                description = '. '.join(sentences[:2]) + '.'
            else:
                description = abstract[:200] + ('...' if len(abstract) > 200 else '')
        
        categories = dd.find('div', class_='list-subjects')
        category = categories.text.replace('Subjects:', '').strip() if categories else "stat.ML"
        date_element = dd.find('div', class_='list-date')
        pub_date = date_element.text.replace('Submitted:', '').strip() if date_element else datetime.now().strftime("%d %b %Y")
        
        # Generate UI elements for modern interface
        ui_theme = get_theme_for_category(category)
        reading_time = estimate_reading_time(abstract) if abstract else "5 min"
        citation_count = get_citation_estimate(arxiv_id)
        
        papers.append({
            'id': arxiv_id,
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'description': description,
            'pdf_link': pdf_link,
            'category': category,
            'pub_date': pub_date,
            'arxiv_link': f"https://arxiv.org/abs/{arxiv_id}",
            # Modern UI enhancements
            'ui_theme': ui_theme,
            'reading_time': reading_time,
            'citation_count': citation_count,
            'tag_keywords': extract_keywords_from_text(title + " " + abstract, max_keywords=5),
            'has_code': check_has_code(title + " " + abstract),
            'thumbnail': generate_thumbnail_url(arxiv_id, category)
        })
    
    print(f"Fetched {len(papers)} papers")
    return papers

def extract_keywords_from_titles(titles):
    """Extract and rank keywords from paper titles"""
    all_text = ' '.join(titles).lower()
    tokens = word_tokenize(all_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        word for word in tokens 
        if word.isalnum() and 
        word not in stop_words and 
        len(word) > 2
    ]
    word_freq = Counter(filtered_tokens)
    return word_freq.most_common(20)

def analyze_with_ollama(papers, model="gemma3", depth="title"):
    """Use Ollama to analyze paper titles and identify trends"""
    if not papers:
        return "No papers to analyze", None
    
    valid_models = ['gemma3', 'deepseek-r1:1.5b', 'qwen2.5:7b']
    if model not in valid_models:
        model = "gemma3"
    current_date = datetime.now().strftime("%B %d, %Y")
    paper_contexts = None
    
    if depth == "title":
        titles = [paper['title'] for paper in papers]
        titles_text = "\n".join(titles)
        keywords = extract_keywords_from_titles(titles)
        top_keywords = ", ".join([k for k, _ in keywords[:10]])
        
        if model == "deepseek-r1:1.5b":
            prompt = f"""
            Analyze these {len(titles)} paper titles from arXiv stat.ML (today: {current_date}):
            {titles_text}   
            Provide a concise analysis (300 words) of:
            1. Main research themes
            2. Common methods/algorithms
            3. Emerging directions
            """
        else:
            prompt = f"""
            You are a research analyst specializing in machine learning. 
            Analyze the following {len(titles)} recent paper titles from arXiv's stat.ML category:
            
            {titles_text}
            
            Today's date is {current_date}.
            The top keywords extracted from these papers are: {top_keywords}
            
            Provide a concise analysis (around 400 words) covering:
            1. Main research themes and trends with specific examples from the papers
            2. Detailed discussion of methods, algorithms, and technical approaches
            3. Connections between the papers and how they relate to each other
            4. How these papers extend or challenge previous research in ML
            
            Format your response as a proper analysis report with sections.        
            Include the current date ({current_date}) in your heading.
            Your analysis should be more detailed and technical than if you had only seen the titles.
            """
        print("Analyzing papers with Ollama (title-only)...")
    else:
        print("Retrieving full paper content for RAG analysis...")
        paper_contexts = get_paper_contexts(papers)
        context_texts = []
        for i, (paper, context) in enumerate(zip(papers, paper_contexts)):
            context_texts.append(f"Paper {i+1}: {paper['title']}\nExcerpt: {context['excerpt']}")
        
        full_context = "\n\n".join(context_texts)
        
        if model == "deepseek-r1:1.5b":
            prompt = f"""
            Analyze these {len(papers)} papers from arXiv stat.ML (today: {current_date}):
            
            {full_context}
            
            Provide a technical analysis (400 words) covering:
            1. Main research themes with examples
            2. Methods and approaches used
            3. How they relate to each other
            """
        else:
            prompt = f"""
            You are a research analyst specializing in machine learning. 
            Analyze the following {len(papers)} recent papers from arXiv's stat.ML category.
            I've provided excerpts from each paper's content below:
            
            {full_context}
            
            Today's date is {current_date}.
            
            Based on detailed analysis of these papers, provide a concise research report (around 500 words) covering:
            1. Main research themes and trends with specific examples from the papers
            2. Detailed discussion of methods, algorithms, and technical approaches
            3. Connections between the papers and how they relate to each other
            4. How these papers extend or challenge previous research in ML
            
            Format your response as a proper analysis report with sections.
            Include the current date ({current_date}) in your heading.
            Your analysis should be more detailed and technical than if you had only seen the titles.
            """
        print("Analyzing papers with Ollama (RAG)...")
    
    print(f"Analyzing papers with Ollama using {model}...")
    try:
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content'], paper_contexts
    except Exception as e:
        print(f"Error using Ollama with model {model}: {e}")
        return f"Error in analysis with {model}: {str(e)}", None

def extract_text_and_tables(pdf_path, extract_images=False):
    """Extract text and tables from PDF, optionally using OCR for images.
    Returns tuple of (text, tables_text)
    """
    try:
        doc = fitz.open(pdf_path)
        all_text = ""    
        images = []
        
        for page_num in range(min(5, len(doc))):
            page = doc.load_page(page_num)
            all_text += page.get_text()
            
            if extract_images and TESSERACT_AVAILABLE:
                images_info = page.get_images(full=True)
                for img in images_info:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    images.append(image)
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {e}")
        all_text = ""
        images = []
    
    tables_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:5]:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        tables_text += " | ".join(cell if cell else "" for cell in row) + "\n"
                    tables_text += "\n"
    except Exception as e:
        print(f"Error extracting tables with pdfplumber: {e}")
    
    image_text = ""
    if extract_images and TESSERACT_AVAILABLE and images:            
        try:
            for image in images:
                text = pytesseract.image_to_string(image)
                image_text += text + "\n"
            if image_text:
                all_text += "\n\n[Image Text]\n" + image_text
        except Exception as e:
            print(f"Error performing OCR: {e}")
    
    return all_text, tables_text

def get_paper_contexts(papers, max_excerpt_length=1000):
    """Download paper PDFs and extract text for RAG analysis"""
    contexts = []
    
    for paper in papers:
        context = {
            "title": paper['title'],
            "excerpt": "Could not retrieve paper content"
        }

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the PDF
                pdf_path = os.path.join(temp_dir, f"{paper['id']}.pdf")

                logger.info(f"Downloading {paper['pdf_link']}...")
                response = requests.get(paper['pdf_link'], stream=True, timeout=20)
                response.raise_for_status()

                with open(pdf_path, 'wb') as f:
                    f.write(response.content)

                text, tables_text = extract_text_and_tables(pdf_path)
                full_text = text
                if tables_text:
                    full_text += "\n\n[Tables]\n" + tables_text

                excerpt = full_text[:max_excerpt_length]
                context["excerpt"] = excerpt
        except requests.exceptions.Timeout as exc:
            message = f"Downloading PDF for {paper['id']} timed out."
            logger.error(message)
            context["error"] = {
                "type": "timeout",
                "message": message,
                "details": str(exc),
            }
            context["excerpt"] = message
        except requests.exceptions.RequestException as exc:
            message = f"Failed to download PDF for {paper['id']}: {exc}"
            logger.error(message)
            context["error"] = {
                "type": "network",
                "message": message,
                "details": str(exc),
            }
            context["excerpt"] = message
        except Exception as e:
            message = f"Error processing paper {paper['id']}: {str(e)}"
            logger.error(message)
            context["error"] = {
                "type": "processing",
                "message": message,
                "details": str(e),
            }
            context["excerpt"] = message

        contexts.append(context)

    return contexts

def compare_with_previous_analysis(current_papers, current_analysis):
    """Compare current papers and analysis with previous ones"""
    json_files = glob.glob('arxiv_analysis_*.json')
    
    if not json_files:
        return {
            "status": "No previous analyses found for comparison",
            "trend_change": None,
            "keywords_comparison": None
        }
    
    json_files.sort(reverse=True)
    try:
        with open(json_files[0], 'r') as f:
            previous_papers = json.load(f)
    except Exception as e:
        return {
            "status": f"Error loading previous analysis: {str(e)}",
            "trend_change": None,
            "keywords_comparison": None
        }
    
    current_titles = [paper['title'] for paper in current_papers]
    previous_titles = [paper['title'] for paper in previous_papers]
    
    current_keywords = extract_keywords_from_titles(current_titles)
    previous_keywords = extract_keywords_from_titles(previous_titles)
    
    current_top_keywords = set([k for k, _ in current_keywords[:15]])
    previous_top_keywords = set([k for k, _ in previous_keywords[:15]])
    
    new_keywords = current_top_keywords - previous_top_keywords
    disappeared_keywords = previous_top_keywords - current_top_keywords
    
    previous_keyword_dict = {k: (15-i) for i, (k, _) in enumerate(previous_keywords[:15])}
    current_keyword_dict = {k: (15-i) for i, (k, _) in enumerate(current_keywords[:15])}
    
    trending_up = []
    trending_down = []
    for keyword, current_rank_value in current_keyword_dict.items():
        if keyword in previous_keyword_dict:
            previous_rank_value = previous_keyword_dict[keyword]
            change = current_rank_value - previous_rank_value
            if change > 3:
                trending_up.append(keyword)
            elif change < -3:
                trending_down.append(keyword)
    
    txt_filename = json_files[0].replace('.json', '.txt')
    previous_analysis_text = ""
    try:
        if os.path.exists(txt_filename):
            with open(txt_filename, 'r') as f:
                content = f.read()
                if "ANALYSIS:" in content:
                    previous_analysis_text = content.split("ANALYSIS:")[1].strip()
    except Exception as e:
        previous_analysis_text = ""
    
    if previous_analysis_text:
        prompt = f"""
        Compare these two research trend analyses for arXiv ML papers:
        PREVIOUS ANALYSIS:
        {previous_analysis_text}
        CURRENT ANALYSIS:
        {current_analysis}
        What are the main differences in research trends? Keep your response under 200 words.
        Focus on:
        1. New emerging topics
        2. Topics that are no longer trending
        3. Shifts in research focus
        """
        try:
            response = ollama.chat(model="gemma3", messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            trend_comparison = response['message']['content']
        except Exception as e:
            trend_comparison = f"Error generating comparison: {str(e)}"
    else:
        trend_comparison = "No previous analysis text found for comparison"
    
    return {
        "status": "success",
        "previous_date": json_files[0].split('_')[1][:8],
        "new_keywords": list(new_keywords),
        "disappeared_keywords": list(disappeared_keywords),
        "trending_up": trending_up,
        "trending_down": trending_down,
        "trend_comparison": trend_comparison
    }

def save_to_file(papers, analysis, filename=None):
    """Save the papers and analysis to a file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arxiv_analysis_{timestamp}"
    
    with open(f"{filename}.json", "w") as f:
        json.dump(papers, f, indent=2)
    
    with open(f"{filename}.txt", "w") as f:
        f.write("ARXIV STAT.ML PAPERS ANALYSIS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of papers analyzed: {len(papers)}\n\n")
        f.write("PAPER TITLES:\n")
        for i, paper in enumerate(papers, 1):
            f.write(f"{i}. {paper['title']}\n")
        f.write("-" * 40 + "\n")
        f.write("ANALYSIS:\n")
        f.write(analysis)
    print(f"Results saved to {filename}.json and {filename}.txt")

def get_papers_for_app():
    """Get papers for the Flask app or load from a cache file"""
    cache_file = 'paper_cache.json'
    try:
        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < 3600:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        papers = fetch_arxiv_papers(num_papers=20)
        if isinstance(papers, dict) and papers.get("error"):
            return papers
        with open(cache_file, 'w') as f:
            json.dump(papers, f, indent=2)
        return papers
    except Exception as e:
        message = f"Error reading cache: {e}"
        logger.error(message)
        return _build_error_response("get_papers_for_app", message, "cache", e)

def main():
    print("Starting arXiv stat.ML papers analysis...")
    try:
        papers = fetch_arxiv_papers(num_papers=20)
        if isinstance(papers, dict) and papers.get("error"):
            print(papers["message"])
            return
        if not papers:
            print("Failed to fetch papers. Exiting.")
            return

        print("\nFetched Paper Titles:")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title']}")
        
        analysis, paper_contexts = analyze_with_ollama(papers, depth="title")
        print("\nAnalysis Results:")
        print(analysis)
        
        save_to_file(papers, analysis)
        
        comparison = compare_with_previous_analysis(papers, analysis)
        print("\nComparison with Previous Analysis:")
        print(comparison)
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
