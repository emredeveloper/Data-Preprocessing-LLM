import os
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from arxiv_analysis import (
    fetch_arxiv_papers,
    analyze_with_ollama,
    get_papers_for_app,
    compare_with_previous_analysis,
    PaperRetrievalError,
)

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', current_year=datetime.now().year)

@app.route('/api/papers')
def get_papers():
    """API endpoint to get papers"""
    try:
        papers = get_papers_for_app()
        return jsonify({"success": True, "papers": papers})
    except PaperRetrievalError as exc:
        error_payload = dict(getattr(exc, 'details', {}) or {})
        if exc.__cause__:
            error_payload.setdefault('cause', str(exc.__cause__))
        return jsonify({"success": False, "error": error_payload}), 503
    except Exception as exc:  # pragma: no cover - catch-all for unexpected errors
        return jsonify({"success": False, "error": {"message": str(exc)}}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze papers"""
    data = request.get_json()
    papers = data.get('papers', [])
    model = data.get('model', 'gemma3')
    compare = data.get('compare', False)
    depth = data.get('depth', 'title')
    
    # Validate model selection
    valid_models = ['gemma3', 'deepseek-r1:1.5b', 'qwen2.5:7b']
    if model not in valid_models:
        model = 'gemma3'  # Default to gemma3 if invalid model is provided
    
    try:
        # Add depth parameter to analyze_with_ollama
        analysis, paper_contexts = analyze_with_ollama(papers, model, depth)
        
        # Get comparison with previous analyses if requested
        comparison_data = None
        if compare:
            comparison_data = compare_with_previous_analysis(papers, analysis)
        
        # Format paper contexts for output
        context_data = None
        if paper_contexts and depth == 'rag':
            context_data = []
            for context in paper_contexts:
                # Limit excerpt length for display
                excerpt = context['excerpt']
                if len(excerpt) > 500:
                    excerpt = excerpt[:500] + "..."
                
                context_data.append({
                    "title": context['title'],
                    "excerpt": excerpt
                })
            
        return jsonify({
            'success': True, 
            'analysis': analysis,
            'comparison': comparison_data,
            'paper_contexts': context_data,
            'model_used': model
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/papers')
def papers_page():
    """Render the papers page"""
    return render_template('papers.html', current_year=datetime.now().year)

@app.route('/analysis')
def analysis_page():
    """Render the analysis page"""
    return render_template('analysis.html', current_year=datetime.now().year)

@app.route('/paper/<paper_id>')
def paper_details(paper_id):
    """Render individual paper details"""
    try:
        papers = get_papers_for_app()
    except PaperRetrievalError as exc:
        return (
            render_template(
                'error.html',
                error_message=exc.details.get('message', 'Unable to load papers.'),
                current_year=datetime.now().year,
            ),
            503,
        )
    paper = next((p for p in papers if p['id'] == paper_id), None)
    if paper:
        return render_template('paper_details.html', paper=paper, current_year=datetime.now().year)
    return render_template('404.html', current_year=datetime.now().year), 404

# Add a new route for the modern UI template

@app.route('/modern')
def modern_papers_page():
    """Render the modern UI papers page"""
    return render_template('modern_papers.html', current_year=datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True)
