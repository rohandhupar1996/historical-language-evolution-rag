# ==========================================
# FILE: src/streamlit_app/styles/custom_css.py
# ==========================================
"""Custom CSS styles for the Streamlit app."""

def get_custom_css():
    """Return custom CSS for the application."""
    return """
    <style>
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .result-card {
            background: #f8f9ff;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
            margin-bottom: 1rem;
        }
        
        .insight-card {
            background: linear-gradient(135deg, #e8f0ff 0%, #f0f8ff 100%);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
            margin-bottom: 1rem;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(42, 82, 152, 0.3);
        }
        
        .highlight {
            background-color: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .stSelectbox > div > div {
            background-color: white;
        }
        
        .stTextArea > div > div > textarea {
            background-color: white;
        }
        
        /* Hide Streamlit menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom scrollbar */
        .stMarkdown div {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .stMarkdown div::-webkit-scrollbar {
            width: 8px;
        }
        
        .stMarkdown div::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        .stMarkdown div::-webkit-scrollbar-thumb {
            background: #2a5298;
            border-radius: 10px;
        }
        
        .stMarkdown div::-webkit-scrollbar-thumb:hover {
            background: #1e3c72;
        }
    </style>
    """
