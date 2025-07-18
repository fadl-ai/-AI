from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import numpy as np
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
import bleach
import secrets

app = Flask(__name__)

# Security configurations
app.config['SECRET_KEY'] = secrets.token_hex(32)  # Generate a random secret key
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Basic CORS setup with specific origins
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins during development
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-Requested-With"],
        "supports_credentials": True
    }
})

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["10 per minute"]
)

# Configure Talisman for security headers
Talisman(app,
    force_https=False,
    strict_transport_security=False,  # Disabled for local development
    session_cookie_secure=False,
    content_security_policy=None,  # Disabled for local development
    feature_policy=None  # Disabled for local development
)

# Configure logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10240,
    backupCount=10,
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Fake News Detector startup')

def download_model_if_needed():
    """Download the model if it doesn't exist locally"""
    model_path = "models/fake_news_model"
    try:
        if not os.path.exists(model_path):
            app.logger.info("Model not found locally. Downloading...")
            os.makedirs('models', exist_ok=True)
            
            # Model name
            model_name = "mrm8488/bert-base-finetuned-fake-news"
            
            # Download and save the model
            app.logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            app.logger.info("Downloading model...")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Save to local directory
            app.logger.info("Saving model and tokenizer...")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            app.logger.info(f"Model downloaded and saved to: {model_path}")
        else:
            app.logger.info("Model found locally.")
    except Exception as e:
        app.logger.error(f"Error downloading model: {str(e)}")
        raise

# Download model if needed
try:
    download_model_if_needed()
except Exception as e:
    app.logger.error(f"Failed to download model: {str(e)}")
    raise

# Load the model from local directory
try:
    app.logger.info("Loading model and tokenizer from local directory...")
    model_path = "models/fake_news_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    app.logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    raise

# Enhanced fake news indicators with weighted scores and categories
FAKE_INDICATORS = {
    'strong_indicators': {
        'words': [
            'miracle cure', 'secret they don\'t want you to know',
            'doctors hate this', 'never before seen',
            'shocking discovery', 'mind-blowing results',
            'conspiracy', 'they\'re hiding', 'government cover-up',
            'mainstream media won\'t tell you', 'big pharma',
            'they don\'t want you to know', 'shocking truth'
        ],
        'weight': 0.25
    },
    'moderate_indicators': {
        'words': [
            'breaking', 'urgent', 'exclusive', 'shocking',
            'unbelievable', 'amazing', 'incredible', 'must see',
            'you won\'t believe', 'mind-blowing', 'viral',
            'trending', 'sensational', 'controversial'
        ],
        'weight': 0.15
    },
    'weak_indicators': {
        'words': [
            'new study', 'research shows', 'experts say',
            'according to', 'sources say', 'studies suggest',
            'scientists claim', 'research indicates', 'experts believe',
            'according to experts', 'recent study'
        ],
        'weight': 0.05
    }
}

# Credibility indicators with weighted scores
CREDIBILITY_INDICATORS = {
    'strong_indicators': {
        'phrases': [
            'peer-reviewed study', 'published in', 'verified by',
            'confirmed by official sources', 'according to official sources',
            'reported by', 'research conducted by', 'study published in',
            'according to the report', 'based on data from',
            'according to official statistics', 'verified by experts'
        ],
        'weight': 0.2
    },
    'moderate_indicators': {
        'phrases': [
            'according to', 'reported by', 'cited by',
            'referenced in', 'documented in', 'recorded in',
            'verified source', 'reliable source', 'trusted source',
            'official statement', 'official report'
        ],
        'weight': 0.1
    }
}

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def analyze_text_style(text):
    # Analyze text style and patterns
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Calculate various metrics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(5)
    
    # Analyze punctuation
    exclamation_count = len(re.findall(r'!', text))
    question_count = len(re.findall(r'\?', text))
    caps_words = len(re.findall(r'\b[A-Z]{4,}\b', text))
    
    # Calculate style score
    style_score = 0
    
    # Penalize excessive punctuation
    if exclamation_count > 2:
        style_score -= 0.15 * min(exclamation_count / 5, 1)
    if question_count > 2:
        style_score -= 0.1 * min(question_count / 5, 1)
    
    # Penalize excessive caps
    if caps_words > 2:
        style_score -= 0.15 * min(caps_words / 5, 1)
    
    # Penalize very short or very long sentences
    if avg_sentence_length < 5:
        style_score -= 0.1
    elif avg_sentence_length > 30:
        style_score -= 0.05
    
    return {
        'style_score': max(-1, min(1, style_score)),
        'metrics': {
            'avg_sentence_length': avg_sentence_length,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_words': caps_words,
            'most_common_words': most_common_words
        }
    }

def calculate_credibility_score(text):
    text_lower = text.lower()
    score = 0
    
    # Check for fake news indicators with reduced weights
    for category in FAKE_INDICATORS.values():
        for word in category['words']:
            if word in text_lower:
                score -= category['weight'] * 0.8  # Reduce impact of fake indicators
    
    # Check for credibility indicators with increased weights
    for category in CREDIBILITY_INDICATORS.values():
        for phrase in category['phrases']:
            if phrase in text_lower:
                score += category['weight'] * 1.2  # Increase impact of credibility indicators
    
    # Analyze text style with reduced impact
    style_analysis = analyze_text_style(text)
    score += style_analysis['style_score'] * 0.15  # Reduced from 0.2 to 0.15
    
    return {
        'score': max(-1, min(1, score)),
        'style_analysis': style_analysis
    }

# Enhanced input validation
def validate_input(text):
    if not text or not isinstance(text, str):
        return False, "Invalid input: Text must be a non-empty string"
    
    # Check for maximum length
    if len(text) > 10000:  # 10k characters limit
        return False, "Input too long: Maximum 10,000 characters allowed"
    
    # Check for minimum words
    words = text.split()
    if len(words) < 50:
        return False, "Input too short: Minimum 50 words required"
    
    # Check for malicious content
    if re.search(r'<script|javascript:|data:', text, re.IGNORECASE):
        return False, "Input contains potentially malicious content"
    
    # Sanitize input
    sanitized_text = bleach.clean(
        text,
        strip=True,
        tags=[],
        attributes={},
        protocols=[],
        strip_comments=True
    )
    
    if sanitized_text != text:
        return False, "Input contains potentially harmful content"
    
    return True, sanitized_text

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_news():
    try:
        # Validate request headers
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        # Validate and sanitize input
        is_valid, result = validate_input(data['text'])
        if not is_valid:
            app.logger.warning(f"Invalid input detected: {result}")
            return jsonify({'error': result}), 400
        
        text = result
        
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        try:
            app.logger.info("Starting model prediction...")
            # Get ML model prediction with confidence
            inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512)
            app.logger.info("Text tokenized successfully.")
            
            outputs = model(**inputs)
            app.logger.info("Model prediction completed.")
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # The model outputs [fake_prob, real_prob]
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
            
            # Log raw probabilities for debugging
            app.logger.info(f"Raw probabilities - Fake: {fake_prob:.3f}, Real: {real_prob:.3f}")
            
            # Calculate confidence based on the higher probability
            ml_confidence = max(fake_prob, real_prob)
            is_fake_ml = fake_prob > real_prob
            
            # Calculate uncertainty based on how close the probabilities are
            uncertainty = 1 - abs(fake_prob - real_prob)
            
        except Exception as model_error:
            app.logger.error(f"Model error: {str(model_error)}")
            app.logger.error(f"Error type: {type(model_error)}")
            app.logger.error(f"Error details: {str(model_error.__dict__)}")
            return jsonify({'error': f'Model processing error: {str(model_error)}'}), 500
        
        # Calculate credibility score and get analysis
        credibility_result = calculate_credibility_score(text)
        credibility_score = credibility_result['score']
        style_analysis = credibility_result['style_analysis']
        
        # Calculate final score with optimized weights
        text_length = len(text.split())
        text_length_factor = min(text_length / 200, 1)
        
        # Optimize ML weight calculation
        base_ml_weight = 0.6  # Increased from 0.8
        ml_weight = base_ml_weight + (0.15 * text_length_factor)  # Reduced text length impact
        ml_weight *= (1 - uncertainty * 0.15)  # Further reduced uncertainty impact
        credibility_weight = 1 - ml_weight
        
        # Calculate final score with adjusted formula
        final_score = (ml_confidence * ml_weight) + ((1 + credibility_score) / 2 * credibility_weight)
        
        # Determine if the news is fake based on both ML and credibility scores
        is_fake = is_fake_ml or final_score < 0.5  # Consider both ML prediction and final score
        
        # Log analysis results with more details
        app.logger.info(f"Final Score: {final_score}, ML: {is_fake_ml}, Credibility Score: {credibility_score}")

        app.logger.info(f"Analysis completed - Score: {final_score:.2f}, Is Fake: {is_fake}, ML Confidence: {ml_confidence:.2f}, Credibility Score: {credibility_score:.2f}")
        
        analysis = {
            'is_fake': is_fake,
            'confidence': ml_confidence if is_fake_ml else (1 - ml_confidence),  # Confidence in the prediction
            'ml_confidence': ml_confidence,  # Raw ML confidence
            'real_confidence': real_prob,
            'fake_confidence': fake_prob,
            'credibility_score': credibility_score,
            'uncertainty': uncertainty,
            'details': {
                'has_strong_indicators': any(word in text.lower() for word in FAKE_INDICATORS['strong_indicators']['words']),
                'has_moderate_indicators': any(word in text.lower() for word in FAKE_INDICATORS['moderate_indicators']['words']),
                'has_weak_indicators': any(word in text.lower() for word in FAKE_INDICATORS['weak_indicators']['words']),
                'style_analysis': style_analysis['metrics'],
                'text_length': text_length,
                'analysis_weights': {
                    'ml_weight': ml_weight,
                    'credibility_weight': credibility_weight
                }
            }
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        app.logger.error(f"General error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def ratelimit_handler(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429

if __name__ == '__main__':
    try:
        print("Starting server...")
        print("Server will be available at: http://127.0.0.1:5000")
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise
