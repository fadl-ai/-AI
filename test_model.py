from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_model_files(model_path):
    """Verify all required model files exist"""
    required_files = [
        'config.json',
        'model.safetensors',
        'tokenizer.json',
        'vocab.txt',
        'special_tokens_map.json',
        'tokenizer_config.json'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        elif os.path.getsize(file_path) == 0:
            missing_files.append(f"{file} (empty file)")
    
    return missing_files

def test_model():
    model_path = "models/fake_news_model"
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Error: Model not found at {model_path}")
            logger.error("Please run download_model.py first to download the model.")
            sys.exit(1)
        
        # Verify model files
        missing_files = verify_model_files(model_path)
        if missing_files:
            logger.error(f"Error: Missing or empty required files: {', '.join(missing_files)}")
            logger.error("Please run download_model.py again to download the model.")
            sys.exit(1)
            
        logger.info(f"Loading model from: {model_path}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Test texts
        test_texts = [
            "Scientists have discovered a new planet in our solar system that could potentially support life.",
            "BREAKING: Aliens have landed in New York and are giving away free iPhones! Click here to learn more!",
            "The World Health Organization has released new guidelines for COVID-19 prevention.",
            "You won't believe what this celebrity did! The secret they don't want you to know!"
        ]
        
        logger.info("\nTesting model with multiple examples:")
        logger.info("-" * 50)
        
        for text in test_texts:
            try:
                # Tokenize and get prediction
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get prediction
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()
                
                result = "FAKE" if prediction == 0 else "REAL"
                
                logger.info(f"\nText: {text}")
                logger.info(f"Prediction: {result}")
                logger.info(f"Confidence: {confidence:.2%}")
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                continue
        
        logger.info("\nModel test completed successfully!")
        
    except Exception as e:
        logger.error(f"\nError testing model: {str(e)}")
        logger.error("\nTroubleshooting steps:")
        logger.error("1. Make sure you've run download_model.py first")
        logger.error("2. Check if all model files are present in the models/fake_news_model directory")
        logger.error("3. Try running the script again")
        sys.exit(1)

if __name__ == "__main__":
    test_model() 