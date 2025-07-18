from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Model name
model_name = "mrm8488/bert-base-finetuned-fake-news"
save_path = "models/fake_news_model"

print(f"Downloading model: {model_name}")
print("This might take a few minutes...")

# Download and save the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save to local directory
print(f"Saving model to: {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Model downloaded and saved successfully!")
print(f"Model location: {os.path.abspath(save_path)}")

# Test the model
print("\nTesting the model...")
test_text = "Scientists have discovered a new planet in our solar system that could potentially support life."

# Get prediction
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
prediction = torch.argmax(probabilities, dim=-1).item()
confidence = probabilities[0][prediction].item()

result = "FAKE" if prediction == 0 else "REAL"

print("\nTest Results:")
print(f"Text: {test_text}")
print(f"Prediction: {result}")
print(f"Confidence: {confidence:.2%}") 