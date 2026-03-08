import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# --- Configuration (Must match training script paths) ---
SAVE_PATH = r"C:\Users\aksha\Downloads\intentmodel2"
MODEL_PATH    = os.path.join(SAVE_PATH, "model.tflite")
WORD_INDEX_PATH = os.path.join(SAVE_PATH, "word_index.json")      # ← new JSON file
LABEL_PATH    = os.path.join(SAVE_PATH, "labels.txt")
MAX_LEN = 20  # Must match MAX_LEN in training
OOV_TOKEN = 1  # Usually 1 in Keras Tokenizer when oov_token="<OOV>"

def predict_intent(query):
    """
    Predict intent from user query using TFLite model and JSON tokenizer.
    """
    # 1. Load word_index from JSON
    if not os.path.exists(WORD_INDEX_PATH):
        raise FileNotFoundError(f"Error: {WORD_INDEX_PATH} not found. Run training script first!")
    
    with open(WORD_INDEX_PATH, 'r', encoding='utf-8') as f:
        word_index = json.load(f)

    # 2. Load the Labels
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"Error: {LABEL_PATH} not found.")
    
    with open(LABEL_PATH, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]

    # 3. Initialize TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 4. Preprocess the query (manual tokenization + padding)
    # Convert text to lowercase and split into words (basic preprocessing)
    words = query.lower().split()  # You can improve this later (remove punctuation, etc.)

    # Convert words → token IDs (unknown words → OOV token = 1)
    sequence = []
    for word in words:
        token = word_index.get(word, OOV_TOKEN)
        sequence.append(token)

    # Pad sequence to fixed length (same as training)
    padded = pad_sequences([sequence], maxlen=MAX_LEN, padding='post', truncating='post')

    # Convert to float32 (most TFLite models expect float input for Embedding layer)
    input_data = np.array(padded, dtype=np.float32)

    # 5. Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 6. Get Result and Confidence
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data[0])
    confidence = float(output_data[0][predicted_index])  # convert to float for clean printing

    predicted_label = labels[predicted_index] if predicted_index < len(labels) else "unknown"

    return predicted_label, confidence


# --- Main Test Loop ---
if __name__ == "__main__":
    print("--- Intent Classifier Inference (using JSON tokenizer) ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_query = input("Enter Command: ")
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
            
        try:
            intent, score = predict_intent(user_query)
            
            print("-" * 50)
            print(f"QUERY      : {user_query}")
            print(f"INTENT     : {intent}")
            print(f"CONFIDENCE : {score:.4f}  ({score:.2%})")
            
            if score < 0.65:
                print("→ LOW CONFIDENCE → treat as 'unknown' / send to Gemini")
            elif intent == "unknown":
                print("→ Explicit 'unknown' intent detected")
            else:
                print("→ CONFIDENT MATCH")
                
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"ERROR: {e}")
            print("Make sure model.tflite, word_index.json and labels.txt exist in the folder.")