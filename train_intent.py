import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import json

# --- Configuration ---
FILE_PATH = r"C:\Users\aksha\Downloads\intentmodel\intent_dataset400.xlsx"
SAVE_PATH = r"C:\Users\aksha\Downloads\intentmodel2"
MAX_WORDS = 1000
MAX_LEN = 20
EPOCHS = 100

# 1. Load Dataset
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Could not find the Excel file at {FILE_PATH}")

df = pd.read_excel(FILE_PATH)
sentences = df['text'].astype(str).tolist()
labels = df['intent'].tolist()

# 2. Preprocessing
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build Model 
# Using explicit Input layer with batch_size=None to allow flexible inference
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN,), batch_size=None, name="input_layer"),
    tf.keras.layers.Embedding(MAX_WORDS, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train
print(f"Starting training on {len(X_train)} samples...")
history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    validation_data=(X_test, y_test), 
    verbose=1
)

# 5. Accuracy Logging
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n" + "="*40)
print(f"REPORT: Intent Classifier Training")
print(f"Total Intents: {num_classes} ({', '.join(le.classes_)})")
print(f"Final Training Accuracy: {final_train_acc:.2%}")
print(f"Final Validation Accuracy: {final_val_acc:.2%}")
print("="*40)

if final_val_acc >= 0.90:
    print("STATUS: SUCCESS - Accuracy goal (>90%) met! 🚀")
else:
    print("STATUS: NEEDS IMPROVEMENT - Accuracy is below 90%.")

# 6. Save All Assets
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Save TFLite Model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(SAVE_PATH, 'model.tflite'), 'wb') as f:
    f.write(tflite_model)

# ─── Tokenizer as JSON (recommended for Android) ───
with open(os.path.join(SAVE_PATH, 'word_index.json'), 'w', encoding='utf-8') as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=2)

# Optional: full config
with open(os.path.join(SAVE_PATH, 'tokenizer_full_config.json'), 'w', encoding='utf-8') as f:
    json.dump({
        "num_words": tokenizer.num_words,
        "oov_tpython train_intent.pyoken": tokenizer.oov_token,
        "word_index": tokenizer.word_index,
    }, f, ensure_ascii=False, indent=2)

# Save Labels (For Android interpretation)
with open(os.path.join(SAVE_PATH, 'labels.txt'), 'w') as f:
    for label in le.classes_:
        f.write(f"{label}\n")

# Save Keras H5 (Backup)
model.save(os.path.join(SAVE_PATH, 'intent_model.h5'))

print(f"\nAll files (model.tflite, tokenizer.pickle, labels.txt) saved to: {SAVE_PATH}")