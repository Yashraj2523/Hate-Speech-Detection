import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

# File paths
hate_speech_file = 'tadata.csv'
transcript_file = 'tatrans.txt'

# Step 1: Load the Data
texts = []
labels = []
with open(hate_speech_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for row in reader:
        if len(row) >= 2:  # Check if the row has both text and label
            texts.append(row[0])
            labels.append(row[1])

# Step 3: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Step 4: Train the Model
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))  # Use linear kernel
model.fit(texts, y)  

# Step 6: Load the transcript data
with open(transcript_file, 'r', encoding='utf-8') as file:
    transcript_text = file.read()

# Step 7: Predict using the Model
y_pred = model.predict([transcript_text])  # Predict for the transcript text

# Step 8: Display result
if len(y_pred) > 0:
    if y_pred[0] == 'Hate-Speech':
        print("Hate speech detected.")
    else:
        print("No hate speech detected")
else:
    print("No text found in the transcript.")

# Step 9: Save the Model
joblib.dump(model, 'hate_speech_detection_model.pkl')
