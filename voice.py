import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import audioread
import librosa

# Function to extract features from audio file
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    except audioread.NoBackendError:
        raise Exception("No audio backend available. Please install ffmpeg or use WAV files.")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    return mfccs_processed

# Path to the folder containing audio files
base_path = r'C:\Users\Dell\Desktop\CSE109\VOICE CLASSIFICATION\voices'

# List of labels
labels = ['convertedvoice', 'humanvoice', 'robotvoice']

# Initialize lists to store features and labels
X = []
y = []

# Iterate over each label
for label in labels:
    # Path to the folder for the current label
    folder_path = os.path.join(base_path, label)
    # Iterate over each audio file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an audio file
        if filename.endswith(('.wav', '.mp3')):
            # Full path to the audio file
            file_path = os.path.join(folder_path, filename)
            # Extract features and append to X
            features = extract_features(file_path)
            X.append(features)
            # Append label to y
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
dump(clf, 'trained_model.joblib')
