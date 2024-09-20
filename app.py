from flask import Flask, render_template, request, redirect, url_for
url_for
from flask_sqlalchemy import SQLAlchemy
import instaloader
import numpy as np
import tensorflow as tf
import os
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from werkzeug.utils import secure_filename
from joblib import load
import instaloader
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    is_fake = db.Column(db.Boolean, nullable=False)
    test_count = db.Column(db.Integer, nullable=False, default=1)

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("Error loading audio:", e)
        return None

model_path = 'trained_model.joblib'
if os.path.exists(model_path):
    clf = load(model_path)
else:
    print(f"Model file '{model_path}' not found.")

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/voice', methods=['GET', 'POST'])
def voice():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('voice.html', prediction_text='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('voice.html', prediction_text='No selected file')
        if file:
            # Create the 'uploads' directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            features = extract_features(file_path)
            if features is not None:
                predicted_label = clf.predict([features])[0]
                return render_template('voice.html', prediction_text=f'Predicted label: {predicted_label}')
            else:
                return render_template('voice.html', prediction_text='Error extracting features from audio')
    return render_template('voice.html')

'''
@app.route('/voice', methods=['GET', 'POST'])
def voice():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('voice.html', prediction_text='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('voice.html', prediction_text='No selected file')
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            features = extract_features(file_path)
            if features is not None:
                predicted_label = clf.predict([features])[0]
                return render_template('voice.html', prediction_text=f'Predicted label: {predicted_label}')
            else:
                return render_template('voice.html', prediction_text='Error extracting features from audio')
    return render_template('voice.html')
'''
def get_instagram_data(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        return {
            "userFollowerCount": profile.followers,
            "userFollowingCount": profile.followees,
            "userBiographyLength": len(profile.biography),
            "userMediaCount": profile.mediacount,
            "userHasProfilPic": int(not profile.is_private and profile.profile_pic_url is not None),
            "userIsPrivate": int(profile.is_private),
            "usernameDigitCount": sum(c.isdigit() for c in profile.username),
            "usernameLength": len(profile.username),
        }
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile with username '{username}' not found.")
        return None

# Load the trained model
load_model = tf.keras.models.load_model('trainedmodel.h5')


@app.route('/about')
def about():
    # Render the about.html template
    return render_template('about.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html')

@app.route('/top_users')
def top_users():
    # Query the database to get the top 10 users with the lowest confidence
    top_users = PredictionResult.query.order_by(PredictionResult.confidence).limit(10).all()
    return render_template('top_users.html', top_users=top_users)


@app.route('/result')
def result():
    # Get parameters from the query string or use default values
    username = request.args.get('username', 'N/A')
    confidence = float(request.args.get('confidence', 'N/A'))  # Ensure confidence is a float
    
    # Display the result in the result.html template
    return render_template('result.html', username=username, confidence=confidence)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the username from the form submission
        username = request.form['username']
        reasons = request.form.getlist('reasons')
        others = request.form.get('others')

        # Get Instagram data
        insta_data = get_instagram_data(username)

        if insta_data:
            # Convert Instagram data to NumPy array
            X_new = np.array([list(insta_data.values())], dtype=np.float32)

            # Make predictions
            predictions = load_model.predict(X_new)

            # Determine the result
            confidence_percentage = round(((1 - predictions[0][0]) * 100),3)  # Subtract probability from 1 and multiply by 100
            is_fake = predictions[0][0] >= 0.5

            # Save the result to the database if it's fake
            if is_fake:
                # Check if the username already exists in the database
                existing_result = PredictionResult.query.filter_by(username=username).first()
                if existing_result:
                    existing_result.test_count += 1  # Increment test count if username exists
                else:
                    result_entry = PredictionResult(username=username, confidence=confidence_percentage, is_fake=is_fake)
                    db.session.add(result_entry)
                db.session.commit()

            # Redirect to the result page with the result as parameters
            return redirect(url_for('result', username=username, confidence=confidence_percentage))
        else:
            return render_template('result.html', username='N/A', confidence='N/A')
    except Exception as e:
        return render_template('result.html', username='N/A', confidence='N/A')




L = instaloader.Instaloader()
L.login("rusharavichandran", "Shinee2008*")


# Initialize NLTK's VADER sentiment analyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Function to analyze comments sentiment
def analyze_sentiment(comment):
    ss = sid.polarity_scores(comment)
    return ss['compound']

# Function to count hate comments
def count_hate_comments(profile):
    hate_comments = []
    hate_comment_count = 0
    try:
        comment_c=0
        print("Analyzing posts and comments for user:", profile.username)
        for post in profile.get_posts():
            comments = post.get_comments()
            for comment in comments:
                comment_c=comment_c+1
                sentiment_score = analyze_sentiment(comment.text)
                if sentiment_score < -0.5:  # Adjust the threshold as needed
                    print(f"Hate comment found on post {post.shortcode}: {comment.text}")
                    hate_comments.append(comment.text)
                    hate_comment_count += 1
                    pr=round(((hate_comment_count/comment_c)*100),2)
    except Exception as e:
        print(f"Error occurred while processing user {profile.username}: {e}")
    finally:
        print(f"Total hate comments found for user {profile.username}: {hate_comment_count}")
        print(f"Pecentage of hate comments ",{pr})
        return hate_comments, hate_comment_count,pr

# Function to analyze a user's posts and comments for hate comments
def analyze_user(username):
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        hate_comments, hate_comment_count, pr = count_hate_comments(profile)
        return hate_comments, hate_comment_count, pr
    except Exception as e:
        print(f"Error occurred while analyzing user {username}: {e}")
        return None, 0, 0  # Return default values if an error occurs


@app.route('/hate')
def hate():
    return render_template('hate.html')

@app.route('/analyze_user_post', methods=['POST'])
def analyze_user_post():
    username = request.form['username']
    hate_comments, hate_comment_count, pr = analyze_user(username)
    return render_template('hate.html', hate_comments=hate_comments, hate_comment_count=hate_comment_count, pr=pr)




if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='127.0.0.1', port=5004)
