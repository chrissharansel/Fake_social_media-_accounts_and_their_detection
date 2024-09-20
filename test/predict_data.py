import http.client
import json
import joblib
import numpy as np

# Function to fetch user data from Twitter API using RapidAPI
def fetch_twitter_data(username):
    conn = http.client.HTTPSConnection("twitter-api45.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "1371697bd4msh353e9c911503646p19f730jsndc6f124ea83b",
        'x-rapidapi-host': "twitter-api45.p.rapidapi.com"
    }
    
    conn.request("GET", f"/profile.php?username={username}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    account_info = json.loads(data.decode("utf-8"))
    
    return {
        'userFollowerCount': account_info.get('followers_count', 0),
        'userFollowingCount': account_info.get('following_count', 0),
        'userBiographyLength': len(account_info.get('description', '')),
        'userMediaCount': account_info.get('media_count', 0),
        'userHasProfilPic': 1 if account_info.get('profile_image_url') else 0,
        'userIsPrivate': 1 if account_info.get('protected', False) else 0,
        'usernameDigitCount': sum(char.isdigit() for char in username),
        'usernameLength': len(username)
    }

# Load the trained model
model = joblib.load('fake_account_model.pkl')

# Function to make a prediction
def predict_fake_account(username):
    # Fetch Twitter data
    user_data = fetch_twitter_data(username)
    
    # Prepare data for prediction
    input_data = np.array([[
        user_data['userFollowerCount'],
        user_data['userFollowingCount'],
        user_data['userBiographyLength'],
        user_data['userMediaCount'],
        user_data['userHasProfilPic'],
        user_data['userIsPrivate'],
        user_data['usernameDigitCount'],
        user_data['usernameLength']
    ]])
    
    # Predict
    prediction = model.predict(input_data)
    return "Fake" if prediction[0] == 1 else "Real"

# Example usage
if __name__ == "__main__":
    username = input("Enter Twitter username: ")
    result = predict_fake_account(username)
    print(f"The account {username} is predicted to be: {result}")
