import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
real_data = pd.read_csv('realAccountData.csv')
fake_data = pd.read_csv('fakeAccountData.csv')

# Combine datasets
data = pd.concat([real_data, fake_data])

# Define features and labels
X = data.drop('isFake', axis=1)
y = data['isFake']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (using RandomForest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
joblib.dump(model, 'fake_account_model.pkl')
