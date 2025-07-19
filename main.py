import pandas as pd

# Step 1: Load the datasets
Fake_df = pd.read_csv("Fake.csv/Fake.csv")
real_df = pd.read_csv("True.csv (1)/True.csv")

# Step 2: Add labels
Fake_df['label'] = 0   # Fake = 0
real_df['label'] = 1   # Real = 1

# Step 3: Combine both datasets
data = pd.concat([Fake_df, real_df], ignore_index=True)

# Step 4: Basic info check
print("✅ Total records:", len(data))
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Step 3.1: Features (text) and Labels (0 = fake, 1 = true)
X = data['text']
y = data['label']

# Step 3.2: Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Step 3.3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

print("✅ Step 3 Done: Text has been vectorized and split into training/testing sets.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 4.1: Model Initialization
model = MultinomialNB()

# Step 4.2: Model Training
model.fit(X_train, y_train)

# Step 4.3: Prediction
y_pred = model.predict(X_test)

# Step 4.4: Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("✅ Step 4 Done: Model trained successfully!")
print("📊 Accuracy:", round(accuracy * 100, 2), "%")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))


import joblib

# Step 5: Save the trained model
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Step 5 Done: Model and Vectorizer saved successfully!")


# Step 6: Load the model and vectorizer (in case you're testing later)
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Step 6.1: User input
print("🧪 Enter a news headline or content:")
user_input = input()

# Step 6.2: Preprocess and Predict
transformed_input = vectorizer.transform([user_input])
prediction = model.predict(transformed_input)

# Step 6.3: Output
if prediction[0] == 0:
    print("🔴 Result: FAKE News")
else:
    print("🟢 Result: REAL News")
