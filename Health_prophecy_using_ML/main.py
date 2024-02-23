import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache_data()
def load_data():
    df = pd.read_csv("Testing.csv")
    return df

df = load_data()

# Encode categorical variables
le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])
df = pd.get_dummies(df)

# Split data into features and target variable
X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


# Streamlit app

import streamlit as st

# Sidebar with user input
st.sidebar.title("Symptoms")
symptoms = {}
for symptom in X.columns:
    symptoms[symptom] = st.sidebar.checkbox(symptom)

# Predict button
predict_button = st.sidebar.button("check now")

# Header
st.title("Health Prophecy Model")

# Main content
if predict_button:
    st.write("---")
    st.subheader("You have following Sickness")
    user_input = pd.DataFrame([symptoms])
    prediction = rf_classifier.predict(user_input)
    predicted_disease = le.inverse_transform(prediction)[0]
    st.success(predicted_disease)

    # Model evaluation
    st.subheader("Model Evaluation")
    y_pred = rf_classifier.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # st.write(f"Accuracy: {accuracy:.2f}")

    st.write("Classification Report:")
    class_report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    st.write(pd.DataFrame(class_report).T)
else:
    st.write("Select symptoms and click on 'check now' button to see the prediction.")
