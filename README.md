Health Prophecy Model
Project Overview
In my country, many impoverished people still face problems accessing health facilities due to overcrowding. However, many of them have access to the digital system. Therefore, this model aims to provide them with everyday health monitoring facilities.

The Health Prophecy Model is an interactive health prediction system that offers a user-friendly web application with reliable predictions and optimized accuracy. By utilizing machine learning, it helps individuals monitor their health conditions based on the symptoms they select.

Features
User-Friendly Web Application: Easy navigation and interaction.
Reliable Predictions: Utilizes a Random Forest classifier for accurate disease prediction.
Data Preprocessing: Encodes categorical variables and handles missing values to ensure data integrity.
Revolutionizing Healthcare: Transitions from classical methodologies to cutting-edge smart health monitoring.
Installation Instructions
To run this project locally, follow the steps below:

Prerequisites
Ensure you have the following installed:

Python 3.x
pip (Python package installer)
Step-by-Step Guide
Clone the Repository

bash
Copy code
git clone https://github.com/sumankr001/Health_prophecy_using_ML.git
cd Health_prophecy_using_ML
Install Required Packages Create a virtual environment (optional but recommended) and install the required packages:

bash
Copy code
pip install -r requirements.txt
Prepare the Dataset Make sure you have the Testing.csv file in the project directory. This file should contain the necessary health data for predictions.

Run the Streamlit App Start the Streamlit application by running:

bash
Copy code
streamlit run app.py
Access the Application Open your web browser and go to http://localhost:8501 to interact with the Health Prophecy Model.

Usage Instructions
Select symptoms from the sidebar checkboxes.
Click the "check now" button to predict potential health issues.
View the predicted disease along with the classification report for model evaluation.
Conclusion
The Health Prophecy Model provides a solution to health monitoring for underserved populations, making healthcare more accessible through digital means.
