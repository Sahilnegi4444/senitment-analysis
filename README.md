# Amazon Review Sentiment Analysis (End-to-End ML Project)

This project is an end-to-end Sentiment Analysis System built using Amazon Review data, which classifies user reviews as Positive or Negative.
It covers the entire machine learning lifecycle — from data ingestion using SQL, data preprocessing, feature engineering, model training, to deployment using Streamlit.

## Project Overview

The goal of this project is to build a production-ready sentiment analysis pipeline that can automatically analyze Amazon product reviews and determine whether the sentiment is positive or negative.

The model was designed to focus more on recall, ensuring that no negative reviews are missed, which is critical for understanding customer dissatisfaction.

⚙️ Project Workflow
1. Data Ingestion

Connected to a MySQL database using SQLAlchemy to fetch Amazon Review data.

This ensured scalable and reliable data access directly from a structured source.

Reason: Using SQL ingestion allows data to be easily queried, updated, and managed efficiently in production pipelines.

2. Handling Imbalanced Data

Applied the SMOTE (Synthetic Minority Over-sampling Technique) method to balance the dataset.

SMOTE generates synthetic examples of the minority class (negative reviews) to ensure the model learns both classes equally well.

Reason: Imbalanced data can bias the model toward the majority class; SMOTE helps improve performance on underrepresented labels.

3. Text Vectorization using TF-IDF

Transformed raw review text into numerical form using TF-IDF Vectorizer (Term Frequency–Inverse Document Frequency).

This technique gives higher weight to important words while reducing the influence of common words like “the”, “is”, etc.

Reason: TF-IDF captures the importance of words in a review, providing a more informative representation for sentiment modeling.

4. Dimensionality Reduction using Truncated SVD

Applied Truncated SVD (Singular Value Decomposition) on the TF-IDF matrix to reduce its high dimensionality.

Reason: TF-IDF creates thousands of features. Truncated SVD helps:

Reduce computational cost,

Prevent overfitting,

Improve generalization and model speed,

Capture the most informative components of text data.

5. Model Training and Evaluation

Trained the model on the processed dataset using a suitable classifier, Logistic Regression.

Focused on optimising **recall** during training and testing to minimise the chance of missing any negative reviews.

Reason: In real-world feedback analysis, missing negative reviews is riskier than misclassifying positives — recall ensures that all dissatisfied customer signals are captured.

6. Deployment with Streamlit

Built an interactive Streamlit web application where users can:

Input a product review,

Get instant sentiment prediction,

See prediction probabilities for both positive and negative sentiments.

Reason: Streamlit provides a lightweight and fast deployment framework for turning ML models into usable web apps without needing frontend development.

🧩 Tech Stack
Category	Tools / Libraries
Language	Python
Data Storage	MySQL
Data Handling	Pandas, SQLAlchemy
Modeling	Scikit-learn, Logistic Regression
Balancing Technique	SMOTE (Imbalanced-learn)
Feature Engineering	TF-IDF Vectorizer, Truncated SVD
Deployment	Streamlit
Serialization	Pickle, Joblib

## Project Structure
```
sentiment_app/
├── app.py                   # Streamlit app
├── model.joblib             # Trained model
├── tf_idf.pkl               # TF-IDF vectorizer
├── dim_reduction.pkl        # Truncated SVD object
├── requirements.txt         # Dependencies
├── .env                     # Database credentials (excluded from repo)
└── README.md                # Project documentation
```
## Example Output

User Input: “The product stopped working within a week. Totally disappointed.”
Sentiment: Negative
Prediction probability: 98.81%

{
"Negative": "98.81%"
"Positive": "1.19%"
}

## How to Run Locally

Clone this repository
git clone https://github.com/<your-username>/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis


## Install dependencies
pip install -r requirements.txt


## Run the Streamlit app
streamlit run app.py

## Author

**Sahil Negi**
Data Science & AI Enthusiast | Machine Learning Engineer
📧 negisahil4444@gmail.com
🔗 www.linkedin.com/in/sahil-negi4444
