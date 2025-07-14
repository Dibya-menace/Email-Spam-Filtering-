# Email-Spam-Filtering-
using Logistic Regression and Naive Bayes 
📧 Email Spam Filter using Machine Learning

This project implements a Spam Detection System using machine learning techniques on SMS/email text messages. It uses TF-IDF vectorization and two classifiers: Logistic Regression and Naive Bayes.

🔍 Problem Statement

Email spam detection is a classic text classification problem where the goal is to identify whether a message is spam or not spam (ham). We use supervised learning on labeled SMS/email messages.

📂 Dataset

SMS Spam Collection Dataset from UCI Machine Learning Repository

The dataset contains ~5,500 labeled messages.

Label: spam or ham

Text: Message content

📚 Libraries Used

pandas
numpy
scikit-learn

🧠 Model Workflow

Data Preprocessing

Loaded dataset using pandas

Replaced labels: spam → 0, ham → 1

Split into training and test sets using train_test_split

Text Vectorization

Used TfidfVectorizer to convert text into numerical features

Model Training

Logistic Regression

Multinomial Naive Bayes

Evaluation Metrics

Accuracy Score

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

🥪 Model Evaluation

Model

Accuracy

Logistic Regression

✅ High (~97%+)

Naive Bayes

⚠️ Slightly Lower (depends on vectorizer)

Naive Bayes may perform better with CountVectorizer than TF-IDF.

📟 Sample Code to Predict New Messages

def predict_message(message):
    input_features = vectorizer.transform([message])
    prediction = model.predict(input_features)
    
    return "Spam ❌" if prediction[0] == 0 else "Not Spam ✅"

# Example
predict_message("Congratulations! You've won a free ticket.")

🛠️ How to Run

Clone the repository or run the notebook on Colab/Jupyter.

Make sure scikit-learn, pandas, and numpy are installed.

Run all cells to train the models and test predictions.


🧑‍💻 Author

Dibyajyoti Dutta
Project for learning ML with text data (Spam/Ham filtering)
