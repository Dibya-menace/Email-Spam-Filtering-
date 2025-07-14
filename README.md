📧 Email Spam Filter using Machine Learning
🔍 Problem Statement
Email spam detection is a classic text classification problem where the goal is to identify whether a message is spam or not spam (ham). We use supervised learning on labeled SMS/email messages.
💾 Dataset
- SMS Spam Collection Dataset from UCI Machine Learning Repository
- The dataset contains ~5,500 labeled messages.
  - Label: spam or ham
  - Text: Message content
📚 Libraries Used
- pandas
- numpy
- scikit-learn
🧠 Model Workflow
1. Data Preprocessing:
   - Loaded dataset using pandas
   - Replaced labels: spam → 0, ham → 1
   - Split into training and test sets using train_test_split

2. Text Vectorization:
   - Used TfidfVectorizer to convert text into numerical features

3. Model Training:
   - Logistic Regression
   - Multinomial Naive Bayes

4. Evaluation Metrics:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix
🧪 Model Evaluation
| Model               | Accuracy       |
|---------------------|----------------|
| Logistic Regression | ✅ High (~97%) |
| Naive Bayes         | ⚠️ Lower       |

Note: Naive Bayes may work better with CountVectorizer instead of TfidfVectorizer.
🧾 Sample Code to Predict New Messages
```python
def predict_message(message):
    input_features = vectorizer.transform([message])
    prediction = model.predict(input_features)
    return "Spam ❌" if prediction[0] == 0 else "Not Spam ✅"

# Example
predict_message("Congratulations! You've won a free ticket.")
```
🛠️ How to Run
1. Clone the repository or run the notebook on Google Colab/Jupyter Notebook.
2. Make sure the following libraries are installed:
   - scikit-learn
   - pandas
   - numpy
3. Run all cells to train the models and test predictions.
🔮 Future Improvements
- Use Deep Learning (LSTM/BERT) for better accuracy
- Train on larger, real-world email datasets
- Build a web interface using Flask or Streamlit
🧑‍💻 Author
Dibyajyoti Dutta
Project to practice Machine Learning on text data (Spam/Ham filtering)
📄 License
MIT License — feel free to use and improve!
