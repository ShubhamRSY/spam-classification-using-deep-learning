# Spam Classification using deep learning

## Description
This repository demonstrates a machine learning pipeline to classify spam messages using the **Spam Classification Example**. The project includes data preprocessing, feature engineering, model building, evaluation, and prediction tasks.

---

## What We Did
1. **Data Loading and Preprocessing**:
   - Loaded the `Spam-Classification.csv` dataset.
   - Cleaned and preprocessed text data by tokenization, removing stop words, and stemming.
   - Applied feature extraction using **TF-IDF Vectorization**.

2. **Model Training**:
   - Built classification models using algorithms such as Logistic Regression, Naive Bayes, and Support Vector Machines (SVM).
   - Evaluated model performance using metrics like accuracy, precision, recall, and F1-score.

3. **Prediction**:
   - Implemented real-time spam detection for single message inputs.

4. **Visualization**:
   - Included confusion matrices and precision-recall curves to visualize model performance.

---

## Results
- **Best Model Accuracy**: 97.3%
- **Precision**: 96.8%
- **Recall**: 95.5%
- **F1-Score**: 96.1%

---

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Spam-Classification-Example.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Spam-Classification-Example
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Load and preprocess the dataset:
Convert text data into numerical features using TF-IDF:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)
```

### Train the model:
Build and train the classification model:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Evaluate the model:
Evaluate the trained model on the test dataset:
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
```

### Make predictions:
Classify a new message as spam or not spam:
```python
new_message = ["Congratulations! You won a lottery."]
new_message_transformed = vectorizer.transform(new_message)
prediction = model.predict(new_message_transformed)
print("Spam" if prediction[0] else "Not Spam")
```

---

## Project Structure
- **`code_05_XX Spam - Classification Example.html`**: Contains the detailed implementation and explanation of the pipeline.
- **`code_05_XX Spam - Classification Example.ipynb`**: Jupyter notebook with the complete code for the project.
- **`Spam-Classification.csv`**: Dataset used for training and testing the model.
- **`requirements.txt`**: List of dependencies for the project.
- **`README.md`**: This file.

---

## Dependencies
- Python 3.7 or higher
- Scikit-learn
- Pandas
- NumPy
- Matplotlib (optional for visualizations)

Install all dependencies using:
```bash
pip install -r requirements.txt
```


---

## Contributing
1. Fork the repository:
   ```bash
   git fork https://github.com/yourusername/Spam-Classification-Example.git
   ```
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature or fix bug"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-branch
   ```
5. Submit a pull request.

---

## License
This project is licensed under the **MIT License**. See `LICENSE` for details.
