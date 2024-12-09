
# Spam Email Classification Project

This project uses machine learning techniques to classify emails as **spam** or **ham** (non-spam). We use **Natural Language Processing (NLP)** to preprocess the text data and build a classification model using **Naive Bayes**.

# Objective

The main goal of this project is to classify SMS or email messages as either **spam** or **ham**. The model will be trained on a labeled dataset, and we will evaluate its performance using accuracy, precision, recall, and F1-score.

## Installation

Follow these steps to set up and run the project on your machine:

1. **Clone the repository** (if you have the project on GitHub):
    ```bash
    git clone https://github.com/Z0s-and-O1s/Spam_Mail_Classification
    ```

2. **Set up a Virtual Environment**:
    - Open your terminal or command prompt and navigate to the project directory.
    - Create a virtual environment:
      ```bash
      python -m venv venv
      ```

    - Activate the virtual environment:
      - On **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
      - On **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

3. **Install Dependencies**:
    Once your virtual environment is active, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

Here are the Python libraries used in the project:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `scikit-learn`: For machine learning algorithms and tools.
- `nltk`: For text preprocessing (e.g., stopword removal).
- `matplotlib`: For plotting and visualizing results.

You can install these dependencies by running:
```bash
pip install pandas numpy scikit-learn nltk matplotlib
```

## Dataset

The project uses the **SMS Spam Collection** dataset, which is a collection of SMS messages labeled as **ham** (non-spam) and **spam**. The dataset is in the **`data`** folder and is named **`spam.csv`**.

## How to Run

1. **Start Jupyter Notebook**:
    - Make sure your virtual environment is active.
    - Start Jupyter Notebook by running:
      ```bash
      jupyter notebook
      ```

2. **Run the Jupyter Notebook**:
    - Open **`spam_mail.ipynb`** in the Jupyter interface.
    - Run the cells in the notebook step by step to:
      - Load and preprocess the dataset.
      - Train the machine learning model.
      - Evaluate the model's performance.

## Model Evaluation

The model is evaluated using the following metrics:

- **Accuracy**: Measures the overall performance of the model.
- **Confusion Matrix**: Shows the number of correct and incorrect classifications for **ham** and **spam**.
- **Precision, Recall, F1-Score**: These metrics help evaluate how well the model performs in identifying **ham** and **spam**.

### Example of Model Output:

```text
Accuracy: 98.39%
Confusion Matrix:
[[963   2]
 [ 16 134]]

Classification Report:
              precision    recall  f1-score   support
         ham       0.98      1.00      0.99       965
        spam       0.99      0.89      0.94       150
```

## Results

- The model achieves an **accuracy of 98.39%**.
- The **precision and recall** for **ham** are very high, meaning it successfully detects most legitimate emails.
- The **precision and recall** for **spam** are also high, but there is still some room for improvement, especially in detecting all spam emails.

## Next Steps

- You can try using other text vectorization techniques, like **TF-IDF**, to improve the model's performance.
- You can also experiment with different machine learning algorithms like **Logistic Regression** or **Support Vector Machines (SVM)**.

## License

This project is open-source and available under the MIT License.