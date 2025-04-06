# 💳 Credit Card Fraud Detection App

A simple and interactive web application to detect fraudulent credit card transactions using a machine learning model trained on the popular Kaggle dataset.

---

### 🧠 About the Project  
Credit card fraud detection is a critical task in the financial industry due to the enormous volume of daily transactions. This app uses anonymized PCA features (`V1` to `V28`) and a normalized transaction amount to predict whether a given transaction is **fraudulent** or **genuine**.  
The model was trained using `RandomForestClassifier` and balanced using **SMOTE** to handle class imbalance.

---

### 🧪 Sample Inputs  
Try these example values (as seen in the test dataset) to test the model:

#### ❌ Fraudulent Transaction  
- `V1`: -2.312  
- `V2`: 1.003  
- …  
- `V28`: 0.354  
- `NormalizedAmount`: 0.87

#### ✅ Genuine Transaction  
- `V1`: 1.092  
- `V2`: -0.813  
- …  
- `V28`: 0.029  
- `NormalizedAmount`: 0.45

> Note: These are PCA-transformed values from the original dataset.

---

### 💡 How to Use the App  
1. Launch the app using `streamlit run app.py`.  
2. Manually enter feature values (`V1` to `V28` + `NormalizedAmount`) OR choose a row from the sample test data.  
3. Click **Predict** to get a classification:
   - 🏷️ Result: **Fraud** or **Genuine**
   - 🔍 Confidence score of prediction
4. View and manage the 🧾 prediction history for reference.

---

### 📊 Test Dataset Preview  
The app includes a preview of sample data (`test_data.csv`) with real transactions.  
You can scroll through it and select a row to auto-fill the prediction input.

---

### 🙌 Acknowledgements  
- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Frontend**: [Streamlit]  
- **Libraries**: scikit-learn, pandas, numpy, joblib

---

