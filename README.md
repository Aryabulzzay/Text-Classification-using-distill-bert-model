# Text-Classification-using-distill-bert-model

A project that demonstrates the power of transformer-based models for binary text classification using **DistilBERT**, fine-tuned on a custom dataset and deployed via an interactive **Streamlit** web app.

---

## 📌 Project Overview

This project builds a lightweight yet powerful **text classification system** using the **DistilBERT** model from Hugging Face Transformers. The model classifies input text into two categories (e.g., spam vs. not spam, positive vs. negative). It is fine-tuned on a labeled dataset using PyTorch and deployed with Streamlit for real-time user interaction.

---

## 🛠️ Tech Stack / Tools Used

- 🔤 **NLP & Transformers**: Hugging Face `transformers`, `DistilBERT`
- ⚙️ **Model Training**: PyTorch, scikit-learn
- 📊 **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- 🧪 **Dataset Handling**: pandas, Excel
- 💻 **Deployment**: Streamlit (Python Web App Framework)
- 📈 **Progress Monitoring**: tqdm
- 💾 **Model Saving**: `model.save_pretrained()`

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web app
├── train_model.ipynb       # Model training notebook
├── distilbert_model/       # Saved model and tokenizer
├── output_dataset.xlsx     # Labeled dataset for training
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 How to Run the Project Locally

### 🔧 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/distilbert-text-classification.git
cd distilbert-text-classification

## 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/text-classification-distilbert.git
cd text-classification-distilbert
```

## 📦 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Step 3: Train the Model (Optional)

```bash
jupyter notebook train_model.ipynb
```

> If you don't want to train the model, you can use the pre-trained one available in the repo.

## 🖥️ Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Sample Output

- **Input Text:** `I loved the service!`  
- **Predicted Label:** `Positive (Label = 1)`

---

## 📊 Model Performance

| Metric     | Class 0 | Class 1 |
|------------|---------|---------|
| Precision  | 0.91    | 0.88    |
| Recall     | 0.89    | 0.91    |
| F1-Score   | 0.90    | 0.89    |

- **Overall Accuracy:** ~90%

---

## ✅ Features

- Fine-tuned **DistilBERT** on binary classification dataset
- Fast tokenization with `DistilBertTokenizerFast`
- **PyTorch** custom dataset class and training loop
- Interactive predictions via **Streamlit** UI
- Saved model for easy deployment and reuse

---

## 🧩 Future Enhancements

- ✅ Multi-class classification
- ✅ File upload support for batch predictions
- ✅ Backend API using **FastAPI**
- ✅ Deployment on **Hugging Face Spaces** or **Streamlit Cloud**

---

## 🧠 Skills Gained

`NLP`, `Transformer Models`, `Hugging Face`, `PyTorch`, `Text Classification`,  
`Fine-tuning`, `Model Deployment`, `Streamlit`, `Data Preprocessing`

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to **use, modify, and distribute** for personal or commercial use.
