# Steam Reviews Analysis: Text Classification and RAG Implementation

This project leverages Natural Language Processing (NLP) and Machine Learning techniques to analyze Steam game reviews. The primary focus is to classify user reviews into categories such as **"recommend"** and **"not recommend"**, followed by utilizing a Retrieval-Augmented Generation (RAG) pipeline with the `microsoft/Phi-3.5-mini-instruct` model for enhanced interpretation and response generation.

---

## 📋 Project Overview

Gaming communities on platforms like Steam generate massive amounts of text data in the form of user reviews. These reviews carry significant information that can guide players and developers alike. 

In this project, we aim to:
1. Classify reviews based on sentiment and recommendation status.
2. Analyze and summarize classified reviews using a pre-trained language model with a RAG pipeline.

---

## 🎯 Objectives

1. Build a **text classification pipeline** to analyze user sentiments from game reviews.
2. Implement a **RAG-based response generation** system to provide insightful summaries and interpretations of user opinions.

---

## 🛠️ Tools & Technologies

| **Category**         | **Technology Used**                     |
|-----------------------|-----------------------------------------|
| **Programming**       | Python 3.9+                            |
| **Libraries**         | `scikit-learn`, `pandas`, `numpy`       |
|                       | `transformers`, `llama-index`            |
|                       | `nltk`,  `gradio`                        |
| **Pre-trained Models**| `microsoft/Phi-3.5-mini-instruct`       |
| **Visualization**     | `matplotlib`, `seaborn`                |
| **Dataset**           | Steam game reviews (CSV format)        |

---

## 📂 Project Structure

The repository is organized as follows:

```
project-directory/
│
├── steam_reviews.csv            # Original dataset
├── reviews_document.txt         # LLM dataset
│
├── notebooks/
│   ├── classifier.ipynb         # Text classification pipeline
│   ├── csv_to_text.ipynb        # Data formatting for RAG
│   ├── rag_LLM.ipynb            # RAG implementation and evaluation
│
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License information
```

---

## 🚀 Key Components

### 1️⃣ **Text Classification**
- Reviews are preprocessed and tokenized.
- Features are extracted using methods like TF-IDF or word embeddings.
- A supervised classification model (Logistic Regression) is trained to categorize reviews.
- The output includes labeled reviews for further analysis.

### 2️⃣ **RAG Implementation**
- **Retrieval**: Relevant data is fetched from a custom knowledge base (game review corpus).
- **Augmented Generation**: `microsoft/Phi-3.5-mini-instruct` generates context-aware summaries or responses for the classified reviews.


---

## 📊 Results & Insights

1. **Classification Accuracy**:
   - Achieved 85% accuracy in distinguishing between "recommend" and "not recommend" categories.
   - Most frequent keywords influencing recommendations: _"best"_, _"great"_, _"awesome"_.
   - Most frequent keywords influencing not recommendations: _"modding"_, _"worst"_, _"ruined"_.
   - Precision: 0.9342, Recall: 0.8552, F1-Score: 0.8930

Tabii ki! İşte düzenlenmiş ve eklediğiniz açıklamalara dayalı olarak "LLM Insights" başlığı altında ikinci maddeyi güncelledim:

1. **LLM Insights**:
- The **RAG-based LLM** successfully processed classified reviews to generate meaningful insights:
  - **Positive Reviews**: Summarized key features such as gameplay mechanics, immersive experiences, and entertainment value.
  - **Negative Reviews**: Highlighted user concerns related to performance issues, bugs, and poor optimization.
- The model demonstrated contextual awareness by:
  - Accurately summarizing reviews for games present in the dataset.
  - For games not found in the dataset:
    - Responded with an acknowledgment of insufficient data.
    - Analyzed related reviews or contextually similar discussions and generated potential insights from these observations.
    - Clearly indicated in its response whether the summary was derived from directly available data or inferred from related contexts.
- This approach ensures transparency in generated outputs and showcases the model's ability to adapt to scenarios with limited or missing information.

---
## 🔍 How to Run

### 1️⃣ Prerequisites
- Install Python 3.9 or later.
- Clone this repository:
  ```bash
  git clone https://github.com/talhasarlik/NewMind-final-project.git
  cd NewMind-final-project
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2️⃣ Running the Pipelines
1. **Data Preprocessing and Classification**:
   - Open and run `classifier.ipynb`.

2. **Data Formatting**:
   - Open and run `csv_to_text.ipynb`.
   - Prepares input for the RAG pipeline.

3. **RAG Implementation**:
   - Execute `rag_LLM.ipynb`.

4. **Evaluation**:
   - Use provided scripts in the `rag_LLM.ipynb` notebook to assess and use model.

---

## 📈 Example Use Case

Imagine a scenario where a game developer wants to analyze community feedback:
- **Input**: A large corpus of Steam reviews.
- **Output**: 
  - Sentiment-based classification of reviews.
  - Summaries highlighting common complaints and praise points.
  - Quantitative evaluation of the insights generated.

---

## 📌 Future Directions

1. **Model Fine-tuning**: Fine-tune the LLM for domain-specific nuances in gaming.
2. **Expanded Dataset**: Include reviews from other platforms like Metacritic or Reddit.
3. **Real-time Analysis**: Develop an API for real-time sentiment analysis and response generation.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a branch, and submit a pull request. You can also open an issue to suggest improvements or report bugs.

---

## 📧 Contact

For any questions or feedback, feel free to reach out:

**Talha Sarlık**  
[GitHub](https://github.com/talhasarlik) | [LinkedIn](https://linkedin.com/in/talha-sarlik/) | [Email](mailto:talhasarlik@gmail.com)

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
