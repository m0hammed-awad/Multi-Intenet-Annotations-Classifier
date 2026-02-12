
                                              Multi-Intent Annotations Classifier
                            Customer Support Conversational AI with Hierarchical Intent Recognition
📌 Overview
<img width="1884" height="866" alt="image" src="https://github.com/user-attachments/assets/2160b092-9fad-4de9-baad-39c9e7133239" />
This project presents a Multi-Intent Annotated Customer Support Classification System designed for Conversational AI applications. 
It enables accurate identification of both fine-grained customer intents and higher-level business categories from natural language queries.

The system leverages:

MiniLM Transformer Embeddings

Deep Neural Network (DNN) Feature Selection

K-Nearest Neighbors (KNN) Classification

Flask Web Integration

Interactive Chatbot Interface

The architecture supports hierarchical dual-layer classification, making it suitable for real-world customer support automation systems.

🎯 Problem Statement

Modern customer service systems struggle with:

Multi-intent queries (e.g., “Cancel my order and issue a refund”)

Overlapping or semantically similar intents

Lack of hierarchical categorization

Poor generalization of rule-based or shallow ML systems

This project addresses these limitations through a hybrid deep-learning + classical ML pipeline.

🚀 Proposed Solution

The system introduces a two-stage classification pipeline:

Text Preprocessing

Tokenization

Stopword removal

Lemmatization

Noise cleaning

MiniLM Embeddings

Contextual semantic representation

Captures syntactic & semantic meaning

DNN-Based Feature Selection

Identifies discriminative embedding dimensions

Reduces noise and dimensionality

KNN Classification

Predicts:

🎯 Fine-grained Intent (26 intents)

🏷️ High-level Category (11 business domains)

This dual-output architecture improves reliability and scalability in chatbot systems.

🏗 System Architecture

Pipeline Flow:

User Query
→ Preprocessing
→ MiniLM Embeddings
→ DNN Feature Selection
→ KNN Classifier
→ Intent Prediction + Category Prediction
→ Chatbot Response

The system is deployed through a Flask backend, enabling real-time inference via a chatbot interface.

📂 Project Modules
1️⃣ Dataset Collection

Structured multi-intent annotated customer support dataset.

2️⃣ Exploratory Data Analysis (EDA)

Text distribution analysis

Cleaning and preprocessing

Keyword analysis

3️⃣ MiniLM Embedding Extraction

Generates contextual vector representations of queries.

4️⃣ DNN Feature Selection + KNN Classifier

Hybrid model improving classification robustness.

5️⃣ Flask Backend

Handles request routing and model inference.

6️⃣ Chatbot Interface

Interactive conversational front-end.

📊 Baseline Models

The following classifiers were evaluated:

Decision Tree

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes (NBC)

These models showed limitations in handling complex multi-intent patterns.

✅ Key Advantages

✔ Hierarchical classification (Intent + Category)

✔ Improved semantic understanding using MiniLM

✔ Noise reduction via DNN feature selection

✔ Better generalization compared to baseline models

✔ Real-time deployment with Flask integration

✔ Scalable for enterprise customer support systems

🧠 Example Predictions
Query	Intent	Category
Cancel my order and refund me	cancel_order	ORDER
I want to change my shipping address	change_shipping_address	SHIPPING
I forgot my password	recover_password	ACCOUNT
🛠 Tech Stack
Software

Python 3.12

Flask

Scikit-learn

Transformers (MiniLM)

NumPy / Pandas

Hardware

Intel i5 Processor

8GB RAM

Windows OS

⚙ Installation
git clone git@github.com:m0hammed-awad/Multi-Intenet-Annotations-Classifier.git
cd Multi-Intenet-Annotations-Classifier
pip install -r requirements.txt
python app.py

📈 Future Improvements

Replace KNN with Transformer fine-tuning

Add multi-label classification

Deploy using Docker

Integrate with cloud APIs

Improve dataset size and diversity

📄 License

This project is licensed under the MIT License.

👨‍💻 Author

Mohammed Awad
Machine Learning & Data Science Enthusiast
Focused on Conversational AI and Intelligent Systems
