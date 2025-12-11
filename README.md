# 🏥 Healthcare Fraud Detection System

A complete end-to-end machine learning solution for detecting healthcare fraud using Databricks, Delta Lake, and RAG-powered analytics.

## 📋 Overview

This project implements an intelligent fraud detection system that processes healthcare claims data through a medallion architecture (Bronze → Silver → Gold) and provides a RAG-powered chatbot interface for natural language queries.

### Key Features

- **Synthetic Data Generation**: Creates realistic healthcare claims with injected fraud patterns
- **Multi-Layer Architecture**: Bronze, Silver, and Gold Delta Lake tables
- **ML Fraud Detection**: Random Forest + Isolation Forest models
- **RAG Chatbot**: Natural language interface using sentence transformers and FAISS
- **Visual Analytics**: Comprehensive fraud detection dashboard

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Generation                         │
│  • 1,000 Patients  • 50 Providers  • 5,000 Claims           │
│  • 8% Fraud Rate   • 5 Fraud Patterns                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Bronze Layer (Raw Data)                    │
│  • patients  • providers  • claims_bronze                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Silver Layer (Enrichment + Features)           │
│  • Feature Engineering  • Statistical Aggregations          │
│  • Provider & Patient Analytics                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             Gold Layer (ML Predictions + RAG)               │
│  • Random Forest Classifier  • Isolation Forest             │
│  • FAISS Vector Search  • Natural Language Queries          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Databricks workspace
- Python 3.8+
- Access to Delta Lake

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/healthcare-fraud-detection.git
cd healthcare-fraud-detection
```

2. **Upload notebooks to Databricks**
   - Import all `.py` files into your Databricks workspace
   - Run notebooks in sequence (01 → 02 → 03)

3. **Run the pipeline**
```bash
# Step 1: Generate synthetic fraud data
# Run: 01_Generate_Fraud_Data.py

# Step 2: ETL and ML model training
# Run: 02_ETL_Fraud_Detection.py

# Step 3: RAG chatbot interface
# Run: 03_RAG_Chatbot_Interface.py
```

## 📊 Fraud Patterns Detected

The system identifies five types of healthcare fraud:

| Fraud Type | Description | Detection Method |
|------------|-------------|------------------|
| **Duplicate Billing** | Same procedure billed multiple times | Pattern matching |
| **Abnormal Amount** | Claims 3-5x normal price range | Statistical analysis |
| **Procedure Mismatch** | Provider performing out-of-specialty procedures | Domain validation |
| **Upcoding** | Billing for more expensive procedures | Price analysis |
| **Unbundling** | Separating bundled procedures | Aggregation analysis |

## 🔍 RAG Chatbot Examples

The system supports natural language queries:

```python
# Example queries
"Show me suspicious cardiology claims with high amounts"
"Find duplicate billing fraud cases"
"What are high-value anomalies over $30,000?"
"Show me procedure mismatch fraud in oncology"
```

## 📈 Model Performance

- **Random Forest Classifier**: Trained on 11 engineered features
- **Isolation Forest**: Unsupervised anomaly detection
- **Detection Rate**: ~8% of total claims flagged
- **Features**: Billed amount, provider statistics, temporal patterns, specialty encoding

## 🗂️ Project Structure

```
healthcare-fraud-detection/
│
├── 01_Generate_Fraud_Data.py        # Synthetic data generation
├── 02_ETL_Fraud_Detection.py        # ETL pipeline + ML models
├── 03_RAG_Chatbot_Interface.py      # RAG chatbot interface
├── Healthcare_Fraud_Detection.pptx   # Project presentation
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore file
```

## 🛠️ Technologies Used

- **Data Processing**: PySpark, Pandas, NumPy
- **Storage**: Delta Lake (Bronze/Silver/Gold layers)
- **ML Models**: scikit-learn (Random Forest, Isolation Forest)
- **NLP/RAG**: sentence-transformers, FAISS
- **Visualization**: Matplotlib, Seaborn
- **Synthetic Data**: Faker library

## 📊 Delta Lake Tables

| Table | Description | Records |
|-------|-------------|---------|
| `fraud_detection.patients` | Patient demographics | 1,000 |
| `fraud_detection.providers` | Healthcare providers | 50 |
| `fraud_detection.claims_bronze` | Raw claims data | 5,000 |
| `fraud_detection.claims_silver` | Enriched with features | 5,000 |
| `fraud_detection.claims_gold` | ML predictions + fraud scores | 5,000 |

## 🎯 Future Enhancements

- [ ] Real-time streaming fraud detection
- [ ] Deep learning models (LSTMs for temporal patterns)
- [ ] Graph neural networks for provider networks
- [ ] Integration with production healthcare systems
- [ ] Advanced explainability (SHAP/LIME)
- [ ] Web-based dashboard interface

## 👤 Author

**Your Name**
- GitHub: [@atulkumar](https://github.com/Atul-0515/)

## 🙏 Acknowledgments

- Built with Databricks platform
- Uses Delta Lake for reliable data storage
- Powered by scikit-learn and sentence-transformers

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐