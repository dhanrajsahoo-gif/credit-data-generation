# Credit Data Generation — Barclays BNPL PoC

A end-to-end pipeline for synthetic credit data generation, unstructured document creation, and LLM-based extraction — built as a Proof of Concept for a Buy Now Pay Later (BNPL) credit risk assessment use case.

---

## Overview

This project demonstrates how to augment a real-world credit dataset (Home Credit Default Risk) with synthetic tabular data and AI-generated unstructured documents (paystubs, bank statements, LinkedIn profiles, ID documents, property docs, credit reports), then use a multimodal LLM to extract structured fields back from those documents for model comparison.

The pipeline is broken into 4 phases:

| Phase | Notebook | Description |
|-------|----------|-------------|
| 1 | `01_baseline_model_executed.ipynb` | Train a LightGBM baseline credit risk model (Model A) |
| 2 & 3 | `02_synthetic_data_generation.ipynb` | Identify fields, generate synthetic applicants, and create unstructured documents via Gemini |
| 4 | `03_extraction_and_validation_executed.ipynb` | Extract structured fields from documents using LLM and compare against ground truth |

---

## Project Structure

```
Credit-Data-Generation/
├── notebooks/
│   ├── 01_baseline_model_executed.ipynb         # Phase 1: LightGBM baseline (Model A)
│   ├── 02_synthetic_data_generation.ipynb       # Phase 2 & 3: Synthetic data + doc generation
│   ├── 03_extraction_and_validation_executed.ipynb  # Phase 4: LLM extraction & validation
│   └── kaggle notebook/                         # Original Kaggle intro notebook + submissions
│       ├── start-here-a-gentle-introduction.ipynb
│       └── *.csv                                # Baseline submission files
│
├── datasets/                                    # Home Credit source data (not tracked in git)
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── application_train_trimmed.csv            # Trimmed for synthetic generation
│   └── trimmed_fields_map.json                  # Field metadata for generation
│
├── unstructured_data/                           # AI-generated documents per applicant
│   ├── paystubs/          # PNG — salary paystubs
│   ├── bank_statements/   # PDF — bank statements
│   ├── linkedin/          # PNG — LinkedIn profile screenshots
│   ├── id_documents/      # PNG — government ID cards
│   ├── property_docs/     # PDF — property/asset documents
│   ├── credit_reports/    # PDF — credit bureau reports
│   └── generation_log.jsonl
│
├── artifacts/                                   # Model outputs
│   ├── model_a_metrics.csv                      # Cross-validation metrics
│   ├── model_a_feature_importances.csv
│   ├── model_a_auc.txt                          # Best AUC: 0.7659
│   └── model_a_submission.csv
│
├── docs/
│   ├── PROMPT.md                                # LLM prompts used for document generation
│   └── SyntheticDataForBNPL.docx                # Project specification document
│
├── requirements.txt
└── .env                                         # API keys (not tracked)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/dhanrajsahoo-gif/credit-data-generation.git
cd credit-data-generation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
```

> Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Download Kaggle datasets

Download the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset and place the CSV files in the `datasets/` folder.

---

## Phases

### Phase 1 — Baseline Model (Model A)

**Notebook:** `01_baseline_model_executed.ipynb`

Trains a LightGBM model with 5-fold cross-validation on the Home Credit dataset with domain-engineered features.

**Results:**

| Metric | Value |
|--------|-------|
| Validation ROC-AUC | **0.7659** |
| Training ROC-AUC | ~0.815 |

---

### Phase 2 & 3 — Synthetic Data Generation

**Notebook:** `02_synthetic_data_generation.ipynb`

1. **Field trimming** — Identifies a subset of fields suitable for LLM-based synthetic generation and saves a `trimmed_fields_map.json`.
2. **Synthetic applicant generation** — Uses Google Gemini to generate realistic applicant records grounded in the original data distribution.
3. **Unstructured document generation** — For each synthetic applicant, generates 6 document types as images/PDFs using Gemini's multimodal capabilities.

---

### Phase 4 — LLM Extraction & Validation

**Notebook:** `03_extraction_and_validation_executed.ipynb`

Uses Gemini to extract structured fields from the generated unstructured documents and compares extracted values against the synthetic ground truth. Evaluates extraction accuracy across document types and fields.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **LightGBM** | Credit risk baseline model |
| **Google Gemini** (`google-genai`) | Synthetic data & document generation, LLM extraction |
| **pandas / scikit-learn** | Data processing and model evaluation |
| **Pillow** | Image handling for generated documents |
| **sentence-transformers** | Semantic similarity for extraction validation |
| **python-dotenv** | Environment variable management |

---

## Notes

- Dataset CSV files are excluded from git (`.gitignore`) due to size — download from Kaggle directly.
- The `.env` file is never tracked — never commit API keys.
- Applicant IDs start from `100002` and increment sequentially.
