# Synthetic Unstructured Data Generation & Validation — Barclays BNPL PoC

## Business Context

This is a **Barclays Proof of Concept** for BNPL (Buy Now Pay Later) default risk modeling. The central question is:

> *Can synthetic unstructured data (paystubs, LinkedIn profiles, bank statements, ID documents, credit reports) derived from structured loan application data serve as a viable replacement — or complement — for structured data in predicting credit default?*

Two potential PoC narratives:

1. **Confidence narrative** — The closer the accuracy between the structured-only model and the unstructured-only model, the higher our confidence in the ability to use alternative unstructured data for BNPL default modelling at scale.
2. **Improvement narrative** — Alternative unstructured income and employment verification data improves predictive power by **X% AUC** and reduces expected loss by **Y%** when combined with structured data.

The PoC produces three models and compares them head-to-head:

| Model | Input | Purpose |
|---|---|---|
| **Model A** | Structured data only | Home Credit baseline — gold standard |
| **Model B** | Unstructured data only | Features extracted by LLM from synthetic docs |
| **Model C** | Structured + Unstructured | Combined hybrid — measures incremental gain |

---

## Repository Structure

```
Credit-Data-Generation/
├── .env                                      # API keys (never commit — in .gitignore)
├── .gitignore
├── requirements.txt
├── PROMPT.md                                 # This file
├── SyntheticDataForBNPL.docx                 # Original PoC brief
├── datasets/
│   ├── application_train.csv                 # 307,511 rows × 122 cols
│   ├── application_test.csv                  # 48,744 rows × 121 cols
│   ├── bureau.csv                            # External credit bureau history
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── installments_payments.csv
│   ├── credit_card_balance.csv
│   ├── HomeCredit_columns_description.csv
│   ├── application_train_trimmed.csv         # generated in Phase 2
│   └── application_train_reconstructed.csv   # generated in Phase 4
├── notebooks/
│   ├── start-here-a-gentle-introduction-executed.ipynb   # Phase 1 reference (already run)
│   ├── 01_baseline_model.ipynb               # Phase 1 — clean baseline notebook
│   ├── 02_synthetic_data_generation.ipynb    # Phase 2 + 3
│   └── 03_extraction_and_validation.ipynb    # Phase 4 — all 3 model comparisons
└── unstructured_data/
    ├── generation_log.jsonl
    ├── extraction_results.jsonl
    ├── paystubs/
    ├── linkedin/
    ├── bank_statements/
    ├── id_documents/
    ├── property_docs/
    └── credit_reports/
```

---

## Environment Setup

### API Keys

All API keys are stored in `.env` at the project root. The `.env` file is listed in `.gitignore` and must **never** be committed to version control.

```
GOOGLE_API_KEY=<your Gemini key>
OPENAI_API_KEY=<your OpenAI key>
ANTHROPIC_API_KEY=<your Anthropic key>
```

### LLM Configuration

Expose a single config block at the top of every notebook. Changing `LLM_PROVIDER` switches the **entire pipeline** (both generation and extraction) to a different model with no other code changes required.

```python
# ── LLM Configuration ──────────────────────────────────────────────────────
LLM_PROVIDER    = "gemini"               # options: "gemini" | "openai" | "anthropic"
GEMINI_MODEL    = "gemini-2.0-flash"
OPENAI_MODEL    = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
# ───────────────────────────────────────────────────────────────────────────
```

---

## Testing & Sample Size

### Initial Test Run (10 data points)

For the **initial end-to-end test**, use only the **first 10 rows** of `application_train.csv`. This validates the full pipeline (generation → extraction → model comparison) cheaply before scaling.

- Sample: first 10 rows — no stratification required at this scale
- All generated documents stored under `unstructured_data/` in the appropriate subfolder
- File naming: `{SK_ID_CURR}_{document_type}.{ext}` (e.g., `100002_paystub.png`, `100002_linkedin.png`)
- All 10 applicants must complete generation and extraction before the model comparison step runs

### Production Run

Once the 10-row test passes end-to-end:

- Scale to a **stratified sample of 2,000 applicants** — 1,000 `TARGET=0`, 1,000 `TARGET=1`
- Implement exponential backoff (initial 1s, multiplier ×2, max 5 retries) on all API calls
- Log every call to `unstructured_data/generation_log.jsonl`

---

## Data Source

**Ground truth:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)

An extremely strong dataset for serious credit modelling. Key tables:

| Table | Description |
|---|---|
| `application_train/test.csv` | Main loan application data — 122 columns covering demographics, income, employment, housing, document flags, credit bureau inquiry counts |
| `bureau.csv` | External credit bureau history per applicant |
| `bureau_balance.csv` | Monthly bureau status history |
| `previous_application.csv` | Prior Home Credit loan applications |
| `POS_CASH_balance.csv` | Monthly POS/cash loan snapshots |
| `installments_payments.csv` | Payment history for prior loans |
| `credit_card_balance.csv` | Monthly credit card balance history |

### Key Field Mapping (from PoC brief)

The following Home Credit fields are the primary inputs into synthetic document generation:

| Home Credit Field | Meaning | Primary Synthetic Document |
|---|---|---|
| `AMT_INCOME_TOTAL` | Annual income | Paystub gross pay |
| `DAYS_EMPLOYED` | Employment duration | Tenure on LinkedIn + Paystub |
| `NAME_INCOME_TYPE` | Employment type | Job type on both documents |
| `OCCUPATION_TYPE` | Occupation | LinkedIn job title |
| `ORGANIZATION_TYPE` | Industry | Employer type / name |
| `EXT_SOURCE_1/2/3` | External risk proxy | Latent stability factor → Credit report |
| `DAYS_BIRTH` | Age | Seniority level (LinkedIn) + ID document |
| `FLAG_OWN_CAR`, `FLAG_OWN_REALTY` | Asset proxy | Profile stability + Property doc |
| `CNT_FAM_MEMBERS` | Household size | Payroll deductions realism |
| `TARGET` | Default label | **DO NOT use directly in generation** |

---

## Latent Stability Score (Critical Design Principle)

**Do not use `TARGET` directly** when generating documents. Instead, derive a **latent employment stability score** that correlates with risk without leaking the label:

```
stability_score = (
    normalize(EXT_SOURCE_2)          × 0.4 +
    normalize(−DAYS_EMPLOYED)        × 0.3 +
    normalize(AMT_INCOME_TOTAL)      × 0.2 +
    normalize(FLAG_OWN_REALTY)       × 0.1
)
```

This score drives the **realism and noise level** of generated documents:

- **High stability score** → clean, well-formatted documents; clear career progression; complete fields; no OCR artifacts
- **Low stability score** → document noise (OCR corruption: `0→O`, `1→l`), missing fields, formatting inconsistencies, employer name truncation, career gaps, lateral job moves

This ensures synthetic documents correlate meaningfully with default risk without directly encoding the label.

---

## Phase 1 — Baseline Model (Structured Data Only = Model A)

**Notebook:** `notebooks/01_baseline_model.ipynb`

**Reference execution:** `notebooks/start-here-a-gentle-introduction-executed.ipynb` (already run — refer to this for the exact pipeline, outputs, and AUC scores)

**Objective:** Reproduce the baseline from the reference notebook in a clean, standalone notebook. This becomes **Model A** — the gold standard.

### Pipeline (mirrors the reference notebook)

**Data loading:**
- Load `application_train.csv` (307,511 rows × 122 columns)
- Load `application_test.csv` (48,744 rows × 121 columns)

**Preprocessing:**
- Apply `LabelEncoder` to categorical columns with ≤ 2 unique values
- Apply `pd.get_dummies()` to remaining categorical columns
- Align train and test columns after one-hot encoding (fill missing columns with 0)
- `DAYS_EMPLOYED` anomaly: the value `365243` is a known data anomaly. Create `DAYS_EMPLOYED_ANOM` flag column, then replace `365243` with `NaN`
- Impute missing values with **median** using `SimpleImputer(strategy='median')`

**Domain feature engineering** (from reference notebook, inspired by Aguiar's Kaggle script):
- `CREDIT_INCOME_PERCENT` = `AMT_CREDIT / AMT_INCOME_TOTAL`
- `ANNUITY_INCOME_PERCENT` = `AMT_ANNUITY / AMT_INCOME_TOTAL`
- `CREDIT_TERM` = `AMT_ANNUITY / AMT_CREDIT`
- `DAYS_EMPLOYED_PERCENT` = `DAYS_EMPLOYED / DAYS_BIRTH`

**Model — LightGBM with 5-fold cross-validation** (exact parameters from reference notebook):

```python
model = lgb.LGBMClassifier(
    n_estimators   = 10000,
    objective      = 'binary',
    class_weight   = 'balanced',
    learning_rate  = 0.05,
    reg_alpha      = 0.1,
    reg_lambda     = 0.1,
    subsample      = 0.8,
    n_jobs         = -1,
    random_state   = 50
)
# Early stopping: patience = 100 rounds, log every 200 rounds
```

**Expected results from reference notebook:**
- Logistic Regression baseline: ~0.671 ROC-AUC
- Random Forest (100 trees): ~0.678 ROC-AUC (Kaggle leaderboard)
- LightGBM 5-fold cross-validation: **~0.7587 validation AUC** (overall)
- LightGBM with domain features: **~0.770 validation AUC**
- Kaggle leaderboard submission: ~0.735

**Record the final validation ROC-AUC as the Model A benchmark.** Save the model, feature importances, and out-of-fold predictions to disk.

---

## Phase 2 — Identify & Trim Fields for Synthetic Generation

**Notebook:** `notebooks/02_synthetic_data_generation.ipynb` (first section)

**Objective:** Remove structured fields that will be represented as unstructured documents, producing `application_train_trimmed.csv` for use in Models B and C.

### Fields to Trim by Document Type

#### Group 1 — Paystub Image *(primary document)*

Remove from `application_train.csv`:

| Field | Description | Used For |
|---|---|---|
| `AMT_INCOME_TOTAL` | Annual income | Gross pay / net pay calculation |
| `DAYS_EMPLOYED` | Employment duration | Tenure / hire date |
| `NAME_INCOME_TYPE` | Working, Pensioner, State servant, Commercial associate | Employment type |
| `OCCUPATION_TYPE` | Laborers, Sales staff, Core staff, Managers, Drivers | Job title |
| `ORGANIZATION_TYPE` | Business Entity Type 1/2/3, Government, School, Medicine | Employer name mapping |
| `CNT_FAM_MEMBERS` | Number of family members | Dependent deductions |
| `FLAG_EMP_PHONE` | Has employer phone | Signals formal employment |

#### Group 2 — LinkedIn Profile Card Image *(primary document)*

Remove from `application_train.csv`:

| Field | Description | Used For |
|---|---|---|
| `NAME_INCOME_TYPE` | Employment type | Headline context |
| `OCCUPATION_TYPE` | Occupation | Current role / job title |
| `ORGANIZATION_TYPE` | Industry | Employer type |
| `DAYS_EMPLOYED` | Employment duration | Number of jobs, seniority, progression |
| `AMT_INCOME_TOTAL` | Annual income | Seniority level proxy |
| `NAME_EDUCATION_TYPE` | Secondary, Higher education, Incomplete higher | Education section |
| `DAYS_BIRTH` | Age | Seniority level |

#### Group 3 — Bank Statement PDF

Remove from `application_train.csv` and aggregated `bureau.csv`:

| Field | Description |
|---|---|
| `AMT_INCOME_TOTAL` | Salary deposits |
| `AMT_ANNUITY` | Regular loan repayment debits |
| `AMT_CREDIT` | Loan amount |
| `AMT_CREDIT_SUM` *(bureau agg)* | Total outstanding credit |
| `AMT_CREDIT_SUM_DEBT` *(bureau agg)* | Outstanding debt |
| `AMT_CREDIT_MAX_OVERDUE` *(bureau agg)* | Maximum overdue amount |
| `CREDIT_DAY_OVERDUE` *(bureau agg)* | Days overdue |
| `CREDIT_ACTIVE` *(bureau agg)* | Active / closed / bad debt |
| `CNT_CREDIT_PROLONG` *(bureau agg)* | Times credit was prolonged |

#### Group 4 — Government ID / Passport Scan Image

Remove from `application_train.csv`:

| Field | Description |
|---|---|
| `CODE_GENDER` | M / F |
| `DAYS_BIRTH` | Convert to date of birth |
| `NAME_FAMILY_STATUS` | Single, Married, Separated, Widow |
| `CNT_CHILDREN` | Number of children |
| `CNT_FAM_MEMBERS` | Number of family members |
| `DAYS_ID_PUBLISH` | How recently ID was reissued |
| `FLAG_DOCUMENT_3`, `FLAG_DOCUMENT_6`, `FLAG_DOCUMENT_8` | Most frequently submitted document flags |

#### Group 5 — Property / Utility Bill PDF

Remove from `application_train.csv`:

| Field | Description |
|---|---|
| `NAME_HOUSING_TYPE` | House/apartment, With parents, Municipal apartment |
| `FLAG_OWN_REALTY` | Owns real estate |
| `FLAG_OWN_CAR` | Owns a car |
| `OWN_CAR_AGE` | Age of car in years |
| `REGION_RATING_CLIENT` | Region quality rating |
| `REGION_RATING_CLIENT_W_CITY` | Region rating including city |
| `REG_CITY_NOT_LIVE_CITY` | Registration vs. living city mismatch |
| `REG_CITY_NOT_WORK_CITY` | Registration vs. work city mismatch |
| `LIVE_CITY_NOT_WORK_CITY` | Living vs. work city mismatch |
| `APARTMENTS_AVG`, `LIVINGAREA_AVG`, `FLOORSMAX_AVG`, `TOTALAREA_MODE` | Building metrics |

#### Group 6 — Credit Bureau Report PDF

Remove from `application_train.csv`:

| Field | Description |
|---|---|
| `EXT_SOURCE_1` | External credit score 1 (normalized 0–1) |
| `EXT_SOURCE_2` | External credit score 2 (normalized 0–1) |
| `EXT_SOURCE_3` | External credit score 3 (normalized 0–1) |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | Bureau inquiries — last hour |
| `AMT_REQ_CREDIT_BUREAU_DAY` | Bureau inquiries — last day |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | Bureau inquiries — last week |
| `AMT_REQ_CREDIT_BUREAU_MON` | Bureau inquiries — last month |
| `AMT_REQ_CREDIT_BUREAU_QRT` | Bureau inquiries — last quarter |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | Bureau inquiries — last year |

**Output:** `datasets/application_train_trimmed.csv` + `datasets/trimmed_fields_map.json` recording which fields were removed for each document type.

---

## Phase 3 — Synthetic Unstructured Data Generation

**Notebook:** `notebooks/02_synthetic_data_generation.ipynb` (second section)

**Objective:** For each applicant in the sample, generate realistic synthetic documents using the removed fields via the configured LLM (default: `gemini-2.0-flash`).

### General Principles

- Embed the **exact numeric and categorical values** from trimmed fields — do not hallucinate data values
- All names, addresses, account numbers are synthetically generated (dataset contains no real PII)
- File naming: `{SK_ID_CURR}_{document_type}.{ext}` in the appropriate `unstructured_data/` subfolder
- Compute `stability_score` per applicant before generation — use it to calibrate noise level
- Log every API call to `unstructured_data/generation_log.jsonl`: `SK_ID_CURR`, document type, model, prompt/response token counts, generation time, success/failure

---

### Document 1 — Paystub *(primary)*

**Storage:** `unstructured_data/paystubs/{SK_ID_CURR}_paystub.png`

**Generation method:** Gemini text generation (JSON schema → rendered paystub image)

**Paystub schema to populate:**

```json
{
  "employer_name": "",
  "industry": "",
  "pay_frequency": "Biweekly",
  "gross_pay": 0,
  "net_pay": 0,
  "federal_tax": 0,
  "state_tax": 0,
  "insurance_deduction": 0,
  "retirement_deduction": 0,
  "hire_date": "",
  "pay_date": ""
}
```

**Field generation rules:**

- **Employer name:** map `ORGANIZATION_TYPE` to plausible employer names:
  ```
  "Business Entity Type 3" → ["Vertex Solutions", "CoreAxis LLC", ...]
  "Self-employed"          → ["Independent Contractor"]
  "Government"             → ["City of [Region]", "Municipal Office"]
  ```
- **Gross pay:** `(AMT_INCOME_TOTAL / 12) / 2` (biweekly), add ±5% realistic variance
- **Net-to-gross ratio** by income tier:
  - Low income → 80–85%
  - Mid income → 70–80%
  - High income → 60–75%
- **Federal / state tax** derived from net-to-gross gap
- **Insurance deduction:** scaled to `CNT_FAM_MEMBERS`
- **Retirement deduction:** present only for higher-stability profiles; omit or zero for low stability
- **Hire date:** `today − (abs(DAYS_EMPLOYED) / 365)`

**Noise injection based on `stability_score`:**
- Low stability → OCR corruption (`0→O`, `1→l`), employer name truncation, missing retirement deduction, inconsistent date formats
- High stability → clean, complete, professional formatting

**Prompt template to Gemini:**

> Generate a realistic paystub image for an employee with the following attributes. The paystub should look like a printed/scanned payroll document from a real employer. Render it as a PNG image.
>
> Employer: `{employer_name}` | Industry: `{ORGANIZATION_TYPE}` | Pay frequency: Biweekly
> Pay date: `{pay_date}` | Pay period: `{period_start}` – `{period_end}`
> Gross pay: `{gross_pay}` | Federal tax: `{federal_tax}` | State tax: `{state_tax}`
> Insurance: `{insurance_deduction}` | 401k/Retirement: `{retirement_deduction}`
> Net pay: `{net_pay}` | Hire date: `{hire_date}`
>
> Stability level: `{stability_score:.2f}` (0=low, 1=high). Apply document quality accordingly — low stability documents should have minor OCR artifacts, formatting inconsistencies, or truncated fields. High stability documents should be clean and professional.
> Return only the image. Do not include any commentary.

---

### Document 2 — LinkedIn Profile Card Image *(primary)*

**Storage:** `unstructured_data/linkedin/{SK_ID_CURR}_linkedin.png`

**Generation method:** Text-to-image via Gemini image generation

**Profile schema:**
```json
{
  "headline": "",
  "current_role": "",
  "industry": "",
  "experience": [],
  "education": "",
  "skills": [],
  "connections": 0,
  "profile_completeness": 0.0
}
```

**Career path logic driven by `stability_score` and `DAYS_EMPLOYED`:**

| Profile trait | High stability | Low stability |
|---|---|---|
| Number of employers | 1–2 | 4–6 |
| Career progression | Clear promotion track | Lateral moves, gaps |
| Skill count | 8–15 strong skills | 2–4 generic skills |
| Connections | 400–700 | 50–150 |
| Profile completeness | 0.85–1.0 | 0.3–0.6 |

**Example high stability (Senior Financial Analyst at Vertex Solutions):**
```
Experience: Senior Financial Analyst (2019–Present), Financial Analyst (2016–2019)
Education: B.S. Finance, State University
Skills: Financial Modeling, Risk Analysis, SQL, Budget Forecasting
Connections: 524
```

**Example low stability:**
```
Experience: Warehouse Associate (2023–Present), Delivery Driver (2022–2023), Retail Associate (2021–2022)
Skills: Customer Service, Inventory
Connections: 87
```

**Prompt template to Gemini:**

> Generate a realistic LinkedIn profile screenshot cropped to the Experience and About sections. The person is from Eastern Europe, gender `{CODE_GENDER}`.
>
> Role: `{OCCUPATION_TYPE}` at `{ORGANIZATION_TYPE}` | Income type: `{NAME_INCOME_TYPE}`
> Years employed: `{abs(DAYS_EMPLOYED)/365:.1f}` | Education: `{NAME_EDUCATION_TYPE}` | Age: `{abs(DAYS_BIRTH)/365:.0f}`
> Stability score: `{stability_score:.2f}` (0=low, 1=high)
>
> High stability → 1–2 employers, clear progression, strong skill set, 400+ connections, high completeness.
> Low stability → 4–6 short jobs, career gaps, sparse skills, under 150 connections.
> The image must look like an authentic browser screenshot, not a diagram or illustration.

---

### Document 3 — Bank Statement PDF

**Storage:** `unstructured_data/bank_statements/{SK_ID_CURR}_bank_statement.pdf`

**Generation method:** Gemini text generation → HTML → `weasyprint` PDF render

**Prompt template to Gemini:**

> Generate complete HTML markup for a realistic 3-month bank statement from a generic Eastern European retail bank. Include:
> - Account holder name consistent with gender `{CODE_GENDER}`
> - Regular salary credits of `{AMT_INCOME_TOTAL/12:.2f}` per month
> - Regular loan repayment debits of `{AMT_ANNUITY/12:.2f}` per month for a loan of `{AMT_CREDIT}`
> - If `{CREDIT_DAY_OVERDUE} > 0`: include a late fee row and overdue notice
> - Prior credit entries: total credit `{AMT_CREDIT_SUM}`, outstanding debt `{AMT_CREDIT_SUM_DEBT}`
> - Plausible closing balance, bank logo placeholder, account number, IBAN footer
>
> Return only the HTML. Do not include markdown fences.

---

### Document 4 — Government ID Scan Image

**Storage:** `unstructured_data/id_documents/{SK_ID_CURR}_id.png`

**Generation method:** Text-to-image via Gemini image generation

**Prompt template to Gemini:**

> Generate a realistic scanned image of a generic Central/Eastern European national ID card. Show:
> - Holder name (plausible for gender `{CODE_GENDER}`)
> - Date of birth: `{date_of_birth}` (derived from `DAYS_BIRTH`)
> - Marital status: `{NAME_FAMILY_STATUS}` | Children: `{CNT_CHILDREN}` | Family members: `{CNT_FAM_MEMBERS}`
> - ID issue date derived from `{DAYS_ID_PUBLISH}`
> - Text in both Latin and Cyrillic scripts; slight scan artifacts (off-center, edge shadow)
>
> Do not replicate security features of any real country's ID.

---

### Document 5 — Property / Utility Bill PDF

**Storage:** `unstructured_data/property_docs/{SK_ID_CURR}_property.pdf`

**Generation method:** Gemini text generation → HTML → `weasyprint` PDF render

**Prompt template to Gemini:**

> Generate HTML for a realistic utility bill or property assessment. Include:
> - Housing type: `{NAME_HOUSING_TYPE}` | Ownership: `{FLAG_OWN_REALTY}` | Car: `{FLAG_OWN_CAR}` (age `{OWN_CAR_AGE}` yrs)
> - Building: `{FLOORSMAX_AVG:.0f}` floors, living area `{LIVINGAREA_AVG:.1f}` m², total `{TOTALAREA_MODE:.1f}` m²
> - Region rating `{REGION_RATING_CLIENT}/3` | City mismatches: reg/live=`{REG_CITY_NOT_LIVE_CITY}`, live/work=`{LIVE_CITY_NOT_WORK_CITY}`
>
> Return only the HTML.

---

### Document 6 — Credit Bureau Report PDF

**Storage:** `unstructured_data/credit_reports/{SK_ID_CURR}_credit_report.pdf`

**Generation method:** Gemini text generation → HTML → `weasyprint` PDF render

**Prompt template to Gemini:**

> Generate HTML for a professional consumer credit bureau report. Include:
> - Credit sub-scores from `EXT_SOURCE_1` (`{EXT_SOURCE_1:.3f}`), `EXT_SOURCE_2` (`{EXT_SOURCE_2:.3f}`), `EXT_SOURCE_3` (`{EXT_SOURCE_3:.3f}`) — scale each to 300–850
> - Risk tier: Excellent / Good / Fair / Poor (from weighted average of the three scores)
> - Inquiry history: hour=`{AMT_REQ_CREDIT_BUREAU_HOUR}`, day=`{AMT_REQ_CREDIT_BUREAU_DAY}`, week=`{AMT_REQ_CREDIT_BUREAU_WEEK}`, month=`{AMT_REQ_CREDIT_BUREAU_MON}`, quarter=`{AMT_REQ_CREDIT_BUREAU_QRT}`, year=`{AMT_REQ_CREDIT_BUREAU_YEAR}`
> - Tradeline summary, report date, reference number, applicant name
>
> Return only the HTML.

---

## Phase 4 — LLM Extraction + Three-Model Comparison

**Notebook:** `notebooks/03_extraction_and_validation.ipynb`

**Objective:** Extract structured features from the synthetic documents using LLM, then train and compare all three models.

### Step 4a — LLM Feature Extraction per Document Type

For each document, call the configured LLM to extract fields as structured JSON with a `confidence` score (1–5) per field.

**From Paystub:**
- Parsed gross pay, net pay
- Net-to-gross ratio
- Employment tenure (string → years)
- Missing field indicator (retirement deduction present/absent)
- OCR noise score (estimated document quality)

**From LinkedIn:**
- Number of job hops
- Seniority keyword score (keyword match: Associate/Analyst/Senior/Manager/Director/VP)
- Skill count
- Career progression index (promotions vs lateral moves)
- SBERT embedding vector (optional, for Model C)

**From Bank Statement:**
- Monthly income → `AMT_INCOME_TOTAL`
- Monthly repayment → `AMT_ANNUITY`
- Overdue flags, `AMT_CREDIT_SUM`, `AMT_CREDIT_SUM_DEBT`

**From ID Document:**
- `CODE_GENDER`, date of birth → `DAYS_BIRTH`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`

**From Property Doc:**
- `NAME_HOUSING_TYPE`, `FLAG_OWN_REALTY`, `FLAG_OWN_CAR`, `OWN_CAR_AGE`, region rating, building metrics

**From Credit Report:**
- `EXT_SOURCE_1/2/3` (reverse-scale from 300–850 back to 0–1), all `AMT_REQ_CREDIT_BUREAU_*` fields

Save all extraction outputs to `unstructured_data/extraction_results.jsonl`.

### Step 4b — Reconstruct Dataset

- Merge `application_train_trimmed.csv` with extracted fields on `SK_ID_CURR`
- For fields with extraction `confidence < 3` or null: impute with **trimmed dataset's population median/mode** (not original values — simulates real-world noise)
- Save as `datasets/application_train_reconstructed.csv`

### Step 4c — Train and Compare All Three Models

Use **identical** LightGBM hyperparameters, feature engineering, and validation split as Phase 1.

| Model | Training data | Description |
|---|---|---|
| **Model A** | `application_train.csv` (full) | Structured only — Phase 1 gold standard |
| **Model B** | Extracted features only (from docs) | Unstructured only — no original structured fields |
| **Model C** | `application_train_reconstructed.csv` | Structured trimmed + LLM-extracted fields combined |

### Evaluation Metrics

Report the following for all three models side-by-side:

| Metric | Description |
|---|---|
| **ROC-AUC** | Primary ranking metric |
| **KS Statistic** | Kolmogorov-Smirnov score (separation between good/bad distributions) |
| **Gini Coefficient** | `2 × AUC − 1` |
| **Approval vs Bad Rate curve** | At various score cutoffs, show approval rate vs. bad rate |

Additional diagnostics:
- Side-by-side confusion matrices (at 0.5 threshold)
- Calibration plots for all three models
- Top-20 feature importance comparison (Models A vs C)
- Extraction accuracy per field: ≥ 85% exact match for categoricals, ≥ 80% within ±10% for numerics

---

## Success Criteria

| Metric | Target |
|---|---|
| Document generation success rate | ≥ 95% of sample |
| LLM extraction valid JSON parse rate | ≥ 90% |
| Extraction accuracy — categorical fields | ≥ 85% exact match |
| Extraction accuracy — numeric fields | ≥ 80% within ±10% of true value |
| ROC-AUC delta (Model A vs Model C) | < 0.02 |
| Model B ROC-AUC | > 0.70 (demonstrates unstructured data alone is viable) |

---

## Technical Constraints

- Default LLM: `gemini-2.0-flash` — change via `LLM_PROVIDER` config block; supports `gemini`, `openai`, `anthropic`
- HTML-to-PDF: `weasyprint` (headless rendering — not raw HTML saved as `.pdf`)
- API calls: exponential backoff, initial 1s, multiplier ×2, max 5 retries
- No real PII — all names, addresses, account numbers are synthetic
- `DAYS_EMPLOYED == 365243` is a known anomaly in the dataset — must be flagged and replaced with `NaN` before any modelling step
- `.env` must never be committed — enforced by `.gitignore`
- All generated files under `unstructured_data/` with consistent subfolder and naming convention
- Every API call logged to `unstructured_data/generation_log.jsonl`
