# LLM-Powered ML Notebooks

Two Jupyter notebooks that combine classical machine learning with large language model reasoning via the Groq API.

---

## Notebooks

### 1. `Choosing_right_parameters_using_Tree_of_Thought_.ipynb`

**Goal:** Automatically select the best hyperparameter configuration for a Random Forest classifier using Tree-of-Thought (ToT) reasoning.

**How it works:**

1. Loads the **Breast Cancer** dataset from scikit-learn.
2. Trains a `RandomForestClassifier` across all combinations of:
   - `n_estimators`: [50, 100, 200]
   - `max_depth`: [3, 5, 10]
   - `min_samples_split`: [2, 5]
3. Records CV accuracy, train/test accuracy, overfitting gap, and training time for each combination.
4. Sends the results table to an LLM (via Groq) with a **Tree-of-Thought prompt** that branches across four analysis dimensions — performance, overfitting, training cost, and bias-variance tradeoff — before merging into a single recommended configuration.

**Outputs:**
- `hyperparameter_results.csv` — full experiment results
- `tree_of_thought_analysis.txt` — LLM's reasoning and final recommendation

**Key dependencies:** `scikit-learn`, `pandas`, `numpy`, `groq`

---

### 2. `Sentiment_with_CoT_from_Customer_Reviews.ipynb`

**Goal:** Classify customer review sentiment (Positive / Neutral / Negative) using **few-shot Chain-of-Thought (CoT)** prompting.

**How it works:**

1. Defines a list of 10 sample customer reviews.
2. Builds a structured CoT prompt with three labeled examples, each walking through five explicit reasoning steps:
   - Identify positive phrases
   - Identify negative phrases
   - Check for mixed/contradictory sentiment
   - Reason step-by-step
   - Output a final label
3. Calls the LLM (via Groq) for each review and parses the `Step 5: Final Label` from the response.
4. Appends the predicted sentiment back to the DataFrame.

**Key dependencies:** `pandas`, `groq`

---

## Setup

### Prerequisites

- Python 3.8+
- A [Groq](https://console.groq.com/) API key

### Install dependencies

```bash
pip install groq pandas scikit-learn numpy
```

### Configure API key

Both notebooks retrieve the Groq API key from Google Colab secrets:

```python
from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('groq')
```

If running outside Colab, replace this with:

```python
os.environ['GROQ_API_KEY'] = "your_groq_api_key_here"
```

---

## Prompting Techniques Used

| Technique | Notebook | Description |
|---|---|---|
| **Tree-of-Thought (ToT)** | Hyperparameter selection | LLM explores multiple reasoning branches in parallel before converging on a decision |
| **Few-shot Chain-of-Thought (CoT)** | Sentiment analysis | LLM is guided by labeled examples that demonstrate explicit step-by-step reasoning |

---

## Model

Both notebooks use `openai/gpt-oss-120b` served through the Groq API.
