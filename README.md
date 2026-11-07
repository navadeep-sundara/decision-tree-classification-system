# Decision Tree Classification System (ID3 vs C4.5)

An interactive Streamlit app that lets you upload a classification dataset (CSV), pick the target
attribute, preprocess the data, train **ID3** and **C4.5** decision trees (implemented from scratch with
NumPy/Pandas), visualize the trees, generate predictions, and compare the two models.

## Features
- CSV upload (or built-in Play Tennis / Iris samples)
- Target selection, column dropping, duplicate removal, missing-value handling (impute or drop)
- Train/test split with adjustable size, seed, stratification
- Hyper-parameters: max depth, minimum samples to split
- Tree visualization with Graphviz (plus a text view)
- Evaluation: confusion matrix, accuracy, precision, recall, F1-score, per-class report
- Side-by-side comparison table and bar chart, depth/nodes/leaves/training time, overfitting gap
- Test-set predictions (downloadable CSV) and single-sample prediction form

## Algorithms
| | ID3 | C4.5 |
|---|---|---|
| Split criterion | Information Gain | Gain Ratio (with average-gain filter) |
| Categorical features | Multiway split | Multiway split |
| Numeric features | Discretized into quantile bins first | Binary split at best threshold |
| Stopping | Pure node, no features left, max depth, min samples | Same |

Unseen categories at prediction time fall back to the majority class of the current node.

## Project structure
```
app.py              Streamlit UI, preprocessing, evaluation, comparison
decision_trees.py   ID3 and C4.5 implementations + Graphviz/text rendering
requirements.txt    Python dependencies
```

## Setup and run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The tree is rendered in the browser by Streamlit, so the Graphviz system binary is not required.
(Install it from https://graphviz.org/download/ only if you want to export PNG/PDF with `dot.render`.)

## Dataset format
A CSV with a header row, one column holding the class label, and any mix of numeric and
categorical feature columns. Drop ID-like columns in the sidebar so the trees do not split on them.

## Push to GitHub
```bash
git init
git add .
git commit -m "Add ID3 and C4.5 decision tree classification system"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```