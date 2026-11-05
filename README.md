# AI-Assisted Defect Detection

## Run in VS Code (Windows)

1. Extract this ZIP and open the folder in VS Code.
2. Open **Terminal > New Terminal**.
3. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install packages:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
streamlit run app.py
```

6. Open the local URL shown in the terminal (usually http://localhost:8501).

## Implemented
- Python static metric extraction using AST
- Random Forest defect-risk classifier
- Defect probability and risk category
- Global feature-importance chart
- Rule-based refactoring recommendations
- KNN retrieval of similar examples
- Accuracy, precision, recall, F1-score and confusion matrix

## Important limitation
The included training dataset is synthetic so the app runs immediately without downloading a dataset. It is not PROMISE/NASA data, and its evaluation scores must not be presented as real-world accuracy. For a valid project evaluation, train and test using a real labelled software-defect dataset.

This starter version analyses Python snippets only. Churn is entered manually. SHAP, automatic Git history extraction, Java/C++ parsing and CI/CD integration are not implemented yet.
