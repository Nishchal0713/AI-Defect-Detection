import streamlit as st
import pandas as pd
import numpy as np
import ast, re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

st.set_page_config(page_title="AI Defect Detection", page_icon="🛠️", layout="wide")
FEATURES = ["loc", "complexity", "nesting_depth", "parameter_count", "branch_count",
            "loop_count", "comment_ratio", "todo_count", "churn"]

def extract_metrics(code, churn=0):
    lines = code.splitlines()
    nonempty = [x for x in lines if x.strip()]
    comments = [x for x in lines if x.strip().startswith("#")]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Python syntax error on line {e.lineno}: {e.msg}")

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.branches = self.loops = self.depth = self.current = self.params = 0
        def enter(self):
            self.branches += 1
            self.current += 1
            self.depth = max(self.depth, self.current)
        def leave(self):
            self.current -= 1
        def visit_If(self, n):
            self.enter(); self.generic_visit(n); self.leave()
        def visit_For(self, n):
            self.loops += 1; self.enter(); self.generic_visit(n); self.leave()
        visit_AsyncFor = visit_For
        def visit_While(self, n):
            self.loops += 1; self.enter(); self.generic_visit(n); self.leave()
        def visit_Try(self, n):
            self.enter(); self.generic_visit(n); self.leave()
        def visit_FunctionDef(self, n):
            self.params += len(n.args.args) + len(n.args.kwonlyargs)
            self.generic_visit(n)
        visit_AsyncFunctionDef = visit_FunctionDef

    v = Visitor()
    v.visit(tree)
    complexity = 1 + v.branches + sum(isinstance(n, ast.BoolOp) for n in ast.walk(tree))
    return {
        "loc": len(nonempty), "complexity": complexity, "nesting_depth": v.depth,
        "parameter_count": v.params, "branch_count": v.branches, "loop_count": v.loops,
        "comment_ratio": len(comments) / max(len(nonempty), 1),
        "todo_count": len(re.findall(r"\\b(TODO|FIXME|BUG)\\b", code, re.I)),
        "churn": int(churn)
    }

def demo_data(n=1800, seed=42):
    rng = check_random_state(seed)
    d = pd.DataFrame({
        "loc": rng.randint(5, 220, n), "complexity": rng.randint(1, 35, n),
        "nesting_depth": rng.randint(0, 9, n), "parameter_count": rng.randint(0, 13, n),
        "branch_count": rng.randint(0, 25, n), "loop_count": rng.randint(0, 12, n),
        "comment_ratio": rng.uniform(0, .45, n), "todo_count": rng.randint(0, 6, n),
        "churn": rng.randint(0, 35, n)
    })
    risk = (.055*d.complexity + .42*d.nesting_depth + .22*d.parameter_count +
            .10*d.branch_count + .07*d.loop_count + .055*d.churn -
            1.1*d.comment_ratio + .35*d.todo_count)
    p = 1 / (1 + np.exp(-(risk - 5)))
    d["defect"] = (rng.uniform(0, 1, n) < p).astype(int)
    return d

@st.cache_resource
def train_model():
    data = demo_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data.defect, test_size=.2, random_state=42, stratify=data.defect)
    model = RandomForestClassifier(n_estimators=250, max_depth=10,
                                   class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "Confusion matrix": confusion_matrix(y_test, pred)
    }
    return model, data, scores

def fixes(m):
    out = []
    if m["complexity"] > 10:
        out.append(("High complexity", "Split the function into smaller functions and extract independent logic."))
    if m["nesting_depth"] > 3:
        out.append(("Deep nesting", "Use guard clauses and early returns to reduce nested conditions."))
    if m["parameter_count"] > 5:
        out.append(("Long parameter list", "Group related parameters into a configuration or parameter object."))
    if m["branch_count"] > 12:
        out.append(("Many branches", "Simplify conditional logic and consider clearer mapping or strategy structures."))
    if m["loop_count"] > 6:
        out.append(("Many loops", "Review repeated iteration and extract reusable processing steps."))
    if m["todo_count"] > 0:
        out.append(("TODO/FIXME markers", "Resolve unfinished or explicitly flagged code before merging."))
    if m["churn"] > 15:
        out.append(("High code churn", "Add focused unit tests around frequently modified logic."))
    return out or [("No major metric warning", "Continue code review and automated testing; low risk does not guarantee bug-free code.")]

def nearest_cases(query, data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[FEATURES])
    q = scaler.transform(pd.DataFrame([query])[FEATURES])
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    distances, indices = nn.kneighbors(q)
    result = data.iloc[indices[0]][FEATURES + ["defect"]].copy()
    result.insert(0, "distance", distances[0].round(3))
    result["label"] = result.defect.map({0: "Clean", 1: "Defective"})
    return result.drop(columns="defect")

st.title("🛠️ AI-Assisted Defect Detection")
st.caption("Classical Machine Learning | Explainable predictions | Fix recommendations")
model, data, scores = train_model()

tab1, tab2, tab3 = st.tabs(["Analyse source code", "Model evaluation", "About the features"])
with tab1:
    example = '''def process_order(items, discount, tax, customer, address, payment):
    total = 0
    if customer:
        if address:
            if payment:
                for item in items:
                    if item > 0:
                        total += item
    return total'''
    code = st.text_area("Paste Python source code", value=example, height=260)
    churn = st.number_input("Changes in the last 30 days (churn)", 0, 1000, 0)
    if st.button("Analyse code", type="primary"):
        try:
            m = extract_metrics(code, churn)
            x = pd.DataFrame([m])[FEATURES]
            probability = float(model.predict_proba(x)[0][1])
            label = "Potential defect" if probability >= .5 else "Lower risk"
            risk = "High" if probability >= .70 else ("Medium" if probability >= .40 else "Low")
            a, b, c = st.columns(3)
            a.metric("Defect risk", f"{probability*100:.1f}%")
            b.metric("Prediction", label)
            c.metric("Risk category", risk)
            st.subheader("Extracted metrics")
            st.dataframe(pd.DataFrame([m]), use_container_width=True)
            st.subheader("Important model features")
            st.bar_chart(pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False))
            st.caption("Global feature importance, not a per-prediction SHAP explanation.")
            st.subheader("Suggested improvements")
            for title, detail in fixes(m):
                st.markdown(f"**{title}:** {detail}")
            st.subheader("Similar historical examples (KNN)")
            st.dataframe(nearest_cases(m, data), use_container_width=True)
            st.info("A risk score is not proof of a bug. Review and test the code.")
        except ValueError as e:
            st.error(str(e))

with tab2:
    st.subheader("Held-out demo test results")
    cols = st.columns(4)
    for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score"]):
        col.metric(key, f"{scores[key]*100:.1f}%")
    st.write("Confusion matrix (rows = actual; columns = predicted)")
    st.dataframe(pd.DataFrame(scores["Confusion matrix"],
        index=["Actual clean", "Actual defective"],
        columns=["Predicted clean", "Predicted defective"]))
    st.caption("These scores are only for synthetic demonstration data, not real-world performance.")

with tab3:
    meaning = {
        "loc": "Non-empty source lines", "complexity": "Approximate cyclomatic complexity",
        "nesting_depth": "Maximum nested decision/loop depth", "parameter_count": "Function parameters",
        "branch_count": "If/loop/try decision structures", "loop_count": "For/while loops",
        "comment_ratio": "Comment lines divided by non-empty lines",
        "todo_count": "TODO, FIXME and BUG markers", "churn": "Changes entered manually for the last 30 days"
    }
    st.table(pd.DataFrame({"Feature": meaning.keys(), "Meaning": meaning.values()}))
    st.markdown("**Current scope:** Python snippets only. Git churn is entered manually. Automatic Git integration, Java/C/C++ parsing, SHAP explanations and CI/CD pull-request comments are future extensions.")
