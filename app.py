import pandas as pd
import unicodedata
import time
import numpy as np

from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# 1. CLEAN TEXT FUNCTION
# =========================
def normalize_text(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv(r"C:\Users\marie\Downloads\dataset_etudiants.csv")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .map(normalize_text)
    .str.replace(" ", "_")
    .str.replace("'", "")
)

# Fill NaN safely
for col in df.columns:
    df[col] = df[col].astype(str).fillna("")

print("Colonnes détectées :", df.columns)


# =========================
# 3. SELECT FEATURES
# =========================
cols = [
    "travaux_collaboratifs",
    "coequipiers",
    "communautes",
    "competences",
    "centres_dinteret"
]

# Build profile safely
df["profile"] = (
    df[cols]
    .astype(str)
    .apply(lambda x: " ".join([w for w in x if w.strip() != ""]), axis=1)
)

# Remove empty profiles
df = df[df["profile"].str.strip() != ""]

print("\nSample profiles:")
print(df["profile"].head(5))


# =========================
# 4. TF-IDF MODEL
# =========================
print("\nRunning TF-IDF...")

start_time1 = time.time()

tfidf_vectorizer = TfidfVectorizer(
    stop_words=None,
    token_pattern=r"(?u)\b\w+\b"
)

X_tfidf = tfidf_vectorizer.fit_transform(df["profile"])
tfidf_similarity = cosine_similarity(X_tfidf)

execution_time1 = time.time() - start_time1


# =========================
# 5. BERT MODEL
# =========================
print("\nLoading BERT model...")

start_time = time.time()

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["profile"].tolist(), show_progress_bar=True)

bert_similarity = cosine_similarity(embeddings)

execution_time = time.time() - start_time


# =========================
# 6. EVALUATION FUNCTION
# =========================
def evaluate(sim_matrix):
    y_true = []
    y_pred = []

    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                y_true.append(
                    1 if df.iloc[i]["competences"] == df.iloc[j]["competences"] else 0
                )
                y_pred.append(sim_matrix[i][j])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return rmse, mae


bert_rmse, bert_mae = evaluate(bert_similarity)
tfidf_rmse, tfidf_mae = evaluate(tfidf_similarity)


print("\n===== MODEL COMPARISON =====")
print(f"BERT   -> RMSE: {bert_rmse:.4f}, MAE: {bert_mae:.4f}")
print(f"TF-IDF -> RMSE: {tfidf_rmse:.4f}, MAE: {tfidf_mae:.4f}")
print(f"BERT Execution Time: {execution_time:.4f} sec")
print(f"TF-IDF Execution Time: {execution_time1:.4f} sec")


# =========================
# 7. FASTAPI APP
# =========================
app = FastAPI()


# =========================
# 8. RECOMMEND FUNCTION
# =========================
def recommend(student_id, top_n=5, model_type="bert"):
    idx_list = df.index[df["id_etudiant"].astype(str) == str(student_id)].tolist()

    if not idx_list:
        return None

    idx = idx_list[0]

    sim_matrix = bert_similarity if model_type == "bert" else tfidf_similarity

    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for i, score in sim_scores:
        results.append({
            "id_etudiant": str(df.iloc[i]["id_etudiant"]),
            "nom": df.iloc[i]["nom"],
            "similarity": round(float(score), 3)
        })

    return results


# =========================
# 9. ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "AI Student Recommender API 🚀"}


@app.get("/recommend")
def get_recommendations(id: str, top_n: int = 5, model: str = "bert"):
    results = recommend(id, top_n, model)

    if results is None:
        raise HTTPException(status_code=404, detail="Student not found")

    return {
        "student_id": id,
        "model_used": model,
        "recommendations": results
    }


@app.get("/metrics")
def get_metrics():
    return {
        "bert": {
            "rmse": round(float(bert_rmse), 4),
            "mae": round(float(bert_mae), 4),
            "execution_time_sec": round(float(execution_time), 4)
        },
        "tfidf": {
            "rmse": round(float(tfidf_rmse), 4),
            "mae": round(float(tfidf_mae), 4),
            "execution_time_sec": round(float(execution_time1), 4)
        }
    }