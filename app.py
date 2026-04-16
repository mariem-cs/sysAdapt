import pandas as pd
import unicodedata
import time
import numpy as np
import os
import ast
import re
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, List, Dict, Any


# =========================
# 1. CLEAN TEXT FUNCTION
# =========================
def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text == "" or str(text).lower() in ['nan', 'none']:
        return ""
    text = str(text)

    # Remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # Remove brackets, quotes, and clean up
    text = re.sub(r'[\[\]\'\"]', ' ', text)
    text = re.sub(r',', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def parse_complex_field(text):
    """Parse complex fields that contain lists and nested data"""
    if pd.isna(text) or text == "":
        return ""

    text = str(text)

    # Try to extract meaningful content
    try:
        # Handle list-like strings
        if '[' in text and ']' in text:
            # Extract all quoted strings and numbers
            quoted = re.findall(r"'([^']*)'", text)
            numbers = re.findall(r'\b\d+\b', text)

            all_items = quoted + numbers
            if all_items:
                return " ".join(all_items)

        # Handle comma-separated values
        if ',' in text and not text.startswith('['):
            parts = text.split(',')
            return " ".join(p.strip() for p in parts if p.strip())

    except:
        pass

    return clean_text(text)


# =========================
# 2. LOAD AND CLEAN DATA
# =========================
def load_and_clean_data(csv_path: str):
    """Load CSV and clean the data properly"""
    # Read CSV with proper handling
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    print("Original columns:", df.columns.tolist())
    print(f"Original shape: {df.shape}")

    # Process each column
    for col in df.columns:
        # Convert to string
        df[col] = df[col].astype(str)

        # Parse complex fields
        df[col] = df[col].apply(parse_complex_field)

        # Clean up
        df[col] = df[col].str.replace(r'^nan$', '', regex=True, case=False)
        df[col] = df[col].str.replace(r'^none$', '', regex=True, case=False)
        df[col] = df[col].str.strip()

    # Extract numeric ID from the ID column
    if 'ID_Étudiant' in df.columns:
        # The ID column contains "1,Etudiant_1,8,..." - extract just the number
        df['ID_Étudiant'] = df['ID_Étudiant'].apply(
            lambda x: re.search(r'^(\d+)', str(x)).group(1) if re.search(r'^(\d+)', str(x)) else str(x)
        )

    return df


# Load data
csv_path = r"C:\Users\marie\Downloads\dataset_etudiants.csv"
df = load_and_clean_data(csv_path)

print("\n" + "=" * 50)
print("DATA LOADING SUMMARY")
print("=" * 50)
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"\nFirst 3 rows after cleaning:")
print(df.head(3))
print(f"\nStudent IDs: {df['ID_Étudiant'].head(10).tolist()}")


# =========================
# 3. COLUMN SELECTION
# =========================
def find_col(possible_names):
    """Find column by checking multiple possible names"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


# Define column mappings
column_mappings = {
    "collaborative_work": ["Travaux_Collaboratifs", "travaux_collaboratifs"],
    "teammates": ["Coéquipiers", "coequipiers"],
    "communities": ["Communautés", "communautes"],
    "skills": ["Compétences", "competences"],
    "interests": ["Centres_d'Intérêt", "centres_dinteret"],
    "interactions": ["Nombre_Interactions", "nombre_interactions"]
}

# Get actual column names
cols = []
for key, names in column_mappings.items():
    col = find_col(names)
    if col:
        cols.append(col)
        print(f"✓ Found '{key}' column: '{col}'")
    else:
        print(f"✗ Column not found for '{key}'")

print(f"\nFinal columns used for profiling: {cols}")


# =========================
# 4. BUILD STUDENT PROFILES
# =========================
def build_profile(row, columns):
    """Build a meaningful text profile from selected columns"""
    profile_parts = []

    for col in columns:
        value = str(row[col]).strip()
        if value and value.lower() not in ['', 'nan', 'none']:
            profile_parts.append(value)

    profile_text = " ".join(profile_parts)

    # If still empty, use student ID
    if not profile_text or len(profile_text) < 5:
        profile_text = f"Student_{row.get('ID_Étudiant', 'unknown')}"

    return profile_text


# Build profiles
df["profile"] = df.apply(lambda row: build_profile(row, cols), axis=1)

print(f"\nSample profiles:")
for i in range(min(3, len(df))):
    print(f"\nStudent {i} (ID: {df.iloc[i]['ID_Étudiant']}):")
    print(f"  Profile: {df.iloc[i]['profile'][:200]}")
    print(f"  Length: {len(df.iloc[i]['profile'])} chars")


# =========================
# 5. CREATE GROUND TRUTH
# =========================
def calculate_ground_truth_similarity(row1, row2):
    """Calculate ground truth similarity based on shared content"""

    # Get all text fields for comparison
    text1 = " ".join([str(row1.get(col, "")) for col in cols])
    text2 = " ".join([str(row2.get(col, "")) for col in cols])

    # Split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    if len(words1.union(words2)) > 0:
        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
    else:
        similarity = 0

    return similarity


# Build ground truth similarity matrix
print("\n" + "=" * 50)
print("Building Ground Truth Similarity Matrix...")
print("=" * 50)

n_students = len(df)
ground_truth_similarity = np.zeros((n_students, n_students))

for i in range(n_students):
    for j in range(i + 1, n_students):
        sim = calculate_ground_truth_similarity(df.iloc[i], df.iloc[j])
        ground_truth_similarity[i][j] = sim
        ground_truth_similarity[j][i] = sim

print(f"✓ Ground truth matrix created")
print(f"  Shape: {ground_truth_similarity.shape}")
print(f"  Range: [{ground_truth_similarity.min():.4f}, {ground_truth_similarity.max():.4f}]")
print(f"  Mean: {ground_truth_similarity.mean():.4f}")

# =========================
# 6. TF-IDF MODEL
# =========================
print("\n" + "=" * 50)
print("Running TF-IDF Model...")
print("=" * 50)

start_time_tfidf = time.time()

try:
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        token_pattern=r'(?u)\b\w+\b',
        stop_words=None
    )
    X_tfidf = vectorizer.fit_transform(df["profile"])
    tfidf_similarity = cosine_similarity(X_tfidf)
    tfidf_success = True
    print(f"✓ TF-IDF completed")
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"  Sample similarity (0 vs 1): {tfidf_similarity[0][1]:.4f}")
except Exception as e:
    print(f"✗ TF-IDF failed: {e}")
    tfidf_similarity = np.eye(len(df))
    tfidf_success = False

execution_time_tfidf = time.time() - start_time_tfidf
print(f"  Execution Time: {execution_time_tfidf:.4f} sec")

# =========================
# 7. BERT MODEL
# =========================
print("\n" + "=" * 50)
print("Loading BERT Model...")
print("=" * 50)

start_time_bert = time.time()

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df["profile"].tolist(), show_progress_bar=True)
    bert_similarity = cosine_similarity(embeddings)
    bert_success = True
    print(f"✓ BERT completed")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Sample similarity (0 vs 1): {bert_similarity[0][1]:.4f}")
except Exception as e:
    print(f"✗ BERT failed: {e}")
    bert_similarity = np.eye(len(df))
    bert_success = False

execution_time_bert = time.time() - start_time_bert
print(f"  Execution Time: {execution_time_bert:.4f} sec")


# =========================
# 8. EVALUATION WITH RMSE AND MAE
# =========================
def calculate_rmse_mae(predicted_sim, ground_truth_sim):
    """Calculate RMSE and MAE between predicted and ground truth"""

    # Extract upper triangular values
    y_true = []
    y_pred = []

    n = len(predicted_sim)
    for i in range(n):
        for j in range(i + 1, n):
            y_true.append(ground_truth_sim[i][j])
            y_pred.append(predicted_sim[i][j])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate correlation (handle NaN case)
    try:
        correlation = np.corrcoef(y_true, y_pred)[0][1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0

    return {
        "rmse": float(round(rmse, 4)),
        "mae": float(round(mae, 4)),
        "correlation": float(round(correlation, 4)),
        "samples_compared": len(y_true)
    }


print("\n" + "=" * 50)
print("Evaluating Models with RMSE & MAE...")
print("=" * 50)

bert_metrics = calculate_rmse_mae(bert_similarity, ground_truth_similarity)
tfidf_metrics = calculate_rmse_mae(tfidf_similarity, ground_truth_similarity)

print(f"\n📊 BERT Model Performance:")
print(f"   ✅ RMSE: {bert_metrics['rmse']:.4f}")
print(f"   ✅ MAE: {bert_metrics['mae']:.4f}")
print(f"   📊 Compared {bert_metrics['samples_compared']} pairs")

print(f"\n📊 TF-IDF Model Performance:")
print(f"   ✅ RMSE: {tfidf_metrics['rmse']:.4f}")
print(f"   ✅ MAE: {tfidf_metrics['mae']:.4f}")
print(f"   📊 Compared {tfidf_metrics['samples_compared']} pairs")

# =========================
# 9. FASTAPI APP
# =========================
app = FastAPI(
    title="Student Recommender API",
    description="AI-powered student recommendation system using BERT and TF-IDF",
    version="1.0.0"
)


# =========================
# 10. RECOMMENDATION FUNCTION
# =========================
def recommend(student_id: str, top_n: int = 5, model_type: str = "bert"):
    """Get top N similar students"""

    try:
        # Clean ID for comparison
        df['ID_Étudiant'] = df['ID_Étudiant'].astype(str).str.strip()
        search_id = str(student_id).strip()

        # Try direct match
        idx_list = df.index[df['ID_Étudiant'] == search_id].tolist()

        if not idx_list:
            return None

        idx = idx_list[0]

    except Exception as e:
        print(f"Error: {e}")
        return None

    # Select similarity matrix
    if model_type == "bert" and bert_success:
        sim_matrix = bert_similarity
    elif model_type == "tfidf" and tfidf_success:
        sim_matrix = tfidf_similarity
    else:
        sim_matrix = bert_similarity if bert_success else tfidf_similarity

    # Get similarity scores
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    # Build results
    results = []
    for i, score in sim_scores:
        result = {
            "similarity_score": round(float(score), 4),
            "student_id": str(df.iloc[i]['ID_Étudiant']),
            "profile_preview": df.iloc[i]['profile'][:100] + "..." if len(df.iloc[i]['profile']) > 100 else df.iloc[i][
                'profile']
        }
        results.append(result)

    return results


# =========================
# 11. API ENDPOINTS
# =========================
@app.get("/")
def home():
    return {
        "message": " AI Student Recommender API",
        "metrics_available": ["RMSE", "MAE"],
        "endpoints": {
            "/recommend": "GET - Get student recommendations",
            "/metrics": "GET - View RMSE & MAE metrics",
            "/students": "GET - List all students",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "bert_model_loaded": bert_success,
        "tfidf_model_loaded": tfidf_success,
        "total_students": len(df),
        "sample_student_ids": df['ID_Étudiant'].head(10).tolist()
    }


@app.get("/students")
def list_students(limit: int = Query(10, ge=1, le=100)):
    """List available students"""
    students = []
    for idx in range(min(limit, len(df))):
        students.append({
            "index": idx,
            "student_id": str(df.iloc[idx]['ID_Étudiant']),
            "profile_length": len(df.iloc[idx]['profile'])
        })

    return {
        "total_students": len(df),
        "students": students
    }


@app.get("/recommend")
def get_recommendations(
        student_id: str = Query(..., description="Student ID"),
        top_n: int = Query(5, ge=1, le=20),
        model: str = Query("bert", pattern="^(bert|tfidf)$")
):
    """Get student recommendations"""

    results = recommend(student_id, top_n, model)

    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"Student ID '{student_id}' not found"
        )

    return {
        "student_id": student_id,
        "model_used": model,
        "total_recommendations": len(results),
        "recommendations": results
    }


@app.get("/metrics")
def get_metrics():
    """Get RMSE, MAE and other performance metrics"""
    return {
        "bert": {
            "status": "loaded" if bert_success else "failed",
            "rmse": bert_metrics['rmse'],
            "mae": bert_metrics['mae'],
            "execution_time_sec": round(execution_time_bert, 4)
        },
        "tfidf": {
            "status": "loaded" if tfidf_success else "failed",
            "rmse": tfidf_metrics['rmse'],
            "mae": tfidf_metrics['mae'],
            "execution_time_sec": round(execution_time_tfidf, 4)
        },
        "dataset_info": {
            "total_students": len(df),
            "columns_used": cols,
        }
    }

# =========================
# 12. RUN THE APP
# =========================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🚀 Starting Student Recommender API")
    print("=" * 50)
    print(f"📊 Loaded {len(df)} students")
    print(f"📈 BERT - RMSE: {bert_metrics['rmse']:.4f}, MAE: {bert_metrics['mae']:.4f}")
    print(f"📈 TF-IDF - RMSE: {tfidf_metrics['rmse']:.4f}, MAE: {tfidf_metrics['mae']:.4f}")
    print(f"📍 API: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)