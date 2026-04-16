import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD DATA

df = pd.read_csv(r"C:\Users\marie\Downloads\dataset_etudiants.csv")

df.columns = df.columns.str.strip()
df.fillna("", inplace=True)

print("Columns in dataset:")
print(df.columns.tolist())

# 2.  COLUMN SELECTION
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None
cols = []

cols.append(find_col(["Travaux_Collaboratifs", "Travaux_Collaboratifs", "Travaux_Collaboratifs"]))
cols.append(find_col(["Coéquipiers", "Coéquipiers", "Coéquipiers"]))
cols.append(find_col(["Communautés", "Communautes"]))
cols.append(find_col(["Compétences", "Compétences"]))
cols.append(find_col(["Centres_d'Intérêt", "Centres_d'Intérêt", "Centres_d'Intérêt", "Centres_d'Intérêt"]))

# remove None values
cols = [c for c in cols if c is not None]

print("\nUsed columns:", cols)

# 3. BUILD PROFILE

df["profile"] = df[cols].astype(str).agg(" ".join, axis=1)

# 4. TF-IDF + SIMILARITY

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["profile"])

similarity = cosine_similarity(X)

# 5. RECOMMENDER

def recommend_students(student_id, top_n=5):
    idx_list = df.index[df["ID_Étudiant"] == student_id].tolist()

    if not idx_list:
        return "Student not found"

    idx = idx_list[0]

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for i, score in sim_scores:
        results.append({
            "ID_Étudiant": df.iloc[i]["ID_Étudiant"],
            "Nom": df.iloc[i]["Nom"],
            "Similarity": round(score, 3)
        })

    return pd.DataFrame(results)

# 6. TEST
print("\nExample recommendation:")
print(recommend_students(df["ID_Étudiant"].iloc[0]))