from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import time
import pandas as pd
from surprise import KNNWithMeans, SVD

# Charger dataset

# Charger CSV
df = pd.read_csv(r"C:\Users\marie\Downloads\dataset_etudiants.csv")
# Définir format
reader = Reader(rating_scale=(1, 5))

# Convertir en dataset Surprise
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split
trainset, testset = train_test_split(data, test_size=0.2)

# Algo KNN
algo = KNNBasic()

# Train
start = time.time()
algo.fit(trainset)
train_time = time.time() - start

# Test
predictions = algo.test(testset)

# Evaluation
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print("Temps:", train_time)
print("RMSE:", rmse)
print("MAE:", mae)

algos = {
    "KNNBasic": KNNBasic(),
    "KNNWithMeans": KNNWithMeans(),
    "SVD": SVD()
}

for name, algo in algos.items():
    algo.fit(trainset)
    preds = algo.test(testset)
    print(name)
    accuracy.rmse(preds)