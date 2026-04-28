#  AI Student Recommender System

An intelligent student recommendation system based on Machine Learning and NLP, designed for e-learning environments.
The system analyzes student profiles and suggests similar students to improve collaboration and personalized learning.

---

##  Features

*  Student similarity recommendation
*  Hybrid models:

  * TF-IDF (classical NLP)
  * BERT (semantic understanding)
*  Evaluation metrics:

  * RMSE
  * MAE
*  REST API built with FastAPI
*  Interactive web interface (`/ui`)
*  Docker support
*  Ready for Moodle integration (API-based)

---

##  How It Works

1. Load and clean student dataset
2. Text preprocessing (normalization, parsing)
3. Build student profiles
4. Apply models:

   * TF-IDF vectorization
   * BERT embeddings
5. Compute similarity (Cosine Similarity)
6. Serve recommendations via API

---

##  Architecture

```text
Moodle (or UI)
        ↓
FastAPI Backend
        ↓
ML Models (TF-IDF / BERT)
        ↓
Similarity Engine
        ↓
Recommendations (JSON)
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/student-recommender.git
cd student-recommender
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the application

```bash
uvicorn app:app --reload
```

---

##  API Endpoints

| Endpoint     | Description         |
| ------------ | ------------------- |
| `/`          | API info            |
| `/health`    | Health check        |
| `/students`  | List students       |
| `/recommend` | Get recommendations |
| `/metrics`   | Model evaluation    |
| `/ui`        | Web interface       |

---

###  Example Request

```bash
GET /recommend?student_id=1&model=bert&top_n=5
```

---

###  Example Response

```json
{
  "student_id": "1",
  "recommendations": [
    {
      "student_id": "3",
      "similarity_score": 0.91
    }
  ]
}
```

---

##  Web Interface

Access the UI at:

```text
http://localhost:8000/ui
```

Features:

* Search by student ID
* Choose model (BERT / TF-IDF)
* View similarity scores
* Clean dashboard interface

---

##  Docker

Build and run:

```bash
docker build -t student-recommender .
docker run -p 8000:8000 student-recommender
```

---

## 🔌 Moodle Integration (Concept)

Moodle can interact with the system via API:

```php
$response = file_get_contents("http://localhost:8000/recommend?student_id=1");
$data = json_decode($response, true);
```

---

##  Evaluation

| Model  | RMSE | MAE |
| ------ | ---- | --- |
| BERT   | ✔    | ✔   |
| TF-IDF | ✔    | ✔   |

---

##  Use Cases

*  Suggest teammates for group work
*  Group students with similar interests
*  Enhance collaborative learning
*  Personalize learning experiences

---

##  Challenges

* Data cleaning and preprocessing
* TF-IDF limitations with sparse data
* Integration between PHP (Moodle) and Python (FastAPI)
* Model performance optimization

---

##  Tech Stack

* Python
* FastAPI
* Scikit-learn
* SentenceTransformers (BERT)
* Pandas / NumPy
* Docker
* HTML / JS (UI)

---

##  Author

Mariem Abid
Research Master in Computer Science student

---

##  Future Improvements

*  Secure API (authentication)
*  Real Moodle plugin integration


##  Acknowledgment

This project was developed as part of an academic project focused on Recommandation systems in education.

---
