# Hybrid Movie Recommendation System

A hybrid movie recommendation system that combines **Content-Based Features** and **Collaborative Filtering Latent Features** using the **Feature Combination (Concatenation) Technique**.

> **Project objective:** build ONE hybrid recommendation model. TF-IDF and SVD are used to extract features; the final recommendations are generated from the combined hybrid representation.

## 1. Project Overview

The system represents each movie using two feature groups:

- **Content features:** TF-IDF vectors extracted from the movie genres in MovieLens metadata.
- **Collaborative features:** latent item vectors learned by SVD from user–movie ratings.

The two vectors are normalized, weighted, and concatenated:

\[
H_i = [\alpha C_i \; || \; \beta F_i]
\]

where:

- `C_i` is the content feature vector for movie `i`.
- `F_i` is the collaborative latent feature vector for movie `i`.
- `alpha` and `beta` control the contribution of each feature group.
- `||` denotes vector concatenation.
- `H_i` is the final hybrid item representation.

A user profile is then built from the hybrid vectors of movies the user has rated. Cosine similarity ranks candidate movies, previously rated movies are excluded, and the system returns Top-N recommendations.

## 2. Dataset

**Dataset:** [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/), published by GroupLens.

The project uses these files:

| File          | Main columns                               | Purpose                             |
| ------------- | ------------------------------------------ | ----------------------------------- |
| `movies.csv`  | `movieId`, `title`, `genres`               | Movie metadata and content features |
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | User–movie interactions and ratings |

The notebook reads the `userId`, `movieId`, and `rating` columns for the recommendation workflow. Genres are used as the content metadata in the current implementation.

### Dataset handling

The MovieLens 32M archive is large. **Do not commit the CSV files or ZIP archive to this GitHub repository.** Download the dataset from GroupLens and place/extract it in the Colab runtime as described in `data/README.md`. The `.gitignore` excludes local dataset files.

## 3. Project Workflow

The implementation follows the five tasks in the project specification.

### Task 1 — Dataset Collection

- Select MovieLens 32M.
- Load `movies.csv` and `ratings.csv` into Pandas DataFrames.
- Inspect sample rows, shapes, users, and movies.

### Task 2 — Content Feature Extraction

- Prepare movie genre text.
- Convert genre text into numerical vectors using `TfidfVectorizer`.
- Store the content feature matrix and feature names.

### Task 3 — Collaborative Feature Extraction

- Construct a user–item interaction matrix.
- Train an SVD model using user ratings.
- Extract user and item latent factors (`pu` and `qi`).
- In the current Colab experiment, a reproducible sample of 1,000,000 ratings is used to keep training practical.

### Task 4 — Feature Combination

- Match movies by `movieId` so content and collaborative rows refer to the same movie.
- Normalize both feature groups.
- Apply weights and concatenate the vectors to create the hybrid matrix.

### Task 5 — Recommendation Generation

- Build a user profile from the hybrid vectors of the user's rated movies.
- Calculate cosine similarity between the user profile and candidate movie vectors.
- Exclude movies the user has already rated.
- Return Top-N recommendations and demonstrate the results for at least three users.

## 4. Repository Structure

```text
hybrid-recommendation-system/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Hybrid_Recommendation_System.ipynb
│   └── README.md
├── data/
│   └── README.md
│   └── movies.csv
│   └── ratings.csv
├── results/
│   └── README.md
└── docs/
    └── PROJECT_NOTES.md
```

## 5. How to Run in Google Colab

1. Download `ml-32m.zip` from the [official MovieLens 32M page](https://grouplens.org/datasets/movielens/32m/).
2. Upload/extract the archive in the Colab runtime, or use the download instructions in `data/README.md`.
3. Open `notebooks/Hybrid_Recommendation_System.ipynb` in Google Colab.
4. Run the notebook cells in order: Task 1 → Task 2 → Task 3 → Task 4 → Task 5.
5. Keep the runtime active between tasks because later cells reuse variables created earlier.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

In Google Colab, install `scikit-surprise` in a cell if it is not already available:

```python
!pip install scikit-surprise -q
```

## 6. Current Experiment Configuration

The best observed SVD configuration in the notebook experiments so far is:

| Parameter          |     Value |
| ------------------ | --------: |
| Rating sample size | 1,000,000 |
| `n_factors`        |        50 |
| `n_epochs`         |        20 |
| `lr_all`           |     0.005 |
| `reg_all`          |      0.05 |
| `random_state`     |        42 |

The recorded comparison result for this configuration was **RMSE = 0.9062** and **MAE = 0.6954** on the current 80/20 split.

**Evaluation note:** the same held-out split was consulted while comparing configurations. Therefore, these values are preliminary experiment results, not an untouched final test score. For a rigorous final evaluation, tune hyperparameters using a validation split and reserve a separate test split for one final evaluation.

The hybrid feature dimensions observed in the current run were:

- 20 TF-IDF content features.
- 50 SVD item latent features.
- 70 combined hybrid features per matched movie.

The actual number of matched movies depends on which movies appear in the SVD training data.

## 7. Deliverables Checklist

- [x] Selected dataset and dataset description documented.
- [ ] Complete source notebook added to `notebooks/`.
- [ ] Practical demonstration exported for at least three different users.
- [ ] Final results and screenshots added to `results/`.

## 8. Notes

- This repository implements one hybrid recommendation system, not two independent recommendation engines.
- Content-Based and Collaborative Filtering are used to extract features for the hybrid model.
- Feature combination is a mandatory part of the implementation.
- The full MovieLens data files are intentionally excluded from GitHub because of their size; download them from GroupLens instead.

## 9. References

- [MovieLens 32M Dataset — GroupLens](https://grouplens.org/datasets/movielens/32m/)
- [Surprise documentation](https://surpriselib.com/)
- [scikit-learn TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn cosine similarity documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
