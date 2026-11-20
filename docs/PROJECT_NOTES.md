# Project Notes

## Dataset

- Name: MovieLens 32M
- Source: GroupLens
- Content metadata used: `genres` from `movies.csv`
- Interaction data used: `userId`, `movieId`, and `rating` from `ratings.csv`

## Method

1. TF-IDF converts movie genre text into content vectors.
2. SVD learns user and item latent factors from the user–movie ratings.
3. Matching `movieId` values align the two item feature groups.
4. The normalized, weighted feature groups are concatenated into one hybrid item representation.
5. A weighted user profile and cosine similarity are used to rank candidate movies, excluding items already rated by the user.

## Current experimental settings

- Rating sample: 1,000,000 rows (`random_state=42`)
- SVD: 50 factors, 20 epochs, learning rate 0.005, regularization 0.05
- Recorded RMSE: 0.9062
- Recorded MAE: 0.6954
- Hybrid dimensions: 20 content + 50 collaborative = 70 features per matched movie

These evaluation values are preliminary because the held-out split was consulted during hyperparameter comparisons. Use a validation set for tuning and an untouched test set for final reporting.
