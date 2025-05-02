import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix, csr_matrix

class RecSys:
    def __init__(self, k , util_mat, test_df):
        self.k = k
        self.util_mat = util_mat
        self.test_df = test_df
        self.indices, self.distances = self.create_ii_model()
        self.sim_mat = self.create_sim_mat()

    def create_ii_model(self):
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=self.k + 1, n_jobs=-1)
        model_knn.fit(self.util_mat.T)

        # Returns distances and indices of top-k similar items
        distances, indices = model_knn.kneighbors(self.util_mat.T)

        return indices, distances
    
    def create_sim_mat(self):
        # creating similarity matrix
        n_items = self.util_mat.shape[1]
        sim_mat = lil_matrix((n_items, n_items))

        for i in range(n_items):
            for j in range(1, self.k + 1):
                sim_mat[i, self.indices[i,j]] = 1 - self.distances[i,j]

        return sim_mat.tocsr()

    def predict_item_item_rating(self,user_idx, item_idx):
        # calculating predicted rating:
        user_ratings = self.util_mat[user_idx].toarray().ravel()
        rated_items = np.where(user_ratings > 0)[0]
        # print(f"User {user_idx} rated items: {rated_items}")
        if len(rated_items) == 0:
            # print(f"User {user_idx} has not rated any items.")
            global_mean = self.util_mat.data.mean()  # All ratings mean
            return global_mean
            # return 0  # No history available

        # Get similarities between target item and items rated by user
        similarities = self.sim_mat[item_idx, rated_items].toarray().ravel()
        ratings = user_ratings[rated_items]

        # Sort top k similar items
        if len(similarities) > self.k:
            top_k_idx = np.argsort(similarities)[::-1][:self.k]
            similarities = similarities[top_k_idx]
            ratings = ratings[top_k_idx]

        # Compute weighted average
        numerator = np.dot(similarities, ratings)
        denominator = np.sum(np.abs(similarities))

        if denominator == 0:
            # print(f"No similar items found for user {user_idx} and item {item_idx}.")
            if len(rated_items) > 0:
                return np.mean(ratings)  # Use user’s average rating
            else:
                global_mean = self.util_mat.data.mean()  # All ratings mean
                return global_mean
            # return 0  # Avoid divide-by-zero

        return numerator / denominator
    
    def calc_mae(self):
        y_true = []
        y_pred = []

        for tup in self.test_df:
            u_idx = tup[0]
            i_idx = tup[1]
            true_rating = tup[2]
            
            pred_rating = self.predict_item_item_rating(u_idx, i_idx)
            
            if not np.isnan(pred_rating):
                y_true.append(true_rating)
                y_pred.append(pred_rating)

        mae = mean_absolute_error(y_true, y_pred)
        return mae
        # print(f"MAE (Top-K Item-Item CF): {mae:.4f}")
    
