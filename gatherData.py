import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle

class GatherData:
    def __init__(self, base_path):
        self.base_path = base_path
        self.users_df, self.books_df, self.ratings_df = self.get_data()
    
    def get_data(self):
        users_df = pd.read_csv(f'{self.base_path}/Users.csv')
        books_df = pd.read_csv(f'{self.base_path}/Books.csv')
        ratings_df = pd.read_csv(f'{self.base_path}/Ratings.csv')

        ratings_df = ratings_df[ratings_df['Book-Rating'] > 0].copy()

        return users_df, books_df, ratings_df

    def preprocess_ratingSet(self):
        user_encoder = LabelEncoder()
        book_encoder = LabelEncoder()

        usr_ids = user_encoder.fit_transform(self.ratings_df['User-ID'])
        book_ids = book_encoder.fit_transform(self.ratings_df['ISBN'])

        utility_matrix = csr_matrix(
            (self.ratings_df['Book-Rating'], (usr_ids, book_ids)),
            shape = (len(user_encoder.classes_), len(book_encoder.classes_))
        )

        return utility_matrix
    
    def preprocess_testSet(self,test_df, user_encoder, book_encoder):   # irrelevant function used during prototyping
        test_df = test_df[
            test_df['User-ID'].isin(user_encoder.classes_) &
            test_df['ISBN'].isin(book_encoder.classes_)
        ].copy()

        # Transform using same encoders
        test_df['user_idx'] = user_encoder.transform(test_df['User-ID'])
        test_df['book_idx'] = book_encoder.transform(test_df['ISBN'])

        return test_df

    def split_data(self,split_ratio = 0.25):

        utility_matrix = self.preprocess_ratingSet()
        utility_matrix, test_df = self.split_util_mat(split_ratio, utility_matrix)

        return utility_matrix, test_df
    
    def split_util_mat(self, split_ratio, utility_matrix):
        np.random.seed(888)

        usr_ids, book_ids = utility_matrix.nonzero()
        usr_ratings = utility_matrix[usr_ids, book_ids].A1

        data = list(zip(usr_ids, book_ids, usr_ratings))

        data = shuffle(data, random_state = 888)
        cutoff = int(len(data) * (1 - split_ratio))
        train_data = data[:cutoff]
        test_data = data[cutoff:]

        usr_ids, book_ids, usr_ratings = zip(*train_data)
        ret_mat = csr_matrix(
            (usr_ratings, (usr_ids, book_ids)),
            shape = utility_matrix.shape
        )

        return ret_mat, test_data
    
    def create_util_mat(self, train_df):    # irrelevant function used during prototyping
        n_users = train_df['user_idx'].nunique()
        n_books = train_df['book_idx'].nunique()

        rating_matrix = csr_matrix(
            (train_df['Book-Rating'], (train_df['user_idx'], train_df['book_idx'])),
            shape=(n_users, n_books)
        )

        return rating_matrix
