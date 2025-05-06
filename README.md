# 📚 Book Recommendation System using Item-Item Collaborative Filtering

This project implements an item-item collaborative filtering system to predict user ratings for books. The dataset used for the project consists of user information, rating information and book information.
Ratings dataset contains information about the user, book rated and rating value. The rating value ranges from 0 - 10, where 0 signifies no rating. 

### Algorithm: Item-Item Collaborative Filtering

A classical algorithm of implementing recommendation systems, this algorithm applies the implicit approach of gathering information used for learning user behaviour and recommending. 
This algorithm is good at making personalized recommendations without requiring extensive information about an individual user. The algorithm learns from similar users and recommends based on that information.
A neighbourhood (a set of similar users/items) is taken into consideration, similarity of current (test) user with all users in neighborhood is multiplied with neighborhood ratings and divided by sum of similarities.
However, in rating systems for inidividual user the intensity of rating varies, so to remove the user bias, each user information is scaled using the mean. 

Eventually, we get an algorithm which gives unique, personalised suggestions without needing extensive information and not plagued with other user bias.

### Problems Faced:

The entire dataset contained over a million data points out of which multiple were no rating values, hence creating a utility matrix representing this information would result in an extremely
huge matrix with over a billion cells and impossible to store in the memory or any further computation. 

1. The first step taken to manage this problem, during preprocessing only datapoints with a non-zero rating is considered for utility matrix creation as no rating values are of no help to us.
   
2. Secondly, the utility matrix created is stored as a sparse matrix using the scipy csr library to reduce memory consumption.

Once the dataset was ready for processing, the next step was splitting the data for further processing. Two approaches were followed both resulting in significant differences in mae values.

1. First approach: The ratings dataset was first split into training and testing. The training set was then used to create a utility matrix and fit into the knn to find similarity matrix.
   The encoders used for isbn and users were sent back to the calling function to be used on the testing set before making predictions.

   - Problems with this approach: The mae calculated was 2.79 for k = 5 and 3.3 for k = 50. Apart from the high mae values the computation time was significantly higher.

2. Second approach: The entire rating dataset was used to create the utility matrix and then the utility matrix was split into training and testing set. This approach bundled the matrix computations in a single step
   and reducing forward and calling of matrix values which were possibly causing extra computation and time consumption.

Finally during prediction stage: After the model was fit with the data and during predictions, there were two different choices regarding handling zero ratings which caused significant differences in mae.

1. First approach: Only non-zero ratings are being considred which caused the mae to explode to 5.10 for k = 5 during testing stage.
 
2. Second approach (solution): The zero values were imputated using the mean rating value of that user which helped in alleviating the exploding mae values.

### Future Optimization:

Future expansion of this project can be using a big data handling framework such as pyspark which can help in handling the data and applying machine learning algorithms to make predictions in 
considerably reudced time. 

Another Approach that can be done (altough might be computationally expensive): for an item to be recommended, find only the users who have rating that particular item. The utility and similarity matrix should be made only of those users.
Possible problems with this approach could be that for multiple recommendations, each step will require a matrix creation which is a very expensive operation and hence might not make it feasible.


## 📁 Project Structure

|- /data                  Contains {users,ratings,book}.csv (not included in the repository)

|─ gatherData.py           Class for loading and preprocessing dataset

    |- get_data              gets data from csv files

    |- preprocess_ratingSet  creates utility matrix of the ratings dataset

    |- split_data            calling preprocess_ratingSet to get utility_matrix then passes to splitting function to get train and test

    |- split_util_mat        splits the utility matrix into training (matrix) and testing (list -> tuple(user_id, book_id, true_rating))

|─ recommender.py          Class for model building, prediction, and evaluation

    |- create_ii_model     initializes the knn and fits it with the data to get indices and distances

    |- predict_item_item_rating    called by the calc_mae function to make prediction using the neighborhood items

    |- calc_mae            gets prediction, true values, finds the mae and returns the mean error value

|─ data_exploration.ipynb  Notebook for prototyping and testing components

|─ tester.ipynb            End-to-end execution and result verification

|─ README.md               Project documentation



## 🚀 Features

- Item-Item Collaborative Filtering with top-K neighbor approach
- Scalable similarity computation using `NearestNeighbors`
- Sparse utility matrix construction with `scipy.sparse`
- Evaluation using Mean Absolute Error (MAE)
- Analysis of performance for:
  - Varying neighborhood sizes (`k = 5, 10, 15, 20, 50, 100`)
  - Varying training sizes with 5% increment steps for optimal k value (`60%–90%`)



## 🧠 Methods

- **Similarity**: Cosine similarity computed via `NearestNeighbors` to avoid full similarity matrix explosion.
- **Prediction**:

  <img width="548" alt="image" src="https://github.com/user-attachments/assets/ab758232-12fe-4f34-adab-2396b627cd44" />

- **Evaluation**: Mean Absolute Error (MAE) between predicted and actual ratings.



## 🛠️ Setup & Dependencies

Make sure to install the required libraries:


```pip install pandas numpy scikit-learn scipy```



## 🧪 How to Run

1. **Run the complete pipeline**:

   **Warning: Make sure to restart the kernel before proceeding, the entire operation is over 2 hours long of computation time**

   * Execute `tester.ipynb` to:

     * Load and preprocess data
     * Build utility matrix
     * Train the item-item similarity model
     * Predict ratings for test data
     * Evaluate MAE for multiple `k` values


## 📊 Results

| k (neighbors) | MAE  |
| ------------- | ---- |
| 5             | 1.33 |
| 10            | 1.30 |
| 15            | 1.29 |
| 20            | 1.29 |
| 50            | 1.28 |
| 100           | 1.28 |

## 📬 Contact

For any further questions, feel free to reach out! Happy coding!
