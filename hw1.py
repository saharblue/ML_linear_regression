###### Your ID ######
# ID1: 316061456
# ID2: 207639881
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    return mean_normalization(X), mean_normalization(y)


def mean_normalization(vector):
    """
    Perform mean normalization on provided vector.

    Input:
    - vector: vector of numbers.

    Returns:
    - X: The mean normalized vector.
    """
    vector = (vector - np.mean(vector, axis=0))/(np.max(vector, axis=0) - np.min(vector, axis=0))
    return vector


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    try:
        X.shape[1]
    except IndexError:
        X = X.reshape((-1, 1))
    return np.hstack((np.ones((X.shape[0], 1)), X))


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    J = 0  # We use J for the cost.
    # m = number of instances
    m = X.shape[0]
    # calculate h-theta(X)
    product = X.dot(theta)
    # calculate J(theta)
    product = (product - y) ** 2
    J = (1 / (2 * m)) * product.sum()
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    # m = number of instances
    m = X.shape[0]
    coefficient = alpha / m
    for _ in range(num_iters):
        # calculate h-theta(X)
        product = X.dot(theta)
        theta = theta - coefficient * ((product - y).dot(X))
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    # calculate x^T
    X_trans = np.transpose(X)
    # calculate pinv(X) * y
    pinv_theta = np.linalg.inv(X_trans.dot(X)).dot(X_trans).dot(y)
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    # m = number of instances
    m = X.shape[0]
    coefficient = alpha / m
    last_cost = compute_cost(X, y, theta) + 1 # an initial value big enough to enter the while loop
    new_cost = compute_cost(X, y, theta)
    # counter for not exceeding the number of iterations
    counter = 0
    while last_cost - new_cost >= 1e-8:
        # calculate h-theta(X)
        product = X.dot(theta)
        theta = theta - coefficient * ((product - y).dot(X))
        last_cost = new_cost
        new_cost = compute_cost(X, y, theta)
        J_history.append(new_cost)
        counter += 1
        # if number of iterations exceeded, exit while loop
        if counter > num_iters:
            break
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    # choose an initial random theta
    np.random.seed(42)
    theta = np.random.random(size=X_train.shape[1])
    for alpha in alphas:
        new_theta, _ = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        loss_value = compute_cost(X_val, y_val, new_theta)
        alpha_dict[alpha] = loss_value
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    # calculate no. of features
    try:
        n = X_train.shape[1]
    except IndexError:
        n = 1

    not_selected_features = list(range(1, n))

    # applying bias trick
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)

    for j in range(5):
        cost_track = {}
        # create a tuple with already selected features for training model
        feature_indices = tuple([0]) + tuple(selected_features)
        # create an initial theta for training the model
        np.random.seed(42)
        theta = np.random.random(size=len(feature_indices) + 1)

        for feature in not_selected_features:
            # add current feature to already existing ones
            all_features = feature_indices + tuple([feature])
            new_theta, _ = efficient_gradient_descent(X_train[:, all_features], y_train, theta,
                                                      best_alpha, iterations)
            # add the cost as value and the feature as key for cost track dictionary
            cost_track[feature] = compute_cost(X_val[:, all_features], y_val, new_theta)

        # check which feature generated the lowest cost, add it to the selected features list
        # and remove it from the non-selected features list
        best_feature = min(cost_track, key=cost_track.get)
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    return np.array(selected_features) - 1


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """
    df_poly = df.copy()
    for i in range(df.shape[1]):
        for j in range(i, df.shape[1]):
            first_column = df.iloc[:, i]
            second_column = df.iloc[:, j]
            if first_column.name == second_column.name:
                df_poly[f"{first_column.name}^2"] = first_column.multiply(second_column)
            else:
                df_poly[f"{first_column.name}*{second_column.name}"] = first_column.multiply(second_column)

    return df_poly
