from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import pickle
from pickle import dump

import seaborn as sns

from kneed import KneeLocator
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def createAndSaveScaler(data):
    '''
    type data: pd.DataFrame
    rtype data: pd.DataFrame

    Normalization for ['latitude', 'longitude', 'log_price']
    '''
    # Fit Scaler
    lat_scaler = StandardScaler().fit(data[['latitude']])
    long_scaler = StandardScaler().fit(data[['longitude']])
    log_price_scaler = MinMaxScaler().fit(data[['log_price']])

    # Transform
    data['latitude'] = lat_scaler.transform(data[['latitude']])
    data['longitude'] = long_scaler.transform(data[['longitude']])
    data['log_price'] = log_price_scaler.transform(data[['log_price']])

    # Save
    dump(lat_scaler, open('models/lat_scaler.pkl', 'wb'))
    dump(long_scaler, open('models/long_scaler.pkl', 'wb'))
    dump(log_price_scaler, open('models/log_price_scaler.pkl', 'wb'))

    return data

def loadScaler(lat, long, log_price):
    '''
    type lat: str
    type long: str
    type log_price: str
    rtype 
    '''
    return pickle.load(open(lat,'rb')), pickle.load(open(long, 'rb')), pickle.load(open(log_price, 'rb'))

def one_hot_encoder(data='Airbnb_Data.csv', category_target=None):
    '''
    type data: str
    type category_target: List[str]
    rtype df_encoded: pd.DataFrame
    '''
    if isinstance(data, str): df = pd.read_csv(data)
    else: df = data

    if category_target == None:
        print("No category to be one hot encoding")
        return

    encoder = OneHotEncoder(drop='first')
    transform = encoder.fit_transform(df[category_target])
    df_encoded = pd.DataFrame(transform.toarray(), columns=encoder.get_feature_names_out(category_target))

    return df_encoded


def concat_data(df, df_to_concat, category_target=None):
    '''
    type df: pd.DataFrame
    type df_to_concat: pd.DataFrame
    type category_taret: List[str]
    rtype df: pd.DataFrame

    '''
    if category_target == None:
        print("No category target")
        return
    
    df = pd.concat([df, df_to_concat], axis=1)
    df.drop(columns=category_target, inplace=True)
    
    return df


def find_optimal_cluster(df, max_cluster=10, target_cluster=None):
    '''
    type df: pd.DataFrame
    type max_cluster: int
    type target_cluster: List[str]
    rtype elbow_point: int

    Clustering for [['latitude', 'longitude']] or ['property_type']
    '''
    if target_cluster is None or df[target_cluster].empty: return

    data = df[target_cluster]
    sum_squared_error = []

    # print(target_cluster)
    k = range(1, max_cluster+1)
    
    # Insert into SSE
    for i in k:
        cluster = KMeans(n_clusters=i, init='k-means++', max_iter=500, random_state=42).fit(data)
        sum_squared_error.append(cluster.inertia_)
    
    # Plot to know the direction and curve
    # plt.plot(k, sum_squared_error, marker='o')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Sum of Squared Errors (SSE)')
    # plt.title('Elbow Method for Optimal Cluster Number')
    # plt.show()

    # After plotted, direction="decreasing" and curve="convex"
    elbow_locator = KneeLocator(x=k, y=sum_squared_error, direction="decreasing", curve="convex")
    elbow_point = elbow_locator.knee - 1
    # print(elbow_point)

    return elbow_point


def createLinearRegression(X_train, y_train, X_test, y_test):
    '''
    type X_train: pd.DataFrame
    type y_train: pd.DataFrame
    type X_test: pd.DataFrame
    type y_test: pd.DataFrame
    rtype: None

    Predict the data using Linear Regression
    '''
    model = LinearRegression().fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index=X_train.index)
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    train_MSE = mean_squared_error(y_train, y_fit)
    test_MSE = mean_squared_error(y_test, y_pred)

    def plot(y_test, y_pred, name=None):
        plt.scatter(X_test.index, y_pred, color='skyblue', label='Predictions')
        plt.scatter(X_test.index, y_test, color='red', label='Actual', alpha=0.2)
        plt.title("Linear Regression")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.show()
        return


    plot(y_test, y_pred, "Linear Regression")
    print("Linear Regression - Train Score: ", train_MSE)
    print("Linear Regression - Test Score: ", test_MSE, "\n")

    dump(model, open('models/LinearRegression.pkl', 'wb'))


def createDecisionTree(X_train, y_train, X_test, y_test):
    '''
    type X_train: pd.DataFrame
    type y_train: pd.DataFrame
    type X_test: pd.DataFrame
    type y_test: pd.DataFrame
    rtype: None

    Predict data using Decision Tree Regressor
    '''
    model = DecisionTreeRegressor()

    grid_search = GridSearchCV(estimator=model, param_grid={'max_features': ['sqrt', 'log2']})

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_fit = pd.Series(best_model.predict(X_train), index=X_train.index)
    y_pred = pd.Series(best_model.predict(X_test), index=X_test.index)

    train_MSE = mean_absolute_error(y_train, y_fit)
    test_MSE = mean_absolute_error(y_test, y_pred)

    print("Decision Tree - Train Score: ", train_MSE)
    print("Decision Tree - Test Score: ", test_MSE, "\n")

    dump(model, open('models/DecisionTreeRegressor.pkl', 'wb'))


def createSVR(X_train, y_train, X_test, y_test):
    '''
    type X_train: pd.DataFrame
    type y_train: pd.DataFrame
    type X_test: pd.DataFrame
    type y_test: pd.DataFrame
    rtype: None

    Predict data using Support Vector Machine
    '''
    model = SVR(kernel='linear').fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index=X_train.index)
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    train_MSE = mean_squared_error(y_train, y_fit)
    test_MSE = mean_squared_error(y_test, y_pred)
    
    print("SVR - Train Score: ", train_MSE)
    print("SVR - Test Score: ", test_MSE)

    
    dump(model, open('models/SVR.pkl', 'wb'))








    
    




