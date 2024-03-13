from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import pickle
from pickle import dump

from kneed import KneeLocator

def createAndSaveScaler(data):

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
    return pickle.load(open(lat,'rb')), pickle.load(open(long, 'rb')), pickle.load(open(log_price, 'rb'))

def one_hot_encoder(data='Airbnb_Data.csv', category_target=None):
    
    df = pd.read_csv(data)
    if category_target == None:
        print("No category to be one hot encoding")
        return

    encoder = OneHotEncoder(drop='first')
    transform = encoder.fit_transform(df[category_target])
    df_encoded = pd.DataFrame(transform.toarray(), columns=encoder.get_feature_names_out(category_target))

    return df_encoded


def concat_data(df, df_to_concat, category_target=None):
    
    if category_target == None:
        print("No category target")
        return
    
    df = pd.concat([df, df_to_concat], axis=1)
    df.drop(columns=category_target, inplace=True)

    return df

def find_optimal_cluster(df, max_cluster=10, target_cluster=None):
    '''
    Clustering for [['latitude', 'longitude']] or ['property_type']
    '''
    if target_cluster is None or df[target_cluster].empty: return

    data = df[target_cluster]

    sum_squared_error = []

    # print(target_cluster)
    k = range(1, max_cluster+1)
    if target_cluster == ['property_type']:
        data = one_hot_encoder(category_target=['property_type'])
    
    
    # Insert into SSE
    for i in k:
        cluster = KMeans(n_clusters=i, init='k-means++', max_iter=500, random_state=42).fit(data)
        sum_squared_error.append(cluster.inertia_)
    
    # Plot to know the direction and curve
    plt.plot(k, sum_squared_error, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal Cluster Number')
    plt.show()

    # After plotted, direction="decreasing" and curve="convex"
    elbow_locator = KneeLocator(x=k, y=sum_squared_error, direction="decreasing", curve="convex")
    elbow_point = elbow_locator.knee - 1
    # print(elbow_point)
    return elbow_point
    
    




