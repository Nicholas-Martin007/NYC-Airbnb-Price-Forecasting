import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from model import concat_data, createAndSaveScaler, find_optimal_cluster, loadScaler, one_hot_encoder

from pickle import dump

def fill_nan(df, columns, strategy='mean'):
    '''
    Using SimpleImputer for filling missing values
    '''
    imputer = SimpleImputer(strategy=strategy)

    for col in columns:
        data_target = df[col].values.reshape(-1, 1)
        imputed_data = imputer.fit_transform(data_target)

        if col in ['bedrooms', 'beds']:
            imputed_data = imputed_data.astype(int)
        else:
            imputed_data = np.round(imputed_data, 1)

        df[col] = imputed_data


def clean_data(df):
    '''
    Remove unused columns.
    Save cleaned data file to csv.
    '''
    df = df[df['city'] == 'NYC'].copy()

    drop_columns = [
        'id', 'amenities', 'cancellation_policy', 'neighbourhood', 'city',
        'cleaning_fee', 'description', 'first_review', 'host_has_profile_pic', 'host_identity_verified',
        'host_response_rate', 'host_since', 'instant_bookable', 'last_review', 'name',
        'number_of_reviews', 'thumbnail_url', 'zipcode'
    ]
    df.drop(columns=drop_columns, inplace=True)

    columns = ['bathrooms', 'review_scores_rating', 'bedrooms', 'beds']
    fill_nan(df, columns=columns, strategy='mean')

    df.to_csv('data_clean.csv', index=False)




# def one_hot_encoder(file):
#     '''
#     Encoded 'room_type', 'bed_type'
#     '''
#     df = pd.read_csv(file)

#     encoder = OneHotEncoder()
#     category_target = ['room_type', 'bed_type']
#     transform = encoder.fit_transform(df[category_target])
#     # print(transform.toarray())
#     # print(encoder.categories_)
#     df_encoded = pd.DataFrame(transform.toarray(), columns=encoder.get_feature_names_out(category_target))

#     df = pd.concat([df, df_encoded], axis=1)
#     df.drop(columns=category_target, inplace=True)

#     return df

def train_pipeline(data, split_ratio=0.2, models=None):
    
    data = createAndSaveScaler(data)

    lat_scaler, long_scaler, log_price_scaler = loadScaler('models/lat_scaler.pkl',
                                                           'models/long_scaler.pkl',
                                                           'models/log_price_scaler.pkl')
    
    # Normalization
    data['latitude'] = lat_scaler.transform(data[['latitude']])
    data['longitude'] = long_scaler.transform(data[['longitude']])
    data['log_price'] = log_price_scaler.transform(data[['log_price']])

    # Split train and test data
    train_data, test_data = train_test_split(data, test_size=split_ratio, random_state=42)
    
    # Clustering lat_long using Elbow Method
    target_cluster = [
        ['latitude', 'longitude'],
        ['property_type_Bed & Breakfast', 'property_type_Boat',
         'property_type_Boutique hotel', 'property_type_Bungalow',
         'property_type_Cabin', 'property_type_Camper/RV',
         'property_type_Casa particular', 'property_type_Castle',
         'property_type_Cave', 'property_type_Chalet',
         'property_type_Condominium', 'property_type_Dorm',
         'property_type_Earth House', 'property_type_Guest suite',
         'property_type_Guesthouse', 'property_type_Hostel',
         'property_type_House', 'property_type_Hut', 'property_type_In-law',
         'property_type_Island', 'property_type_Lighthouse',
         'property_type_Loft', 'property_type_Other',
         'property_type_Parking Space', 'property_type_Serviced apartment',
         'property_type_Tent', 'property_type_Timeshare', 'property_type_Tipi',
         'property_type_Townhouse', 'property_type_Train',
         'property_type_Treehouse', 'property_type_Vacation home',
         'property_type_Villa', 'property_type_Yurt'],
    ]
    lat_long_n_clusters = find_optimal_cluster(train_data, max_cluster=10, target_cluster=target_cluster[0])
    cluster_lat_long = KMeans(n_clusters=lat_long_n_clusters, init='k-means++', max_iter=500, random_state=42).fit(train_data.drop(target_cluster[0], axis=1))
    dump(cluster_lat_long, open('models/cluster_lat_long.pkl', 'wb'))
    
    property_type_n_clusters = find_optimal_cluster(train_data, max_cluster=10, target_cluster=target_cluster[1])
    cluster_property_type = KMeans(n_clusters=property_type_n_clusters, init="k-means++", max_iter=500, random_state=42).fit(train_data.drop(target_cluster[1], axis=1))
    dump(cluster_property_type, open('models/cluster_property_type.pkl', 'wb'))

    # Predict Cluster on train_data + test_data
    train_data['cluster_lat_long'] = cluster_lat_long.predict(train_data.drop(target_cluster[0], axis=1))
    test_data['cluster_lat_long'] = cluster_lat_long.predict(test_data.drop(target_cluster[0], axis=1))

    train_data['cluster_property_type'] = cluster_property_type.predict(train_data.drop(target_cluster[1], axis=1).drop('cluster_lat_long', axis=1))
    test_data['cluster_property_type'] = cluster_property_type.predict(test_data.drop(target_cluster[1], axis=1).drop('cluster_lat_long', axis=1))

    train_data.drop(target_cluster[0], axis=1).drop(target_cluster[1], axis=1)
    test_data.drop(target_cluster[0], axis=1).drop(target_cluster[1], axis=1)

    # X_train, y_train, X_test, y_test = train_data.drop('log_price', axis=1), train_data['log_price'], test_data.drop('log_price', axis=1), test_data['log_price']

    # # createLogisticRegression()
    
if __name__ == '__main__':
    
    # data = pd.read_csv('Airbnb_Data.csv')
    
    # target_encode = ['room_type', 'bed_type', 'property_type']
    # encoded_data = one_hot_encoder('Airbnb_Data.csv', category_target=target_encode)
    # clean_data(concat_data(data, encoded_data, target_encode))

    df = pd.read_csv('data_clean.csv')

    train_pipeline(df, split_ratio=0.2, models=None)

    # models = [
    #     'LogisticRegression.pkl'
    # ]

    # # #Training
    # train_test_split_ratios = [0.1, 0.2, 0.3]

    # for split_ratio in train_test_split_ratios:
    #     train_pipeline(df_NYC, train_test_split=split_ratio, models=models)