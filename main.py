import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

def clean_data(df):
    '''
    Remove unused columns.
    Save cleaned data file to csv.
    '''
    # df = pd.read_csv(file)

    drop_columns = [
        'id', 'amenities', 'neighbourhood', 'cancellation_policy',
        'cleaning_fee', 'description', 'first_review', 'host_has_profile_pic', 'host_identity_verified',
        'host_response_rate', 'host_since', 'instant_bookable', 'last_review', 'name',
        'neighbourhood', 'number_of_reviews', 'review_scores_rating', 'thumbnail_url', 'zipcode'
    ]
    df.drop(columns=drop_columns, inplace=True)

    item = ['bathrooms', 'bedrooms', 'beds']
    
    def fill_nan(item):
        for i in item:
            df[i] = df[i].fillna(0)

    
    fill_nan(item)
    df.to_csv('data_clean.csv', index=False)


def one_hot_encoder(file):
    '''
    Encoded 'room_type', 'bed_type'
    '''
    df = pd.read_csv(file)

    encoder = OneHotEncoder()
    category_target = ['room_type', 'bed_type']
    transform = encoder.fit_transform(df[category_target])
    # print(transform.toarray())
    # print(encoder.categories_)
    df_encoded = pd.DataFrame(transform.toarray(), columns=encoder.get_feature_names_out(category_target))

    df = pd.concat([df, df_encoded], axis=1)
    df.drop(columns=category_target, inplace=True)

    return df

if __name__ == '__main__':
    
    encoded_data = one_hot_encoder('Airbnb_Data.csv')
    clean_data(encoded_data)

    df = pd.read_csv('data_clean.csv')
    df_NYC = df[df['city'] == 'NYC']