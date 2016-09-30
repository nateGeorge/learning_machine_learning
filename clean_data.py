import pandas as pd
import numpy as np

def load_clean_data(standardize=False, intercept=False):
    # to use:
    # from clean_data import load_clean_data
    # df, X, y = load_clean_data()
    # standardize will mean-center and std-normalize some columns
    df = pd.read_csv('data/churn.csv')

    #print df.info()

    #print df.dropna().describe() # gets means for all columns, even ones with na

    #the data was pulled on July 1, 2014; we consider a user retained if they were active
    #(i.e. took a trip) in the preceding 30 days (from the day the data was pulled)

    # convert dates to datetimes
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['wknd'] = df['signup_date'].dt.dayofweek
    df['wknd'] = np.where(df['wknd'] < 5, 0, 1)
    df['signup_day'] = df['signup_date'].dt.day
    df.drop('signup_date', inplace=True, axis=1)

    # fill na values
    df['phone'] = df['phone'].fillna('iPhone') # fill with mode
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(5.0) # 5 is by far the mode
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(5.0) # for this too

    df['iPhone'] = pd.get_dummies(df['phone'], drop_first=True)
    df.drop('phone', inplace=True, axis=1)

    df = df.merge(pd.get_dummies(df['city'], drop_first=True), left_index=True, right_index=True)
    df.drop('city', inplace=True, axis=1)


    df['churn'] = np.where(df['last_trip_date'].dt.month == 6, 0, 1)

    df.drop('last_trip_date', inplace=True, axis=1)

    df['luxury_car_user'] = df['luxury_car_user'].map({True: 1, False: 0})

    if standardize: # mean-center and stddev norm data
        standardized = {}
        cols = ['avg_dist', 'avg_rating_of_driver', 'avg_rating_by_driver',
        'surge_pct', 'avg_surge', 'trips_in_first_30_days']
        for c in cols:
            mean = df[c].mean()
            std = df[c].std()
            standardized[c] = {}
            standardized[c][mean] = mean
            standardized[c][std] = std
            df[c] = (df[c] - mean) / std

    if intercept:
        df['constant'] = 1.

    df2 = df.copy()
    y = df.pop('churn').values
    X = df.values

    if standardize:
        return standardized, df2, X, y

    return df2, X, y
