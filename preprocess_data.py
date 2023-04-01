from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from utils.transformers import *
import pandas as pd
import numpy as np


df_train = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')

X = df_train.drop(columns=['target'])
y = df_train[['target']]

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=22)

pipe_features = Pipeline(steps=[
    ('col_dropper',
     ColumnDropper(
        columns_to_drop=[
            'property_id',
            'property_name',
            'host_id',
            'host_location',
            'host_since',
            'host_nr_listings_total',
            'host_response_time',
            'property_desc',
            'property_last_updated',
            'property_scraped_at',
            'property_zipcode',
            'property_sqfeet', 
            'property_neighborhood',
            'property_notes',
            'property_transit', 
            'property_access',
            'property_interaction',
            'property_rules',
            'reviews_first',
            'reviews_last',
            'property_bed_type'
            ]
        )
     ),
    ('missing_flagger',
     MissingFlagger(
        columns_to_flag=[
            'property_summary',
            'property_space',
            'host_about'    
            ]
        )
     ),
    ('amenities_counter',
     AmenitiesCounter()
     ),
    ('host_verified_counter',
     HostVerificationsCounter()
     ),
    ('extras_handler',
     ExtrasHandler()
     ),
    ('clust_location',
     GaussianClusterer(
        n_clusters=7,
        features_cluster=['property_lat', 'property_lon'],
        initial_centroids = np.array([
            [51.24, 4.34], [51.20, 4.41], [51.20, 4.45],
            [50.85, 4.30], [50.85, 4.35], [50.85, 4.38], [50.85, 4.43]
            ])
        )
     ),
    ('property_type_handler',
     PropertyTypeHandler()
     ),
    ('booking_cancel_handler',
     BookingCancelHandler()
     ),
    ('encoder',
     CustomOneHotEncoder(
        columns=[
            'property_type_new',
            'property_room_type',
            'location_zone_g',
            'booking_cancel_policy'
            ]
        )
     ),
    ('imputer',
     CustomIterativeImputer()
     ),
    ('scaler',
     CustomStandardScaler(
        columns=[
            'property_max_guests',
            'property_bathrooms',
            'property_bedrooms',
            'property_beds',
            'host_response_rate',
            'host_nr_listings',
            'booking_price_covers',
            'booking_min_nights',
            'booking_max_nights',
            'booking_availability_30',
            'booking_availability_60',
            'booking_availability_90',
            'booking_availability_365',
            'reviews_num',
            'reviews_rating',
            'reviews_acc',
            'reviews_cleanliness',
            'reviews_checkin',
            'reviews_communication',
            'reviews_location',
            'reviews_value',
            'reviews_per_month',
            'count_amenities',
            'count_host_verifications'
        ]
     )),
    ('pca_reviews',
     PCATransformer(
        n_components=3,
        columns=[
            'reviews_acc',
            'reviews_cleanliness',
            'reviews_checkin',
            'reviews_communication',
            'reviews_location',
            'reviews_value'
        ]
     )),
    ('lgbm', LGBMRegressor(reg_lambda = 10, reg_alpha=10, learning_rate=0.01, objetive='quantile'))
])

pipe_target = Pipeline(steps=[
    ('log_transform', FunctionTransformer(np.log, inverse_func = np.exp, check_inverse = True))
])
#('lgbm', LGBMRegressor(reg_lambda = 10, reg_alpha=10, learning_rate=0.01))
#('xgb', XGBRegressor(reg_lambda = 10, gamma=1, reg_alpha=10, max_depth=3, objective='reg:squarederror'))

model = TransformedTargetRegressor(regressor=pipe_features, transformer=pipe_target)

cv = KFold(n_splits=10)
scores = cross_val_score(pipe_features, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)

cv2 = KFold(n_splits=10)
scores2 = cross_val_score(model, X, y, scoring='neg_median_absolute_error', cv=cv2, n_jobs=-1)

dummy_pred = np.array([np.median(y) for i in range(len(y_val))])

dummy_error = mean_squared_error(y_val, dummy_pred, squared=False)

model.fit(X,y)

pred = model.predict(df_test)

pred_df = df_test[['property_id']]

pred_df['pred_price'] = pred

pred_df.to_csv('pred_v1.csv', header=False, index=False)

###############################################################################
###############################################################################

pipe_features2 = Pipeline(steps=[
    ('col_dropper',
     ColumnDropper(
        columns_to_drop=[
            'property_id',
            'property_name',
            'host_id',
            'host_location',
            'host_since',
            'host_nr_listings_total',
            'host_response_time',
            'property_desc',
            'property_last_updated',
            'property_scraped_at',
            'property_zipcode',
            'property_sqfeet', 
            'property_neighborhood',
            'property_notes',
            'property_transit', 
            'property_access',
            'property_interaction',
            'property_rules',
            'reviews_first',
            'reviews_last',
            'property_bed_type',
            'property_space',
            'property_amenities',
            'host_about',
            'property_summary'
            ]
        )
     ),
    ('host_verified_counter',
     HostVerificationsCounter()
     ),
    ('extras_handler',
     ExtrasHandler()
     ),
    ('clust_location',
     GaussianClusterer(
        n_clusters=7,
        features_cluster=['property_lat', 'property_lon'],
        initial_centroids = np.array([
            [51.24, 4.34], [51.20, 4.41], [51.20, 4.45],
            [50.85, 4.30], [50.85, 4.35], [50.85, 4.38], [50.85, 4.43]
            ])
        )
     ),
    ('property_type_handler',
     PropertyTypeHandler()
     ),
    ('booking_cancel_handler',
     BookingCancelHandler()
     ),
    ('encoder',
     CustomOneHotEncoder(
        columns=[
            'property_type_new',
            'property_room_type',
            'location_zone_g'
            ]
        )
     ),
    ('imputer',
     CustomIterativeImputer()
     ),
    ('scaler',
     CustomStandardScaler(
        columns=[
            'property_max_guests',
            'property_bathrooms',
            'property_bedrooms',
            'property_beds',
            'host_response_rate',
            'host_nr_listings',
            'booking_price_covers',
            'booking_min_nights',
            'booking_max_nights',
            'booking_availability_30',
            'booking_availability_60',
            'booking_availability_90',
            'booking_availability_365',
            'reviews_num',
            'reviews_rating',
            'reviews_acc',
            'reviews_cleanliness',
            'reviews_checkin',
            'reviews_communication',
            'reviews_location',
            'reviews_value',
            'reviews_per_month',
            'count_host_verifications'
        ]
     )),
    ('pca_reviews',
     PCATransformer(
        n_components=3,
        columns=[
            'reviews_acc',
            'reviews_cleanliness',
            'reviews_checkin',
            'reviews_communication',
            'reviews_location',
            'reviews_value'
        ]
     )),
    ('lgbm', LGBMRegressor())
])

#reg_lambda = 10, reg_alpha=10, learning_rate=0.01
pipe_target2 = Pipeline(steps=[
    ('log_transform', FunctionTransformer(np.log, inverse_func = np.exp, check_inverse = True))
])

model2 = TransformedTargetRegressor(regressor=pipe_features2, transformer=pipe_target2)

cv2 = KFold(n_splits=10)
scores2 = cross_val_score(model2, X, y, scoring='neg_root_mean_squared_error', cv=cv2, n_jobs=-1)

cv2 = KFold(n_splits=10)
scores2 = cross_val_score(model2, X, y, scoring='neg_root_mean_squared_error', cv=cv2, n_jobs=-1)


model2.fit(X_train,y_train['target_night'])

pred = model2.predict(X_val)

pred_correct = pred/\
    ((1/3)*X_val.booking_price_covers.values + (2/3)*X_val.property_max_guests.values)


error = mean_squared_error(y_val.target.values, pred_correct, squared=False)


# model2.fit(X,y['target_night'])

# pred = model2.predict(df_test)

# pred_correct = pred/\
#     ((1/3)*df_test.booking_price_covers.values + (2/3)*df_test.property_max_guests.values)

# pred_df = df_test[['property_id']]

# pred_df['pred_price'] = pred_correct

# pred_df.to_csv('pred_v2.csv', header=False, index=False)



###################################################################################################

