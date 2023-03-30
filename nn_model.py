import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from torchmetrics.regression import MeanSquaredError
from utils.transformers import *

class TorchDS(data_utils.Dataset):
 
  def __init__(self, feats, target):
    #x=feats.values
    #y=target.values
 
    self.x_ds=torch.as_tensor(feats.values,dtype=torch.float32)
    self.y_ds=torch.as_tensor(target,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_ds)
   
  def __getitem__(self,idx):
    return self.x_ds[idx],self.y_ds[idx]


# Neural Network Model Class
class NeuralNetwork(nn.Module):
    def __init__(self, input_vars, params):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_vars, params['n_unit_l1']),
            nn.LeakyReLU(),
            nn.Linear(params['n_unit_l1'], params['n_unit_l2']),
            nn.LeakyReLU(),
            nn.Linear(params['n_unit_l2'], 1),
            #nn.ReLU(),
            #nn.Linear(params['n_unit_l3'], 1)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


# Build neural network model
def build_model(params, input_vars):
    
    model = NeuralNetwork(input_vars, params)

    return model

def train_one_epoch(model, error_fn, optimizer, train_data, train_loader):
    running_rmse = 0.
    last_rmse = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_data):

        X, y = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        pred = model(X)

        # Compute the loss and its gradients (add eps to avoid sqrt of 0 resulting in NaN)
        eps = 1e-6
        error = torch.sqrt(error_fn(pred, y) + eps)
        error.backward()

        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_rmse += error.item()

        if len(train_loader) == i+1:
            last_rmse = running_rmse / len(train_loader) # loss per epoch
            running_rmse = 0.

    return last_rmse


# Train and evaluate the accuarcy of neural network model
def train_and_evaluate(params):
    # df = pd.read_csv('data.csv')

    # df.drop(columns=['id'], inplace=True)
    # scaler = StandardScaler()
    # scaled_vars = scaler.fit_transform(df.drop(columns=['diagnosis']))
    # scaled_features_df = pd.DataFrame(scaled_vars, index=df.index, columns=df.drop(columns=['diagnosis']).columns)
    # df['diagnosis_encoded'] = np.where(df['diagnosis'] == 'M', 1, 0)

    # df_scaled = pd.concat([df[['diagnosis_encoded']], scaled_features_df], axis=1)

    # features = df_scaled.drop(columns=['diagnosis_encoded'])
    # target = df_scaled[['diagnosis_encoded']]

    df_train = pd.read_csv('train.csv')

    #df_test = pd.read_csv('test.csv')

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
        ))
    ])

    pipe_target = Pipeline(steps=[
        ('log_transform', FunctionTransformer(np.log, inverse_func = np.exp, check_inverse = True)),
        ('scaler', StandardScaler())
    ])

    features = df_train.drop(columns=['target'])
    target = df_train[['target']]

    #X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=1)

    kf = KFold(n_splits=5, shuffle=True)

    cv_train_rmse_values = []
    cv_val_rmse_values = []
    
    print('Parameters Being Tested:\n')
    for key,value in params.items():
        print('{}:{}\n'.format(key, value))
    
    for train_index, val_index in kf.split(features, target):
        # Creates the train features and target datasets for the fold
        train_feats_fold = features.iloc[train_index]
        train_target_fold = target.iloc[train_index]

        train_target_fold = train_target_fold

        # Creates the validation features and target dataset for the fold
        val_feats_fold = features.iloc[val_index]
        val_target_fold = target.iloc[val_index].values

        # Fit transforms the features train dataset and use that to transform the features validation dataset
        train_feats_fold = pipe_features.fit_transform(train_feats_fold)
        val_feats_fold = pipe_features.transform(val_feats_fold)

        # Fit transforms the target train dataset
        train_target_fold = pipe_target.fit_transform(train_target_fold)

        input_vars = train_feats_fold.shape[1]

        train_ds = TorchDS(train_feats_fold, train_target_fold)

        val_ds = TorchDS(val_feats_fold, val_target_fold)

        batch_size = len(train_index)
        #batch_size = params['batch_size']

        train_loader = data_utils.DataLoader(train_ds, batch_size=batch_size)

        val_loader = data_utils.DataLoader(val_ds, batch_size=batch_size)

        model = build_model(params, input_vars)

        error_fn = nn.MSELoss()

        #optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['lr'], weight_decay=params['L2'])

        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['L2'])

        num_epochs = params['epochs']
        train_rmse_values = []
        val_rmse_values = []

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_rmse = train_one_epoch(model, error_fn, optimizer, train_loader, train_loader)
            train_rmse_values.append(avg_rmse)

            # We don't need gradients on to do reporting
            with torch.no_grad():
                model.train(False) 

                running_vrmse = 0.0
                for i, vdata in enumerate(val_loader):
                    vX, vy = vdata
                    
                    # Predicted values are on the log scale
                    vpred_log = model(vX)

                    # Transforms back to normal scale
                    vpred = torch.as_tensor(pipe_target.inverse_transform(vpred_log))

                    vrmse = torch.sqrt(error_fn(vpred, vy))
                    running_vrmse += vrmse.item()
                    
            avg_vrmse = running_vrmse / (i + 1)

            print('CURRENT VAL RMSE: {}'.format(round(avg_vrmse, 3)))
        
            val_rmse_values.append(avg_vrmse)

        cv_train_rmse_values.append(train_rmse_values[-1])
        cv_val_rmse_values.append(val_rmse_values[-1])

    cv_val_rmse = np.array(cv_val_rmse_values).mean()
    
    print('\nFINISHED CROSS VAL WITH RMSE={}:\n'.format(cv_val_rmse))

    return cv_val_rmse
    
  
# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy 
def objective(trial):

    params = {
              'lr': trial.suggest_float('lr', 1e-3, 1e-1),
              #'optimizer': trial.suggest_categorical("optimizer", ["SGD", "Adam"]),
              'n_unit_l1': trial.suggest_int("n_unit_l1", 189, 336, step=21),
              'n_unit_l2': trial.suggest_int("n_unit_l2", 63, 168, step=21),
              'epochs': trial.suggest_int("epochs", 30, 80, step=10),
              'L2': trial.suggest_float('L2', 1e-1, 5),
              'batch_size': trial.suggest_int("batch_size", 200, 2000, step=200)
              }
    
    cv_val_rmse = train_and_evaluate(params)

    return cv_val_rmse

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=200)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

df_trials = study.trials_dataframe().sort_values(by=['value'], ascending=True)

df_trials.to_csv('hyperparameter_opt_trial_2.csv', index=False)