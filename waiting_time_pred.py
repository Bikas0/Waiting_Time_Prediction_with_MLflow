import os
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('data_with_waiting_times.csv')
data = data[["TKIS_TIME","SERVICE_NAME", "QLength", "waiting_time_minutes"]]
# Convert TKIS_TIME to datetime
data['TKIS_TIME'] = pd.to_datetime(data['TKIS_TIME'], errors = 'coerce')

# Extract day name
data['Day_Name'] = data['TKIS_TIME'].dt.day_name()

# Extract time and classify as morning or evening
data['Time'] = data['TKIS_TIME'].dt.strftime('%H:%M:%S')
data['Period'] = data['TKIS_TIME'].apply(lambda x: 'Morning' if x.hour < 12 else 'Evening')
data.drop('TKIS_TIME', axis = 1, inplace=True)
# Initialize LabelEncoders
le_service_name = LabelEncoder()
le_day_name = LabelEncoder()
le_period = LabelEncoder()

# Apply label encoding
data['SERVICE_NAME'] = le_service_name.fit_transform(data['SERVICE_NAME'])
data['Day_Name'] = le_day_name.fit_transform(data['Day_Name'])
data['Period'] = le_period.fit_transform(data['Period'])

# Convert 'Time' column to minutes past midnight
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(data['Time'], format='%H:%M:%S').dt.minute
# Split the dataset
X = data.drop('waiting_time_minutes', axis=1)  # Features
y = data['waiting_time_minutes']  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalize the features
# # Min-Max Scaling
# min_max_scaler = MinMaxScaler()
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.transform(X_test)
# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)

# Initialize and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)

mlflow.set_experiment("Waiting_Time_Prediction")
# Model evelaution metrics
def eval_metrics(actual, pred):
    rmse = mean_squared_error(actual,pred,squared=False)
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    
    return(rmse, mae, r2)


from mlflow.data.pandas_dataset import PandasDataset

# Create an instance of a PandasDataset
dataset = mlflow.data.from_pandas(
    data, name="QPro Dataset", targets="waiting_time_minutes"
)

def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(run)
        mlflow.set_tag('Version', run_id)
        mlflow.log_input(dataset, context="training")
        pred = model.predict(X)
        #metrics
        (rmse, mae, r2) = eval_metrics(y, pred)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        # model
        mlflow.sklearn.log_model(model, name)
        
    
    mlflow.end_run()

mlflow_logging(rf_model, X_test, y_test, 'Random_Forest_Regressor')
mlflow_logging(lr_model, X_test, y_test, 'Linear_Regression')