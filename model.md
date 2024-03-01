import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace with your data)
df = pd.read_excel("F:\Project\sashikant 2\Shasikant 2 ml data.xlsx")
x = df.values[:, :9]
y = df.values[:, 9]

# Normalize the input data
#scaler_x = MinMaxScaler()
#scaler_y = MinMaxScaler()
#x_normalised = scaler_x.fit_transform(x)
#y_normalised = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

# Define the objective function to minimize (RMSE)
def objective_function(params):
    n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf = params
    
    reg = RandomForestRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=2
    )
    
    reg.fit(x_train, y_train)
    y_train_pred = reg.predict(x_train)
    y_test_pred = reg.predict(x_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return rmse_test  # We aim to minimize RMSE on the test data

# Define the bounds for each hyperparameter
bounds = [(60, 110), (0.05, 1.0), (12, 30), (2, 10), (1, 10)]

# Perform hyperparameter optimization using Differential Evolution
result = differential_evolution(objective_function, bounds, maxiter=50, seed=2)

# Get the best hyperparameters
best_hyperparameters = result.x
best_test_rmse = result.fun

print('Best Hyperparameters:', best_hyperparameters)
print('Best RMSE on Test Data:', best_test_rmse)

# Train a Gradient Boosting Regressor with the best hyperparameters on the full training data
best_n_estimators, best_learning_rate, best_max_depth, best_min_samples_split, best_min_samples_leaf = best_hyperparameters

final_reg = GradientBoostingRegressor(
    n_estimators=int(best_n_estimators),
    learning_rate=best_learning_rate,
    max_depth=int(best_max_depth),
    min_samples_split=int(best_min_samples_split),
    min_samples_leaf=int(best_min_samples_leaf),
    random_state=2
)

final_reg.fit(x_train, y_train)

# Evaluate the final model on both training and test data
y_train_pred = final_reg.predict(x_train)
y_test_pred = final_reg.predict(x_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print('RMSE on Training Data with Best Hyperparameters:', train_rmse)
print('RMSE on Test Data with Best Hyperparameters:', test_rmse)
print('R-squared (R2) on Training Data:', train_r2)
print('R-squared (R2) on Test Data:', test_r2)
# Create a DataFrame for visualization
df_final = pd.DataFrame()
df_final['Input1'] = x_train[:,0]
df_final['Input2'] = x_train[:,1]
df_final['Input3'] = x_train[:,2]
df_final['Input4'] = x_train[:,3]
df_final['Input5'] = x_train[:,4]
df_final['Input6'] = x_train[:,5]
df_final['Input7'] = x_train[:,6]
df_final['Input8'] = x_train[:,7]
df_final['Input9'] = x_train[:,8]
df_final['Actual out1'] = y_train
df_final['Predicted out1'] = y_train_pred
df_final
import pickle
with open("gb_de_scc.pkl", "wb") as file:
    pickle.dump(final_reg, file)
