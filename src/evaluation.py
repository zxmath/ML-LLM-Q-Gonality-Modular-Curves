from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
def compute_r2_and_accuracy(y_pred, y_test):
    #y_pred = model.predict(X_test)
    y_pred = np.floor(y_pred+0.5)
    # Calculate R^2 score
    r2 = r2_score(y_test, y_pred)
    # Calculate accuracy (as decimal 0-1, not percentage)
    accuracy = np.mean(y_pred == y_test)
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Print the results
    print(f"R^2: {r2:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"RMSE: {rmse:.4f}")
    return r2, accuracy, rmse
def compute_accuracy_in_bounds(y_pred, y_bounds):
    
    y_pred = np.floor(y_pred+0.5)
    # Check if the predicted values are within the bounds
    in_bounds = (y_pred >= y_bounds.map(lambda b: b[0])) & (y_pred <= y_bounds.map(lambda b: b[1]))
    # Calculate accuracy (as decimal 0-1, not percentage)
    accuracy = np.mean(in_bounds)
    # Print the results
    print(f"Accuracy within bounds: {accuracy*100:.2f}%")
    return accuracy