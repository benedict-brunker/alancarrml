from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score


def score_regressors(y_preds, y_actuals, set_name): 
    """
    Prints RMSE, MSE and MAE scores given y_preds and y_actuals. 

    Parameters 
    _________________________________________________________________________

    Returns 
    _________________________________________________________________________ 
    Dict of scores keyed by names: {'score_name': score}
    Example: {
        'RMSE': 1999, 
        'MSE': 8790,
        'MAE': 2879 
    }
    """
    # compute RMSE, MSE and MAE scores 
    rmse = root_mean_squared_error(y_actuals, y_preds) 
    mse = mean_squared_error(y_actuals, y_preds) 
    mae = mean_absolute_error(y_actuals, y_preds) 
    r2 = r2_score(y_actuals, y_preds) 

    # Print scores 
    print(f"Scores for {set_name}: ") 
    print(f"\n      Root Mean Squared Error: {rmse}") 
    print(f"        Mean Squared Error: {mse}")  
    print(f"        Mean Absolute Error: {mae}") 
    print(f"        R2 Score: {r2}") 

    # Return scores 
    return {
        'name': set_name, 
        'rmse': rmse, 
        'mse': mse, 
        'mae': mae, 
        'r2': r2
    }