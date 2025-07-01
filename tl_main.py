import pandas as pd
import json

def get_params_from_trial_info(csv_path, model_name, params_column='params', name_column='model_name'):
    """
    Given a CSV file path and a model name, returns the parameters as a dictionary.
    Assumes 'params' is a JSON-formatted string in the CSV.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Find the row matching the model name
    row = df[df[name_column] == model_name]

    if row.empty:
        raise ValueError(f"Model name '{model_name}' not found in {csv_path}")

    # Get the stringified JSON from the params column
    params_str = row.iloc[0][params_column]
    
    try:
        params_dict = json.loads(params_str)
    except Exception as e:
        print(f"Failed to parse params for {model_name}: {params_str}")
        raise e

    return params_dict


if __name__ == "__main__":
    model_name = '20250701194739_t21_base_residual.pt'
    #model_name = '20250701194906_t16_base_residual.pt'

    params = get_params_from_trial_info('logs/trial_info.csv', model_name, 'params', 'model_name')
    data_config = get_params_from_trial_info('logs/trial_info.csv', model_name, 'data', 'model_name')


    