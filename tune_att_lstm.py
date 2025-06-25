"""
Hyperparameter Tuning Orchestration Script for LSTM-based Stock Prediction Models.

This script provides a framework for systematically tuning hyperparameters for
Attention LSTM (ATTLSTM) and Standard LSTM models for stock price prediction.
It iterates over specified ranges of LASSO alphas (for feature selection),
look-back periods (for sequence creation), and model-specific hyperparameters.

Key Components:
1.  Global Configurations:
    - STOCK_TICKER, YEARS_OF_DATA, LASSO_ALPHAS_TO_TEST, LOOK_BACK_PERIODS_TO_TEST
    - MAX_EPOCHS_HYPERBAND, HYPERBAND_ITERATIONS, EARLY_STOPPING_PATIENCE
    - TUNER_BATCH_SIZE, BASE_PROJECT_DIR (for results), SEED (for reproducibility).
    - These can be adjusted for quick tests or comprehensive runs.

2.  `set_seeds(seed_value)`: Utility to set random seeds for numpy, TensorFlow, etc.

3.  `create_sequences(data_df, look_back, target_col_name)`: Utility to transform
    time-series data into sequences suitable for LSTM input.

4.  `run_one_tuning_configuration(...)`:
    - Manages the KerasTuner hyperparameter search for a *single* combination of:
        - model_type ('att_lstm' or 'standard_lstm')
        - look_back_period
        - lasso_alpha
        - preprocessed_data_df (data already processed for the specific lasso_alpha)
    - Creates sequences from the provided data using the current look_back_period.
    - Performs a temporal train/validation/test split (70%/15%/15%).
    - Sets up and runs KerasTuner (Hyperband) to find the best model hyperparameters.
    - Stores tuner logs in a structured directory:
      `BASE_PROJECT_DIR/model_type/lb_XX/alpha_YY/tuning_logs/`.
    - Evaluates the best found model on a dedicated test set (MSE, MAE, RMSE).
    - Returns a dictionary summarizing the configuration, results, and status.

5.  `main_tuning_orchestrator(...)`:
    - The main driver function for the entire tuning process.
    - Takes parameters like `model_type_to_tune`, `stock_ticker`, `years_of_data`,
      lists of `lasso_alphas` and `look_back_periods` to test, and tuning settings.
    - Iterates through:
        - Model types (if 'both' is selected).
        - Differencing options (e.g., `use_differencing=[False, True]`).
        - LASSO alphas: For each alpha, it preprocesses the full dataset.
            - Inner loop: Look-back periods: Calls `run_one_tuning_configuration`.
    - Aggregates results from all configurations.
    - Saves comprehensive JSON summaries of results:
        - Per model type (e.g., `tuning_summary_att_lstm.json`).
        - An overall summary (`tuning_summary_ALL_CONFIGS.json`).
    - Prints the overall best configuration found based on test MSE.

Usage:
- Configure the global variables at the top of the script for desired stock,
  data range, and hyperparameter search spaces.
- In the `if __name__ == '__main__':` block, set `MODEL_TO_TUNE`
  ('att_lstm', 'standard_lstm', or 'both') and `USE_DIFFERENCING_OPTIONS`.
- Execute the script: `python tune_att_lstm.py`

The script relies on:
- `data_preprocessing_module.py`: For data downloading, indicator calculation,
  LASSO feature selection, and scaling.
- `att_lstm_module.py`: Defines the Attention LSTM model and its hyperparameter space.
- `standard_lstm_module.py`: Defines the Standard LSTM model and its hyperparameter space.
"""

import keras_tuner as kt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
import time
import random # For os.environ['PYTHONHASHSEED']

from data_preprocessing_module import DataPreprocessor
from att_lstm_module import ATTLSTMModel
from standard_lstm_module import StandardLSTMModel

# --- Global Configurations ---
# These parameters define the scope and intensity of the hyperparameter tuning process.
# Adjust them for quick tests or for comprehensive, long-running tuning sessions.

STOCK_TICKER = "^AEX"  # Target stock ticker for Yahoo Finance
YEARS_OF_DATA = 15     # Number of years of historical data to use for tuning
# Example for a quick test: YEARS_OF_DATA = 1

# LASSO alphas for feature selection. Data will be preprocessed for each alpha.
LASSO_ALPHAS_TO_TEST = [0.001, 0.005, 0.01]
# Example for a quick test: LASSO_ALPHAS_TO_TEST = [0.01]

# Look-back periods (sequence lengths) for LSTM models.
LOOK_BACK_PERIODS_TO_TEST = [60, 90, 120, 180, 250] # In trading days
# Example for a quick test: LOOK_BACK_PERIODS_TO_TEST = [30, 60]

# KerasTuner Hyperband algorithm parameters
MAX_EPOCHS_HYPERBAND = 100  # Max epochs a single model can be trained for in Hyperband (e.g., 100-200 for full runs)
# Example for a quick test: MAX_EPOCHS_HYPERBAND = 10
HYPERBAND_ITERATIONS = 2    # Number of times to iterate over the Hyperband algorithm (e.g., 1-3)
# Example for a quick test: HYPERBAND_ITERATIONS = 1

# Early stopping configuration for model training within KerasTuner
EARLY_STOPPING_PATIENCE = 15 # Patience for early stopping (e.g., 15-25 for full runs)
# Example for a quick test: EARLY_STOPPING_PATIENCE = 3

# Batch size for training models during tuning. Can also be a hyperparameter.
TUNER_BATCH_SIZE = 64

# Output directory for all tuning results, logs, and summaries
BASE_PROJECT_DIR = "keras_tuner_results"
SEED = 42  # Random seed for reproducibility

def set_seeds(seed_value=SEED):
    """
    Sets random seeds for major libraries to ensure reproducibility.

    Args:
        seed_value (int): The seed value to use.
    """
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value) # For os.environ['PYTHONHASHSEED'] potentially, and general python 'random'
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # For Keras/TF operations that might use other sources of randomness,
    # tf.config.experimental.enable_op_determinism() can be used in TF >= 2.8
    # but might impact performance and not all ops are supported.
    print(f"Seeds set to: {seed_value}")

set_seeds(SEED) # Set seeds at the beginning of the script


def create_sequences(data_df, look_back, target_col_name):
    """
    Creates sequences (X) and corresponding targets (y) from time-series data.

    Args:
        data_df (pd.DataFrame): DataFrame containing features and target, scaled.
                                Assumes target is one of the columns.
        look_back (int): The number of previous time steps to use as input features.
        target_col_name (str): The name of the target column in data_df.

    Returns:
        tuple: (np.array, np.array)
            - X_seq: Array of input sequences. Shape: (samples, look_back, num_features_in_df).
            - y_seq: Array of target values. Shape: (samples,).
    """
    target_col_idx = data_df.columns.get_loc(target_col_name)
    X, y = [], []
    # Iterate up to the point where a full sequence and its target can be formed
    for i in range(len(data_df) - look_back):
        # Extract all columns for the sequence input X
        X.append(data_df.iloc[i:(i + look_back), :].values)
        # Extract the target value for y
        y.append(data_df.iloc[i + look_back, target_col_idx])
    return np.array(X), np.array(y)


def run_one_tuning_configuration(
    model_type,
    look_back_period,
    lasso_alpha,
    preprocessed_data_df,
    # num_features parameter was here, but input_shape is derived from X_seq.shape directly.
    # If needed for other purposes, it can be re-added.
    config_params
):
    """
    Runs KerasTuner for a single hyperparameter tuning configuration.

    This function takes a specific model type, look-back period, LASSO alpha,
    the corresponding preprocessed data, and tuning settings. It then:
    1. Creates time-series sequences.
    2. Splits data into training, validation, and test sets (temporally).
    3. Initializes and runs the KerasTuner (Hyperband) search.
    4. Retrieves the best hyperparameters found by the tuner.
    5. Builds the best model using these hyperparameters.
    6. Evaluates the best model on the dedicated test set.
    7. Returns a dictionary summarizing the results of this configuration.

    Args:
        model_type (str): Type of model to tune ('att_lstm' or 'standard_lstm').
        look_back_period (int): The sequence length for LSTM input.
        lasso_alpha (float): The LASSO alpha value used for preprocessing this data.
        preprocessed_data_df (pd.DataFrame): The fully preprocessed (selected features, scaled)
                                             data for the given `lasso_alpha` and `use_differencing`.
                                             Target column is assumed to be the last column.
        config_params (dict): Dictionary containing other tuning parameters like
                              'max_epochs_hyperband', 'hyperband_iterations',
                              'early_stopping_patience', 'tuner_batch_size'.

    Returns:
        dict: A dictionary containing the results of this tuning run, including
              status, best hyperparameters, test metrics, duration, and tuner log directory.
    """
    print(f"\n--- Starting Tuning for: Model={model_type}, LookBack={look_back_period}, LassoAlpha={lasso_alpha}, Diff={config_params.get('use_differencing')} ---")
    start_time_config = time.time()

    if preprocessed_data_df.empty:
        print(f"Error: Preprocessed data for Lasso Alpha {lasso_alpha} is empty. Skipping this configuration.")
        return {
            "model_type": model_type,
            "look_back_period": look_back_period,
            "lasso_alpha": lasso_alpha,
            "use_differencing": config_params.get('use_differencing'),
            "status": "failed_preprocessing_empty",
            "best_hyperparameters": None,
            "test_metrics": None,
            "duration_seconds": time.time() - start_time_config
        }

    target_column_name = preprocessed_data_df.columns[-1]

    # 1. Create sequences based on current look_back_period
    print(f"Creating sequences with look_back = {look_back_period}...")
    X_seq, y_seq = create_sequences(preprocessed_data_df, look_back_period, target_column_name)

    if len(X_seq) == 0:
        print(f"Error: No sequences created with look_back = {look_back_period}. Check data length. Skipping.")
        return {
            "model_type": model_type,
            "look_back_period": look_back_period,
            "lasso_alpha": lasso_alpha,
            "use_differencing": config_params.get('use_differencing'),
            "status": "failed_no_sequences",
            "best_hyperparameters": None,
            "test_metrics": None,
            "duration_seconds": time.time() - start_time_config
        }
    print(f"Sequences created: X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

    # 2. Temporal train/validation/test split (e.g., 70% train, 15% val, 15% test)
    total_samples = len(X_seq)
    # Check if there's enough data for a meaningful split (e.g., at least a few samples for each set)
    if total_samples < 10: # Minimum threshold for samples, can be adjusted.
        print(f"Error: Not enough sequence samples ({total_samples}) for splitting. Skipping.")
        return {
            "model_type": model_type,
            "look_back_period": look_back_period,
            "lasso_alpha": lasso_alpha,
            "use_differencing": config_params.get('use_differencing'),
            "status": "failed_insufficient_samples_for_split",
            "best_hyperparameters": None,
            "test_metrics": None,
            "duration_seconds": time.time() - start_time_config
        }

    train_size = int(total_samples * 0.70)
    val_size = int(total_samples * 0.15)
    # test_size is the remainder: total_samples - train_size - val_size

    # First split: Separate training set from temporary set (validation + test)
    X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(
        X_seq, y_seq, train_size=train_size, shuffle=False, random_state=SEED
    )

    # Second split: Separate validation set from test set from the temporary set
    # Adjust val_size calculation to be a proportion of X_temp_seq to ensure it's valid
    # Or ensure val_size does not exceed X_temp_seq length.
    if len(X_temp_seq) > 1 : # Need at least 2 samples in temp to split into val and test
        # Ensure validation set size is reasonable and leaves some for test if possible
        # If val_size (15% of total) is larger than X_temp_seq, it means test set would be empty or negative.
        # Cap val_size to leave at least one sample for test if X_temp_seq is small.
        actual_val_size_from_total_percentage = val_size
        # If remaining (X_temp_seq) is too small for the val_size based on total percentage, adjust.
        if actual_val_size_from_total_percentage >= len(X_temp_seq):
            actual_val_size = max(1, len(X_temp_seq) - 1) # Ensure val is at least 1, leave 1 for test
        else:
            actual_val_size = actual_val_size_from_total_percentage

        # Ensure actual_val_size is not zero if X_temp_seq has elements
        if actual_val_size == 0 and len(X_temp_seq) > 0: actual_val_size = 1


        X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(
            X_temp_seq, y_temp_seq, train_size=actual_val_size, shuffle=False, random_state=SEED
        )
    elif len(X_temp_seq) == 1: # Only one sample left, use it for validation, none for test
        X_val_seq, y_val_seq = X_temp_seq, y_temp_seq
        X_test_seq, y_test_seq = np.array([]), np.array([]) # Empty test set
    else: # No data left for validation or test
         X_val_seq, X_test_seq, y_val_seq, y_test_seq = np.array([]), np.array([]), np.array([]), np.array([])


    if len(X_train_seq) == 0 or len(X_val_seq) == 0: # Critical: Need train and val for tuner
        print(f"Error: Not enough data for train/validation split. Train: {len(X_train_seq)}, Val: {len(X_val_seq)}. Skipping.")
        return {
            "model_type": model_type,
            "look_back_period": look_back_period,
            "lasso_alpha": lasso_alpha,
            "use_differencing": config_params.get('use_differencing'),
            "status": "failed_split_insufficient_data",
            "best_hyperparameters": None,
            "test_metrics": None,
            "duration_seconds": time.time() - start_time_config
        }

    print(f"Data split: X_train_seq: {X_train_seq.shape}, X_val_seq: {X_val_seq.shape}, X_test_seq: {X_test_seq.shape}")

    # 3. KerasTuner Setup
    # Model input shape is derived from the training sequences: (timesteps, features_in_sequence)
    # X_train_seq.shape[1] is look_back_period, X_train_seq.shape[2] is num features in preprocessed_data_df.
    current_input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    print(f"Model input shape: {current_input_shape}")

    # Wrapper function to build the model, capturing necessary variables in its closure.
    def build_hypermodel_wrapper(hp):
        if model_type == 'att_lstm':
            model_builder_instance = ATTLSTMModel(
                input_shape=current_input_shape,
                look_back=look_back_period,
                random_seed=SEED # Pass seed for model's internal weight initializations
            )
        elif model_type == 'standard_lstm':
            model_builder_instance = StandardLSTMModel(
                input_shape=current_input_shape,
                look_back=look_back_period,
                random_seed=SEED
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        # The build_model method of the instance will use the hp object
        return model_builder_instance.build_model(hp)

    # Define a unique directory for KerasTuner to store logs for this specific configuration.
    # Suffix for differencing option in directory name
    diff_suffix = "_diff" if config_params.get('use_differencing') else "_nodiff"
    tuner_log_dir_name = f"lb_{look_back_period}_alpha_{str(lasso_alpha).replace('.', 'p')}{diff_suffix}"
    tuner_directory = os.path.join(BASE_PROJECT_DIR, model_type, tuner_log_dir_name)

    # Initialize KerasTuner (Hyperband)
    tuner = kt.Hyperband(
        hypermodel=build_hypermodel_wrapper,
        objective='val_loss', # Objective to minimize
        max_epochs=config_params['max_epochs_hyperband'],
        factor=3, # Reduction factor for Hyperband
        hyperband_iterations=config_params['hyperband_iterations'],
        directory=tuner_directory,
        project_name="tuning_logs", # Subdirectory within tuner_directory for this run's trials
        overwrite=True, # Overwrite previous logs if this exact configuration is run again
        seed=SEED # Seed for KerasTuner's internal operations for reproducibility
    )

    tuner.search_space_summary() # Log the search space being used

    # Define EarlyStopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config_params['early_stopping_patience'],
        restore_best_weights=True, # Restore model weights from the epoch with the best val_loss
        verbose=1
    )

    print(f"Starting KerasTuner search for {model_type} LB={look_back_period} Alpha={lasso_alpha} Diff={config_params.get('use_differencing')}...")
    # Run the hyperparameter search
    tuner.search(
        X_train_seq, y_train_seq,
        epochs=config_params['max_epochs_hyperband'], # Max epochs for a single trial in Hyperband
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stopping_cb],
        batch_size=config_params['tuner_batch_size'], # Batch size for training
        verbose=1 # Verbosity: 1 for progress bar per trial, 2 for per epoch.
    )

    print("\nTuning search complete.")
    best_hps_dict = None
    try:
        # Retrieve the best hyperparameters found
        best_hps_retrieved = tuner.get_best_hyperparameters(num_trials=1)
        if not best_hps_retrieved: # Check if any HPs were found
            print("Error: KerasTuner found no best hyperparameters. Skipping evaluation.")
            return {
                "model_type": model_type,
                "look_back_period": look_back_period,
                "lasso_alpha": lasso_alpha,
                "use_differencing": config_params.get('use_differencing'),
                "status": "failed_tuner_no_best_hps",
                "best_hyperparameters": None,
                "test_metrics": None,
                "duration_seconds": time.time() - start_time_config
            }
        best_hps = best_hps_retrieved[0] # Get the HyperParameters object
        best_hps_dict = best_hps.values # Convert to dictionary for easier logging/storage
        print("\nBest Hyperparameters Found:")
        for param, value in best_hps_dict.items():
            print(f"- {param}: {value}")

        # Build the best model using the retrieved hyperparameters
        best_model = tuner.hypermodel.build(best_hps)
    except Exception as e:
        print(f"Error retrieving best model or HPs from tuner: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model_type": model_type,
            "look_back_period": look_back_period,
            "lasso_alpha": lasso_alpha,
            "use_differencing": config_params.get('use_differencing'),
            "status": f"failed_tuner_exception_getting_hps_or_model: {str(e)}",
            "best_hyperparameters": best_hps_dict, # Log HPs if retrieved before error
            "test_metrics": None,
            "duration_seconds": time.time() - start_time_config
        }

    # 4. Evaluate the best model on the dedicated test set
    test_metrics = {"mse": None, "mae": None, "rmse": None}
    if len(X_test_seq) > 0 and best_model is not None:
        print("\nEvaluating the best model on the test set...")
        predictions_scaled = best_model.predict(X_test_seq).flatten()
        y_test_true_scaled = y_test_seq.flatten()

        test_metrics["mse"] = mean_squared_error(y_test_true_scaled, predictions_scaled)
        test_metrics["mae"] = mean_absolute_error(y_test_true_scaled, predictions_scaled)
        test_metrics["rmse"] = np.sqrt(test_metrics["mse"])
        print(f"Test Set Performance: MSE={test_metrics['mse']:.6f}, MAE={test_metrics['mae']:.6f}, RMSE={test_metrics['rmse']:.6f}")
    elif best_model is None:
        print("Best model could not be built, skipping test set evaluation.")
    else: # No test data
        print("No test data to evaluate on. Test metrics will be null.")

    duration_config = time.time() - start_time_config
    print(f"--- Finished Tuning for: Model={model_type}, LB={look_back_period}, Alpha={lasso_alpha}, Diff={config_params.get('use_differencing')} in {duration_config:.2f}s ---")

    return {
        "model_type": model_type,
        "look_back_period": look_back_period,
        "lasso_alpha": lasso_alpha,
        "use_differencing": config_params.get('use_differencing'),
        "status": "completed" if best_model is not None else "failed_model_build_or_eval",
        "best_hyperparameters": best_hps_dict,
        "test_metrics": test_metrics,
        "duration_seconds": duration_config,
        "tuner_directory": tuner_directory
    }


def main_tuning_orchestrator(
    model_type_to_tune,
    stock_ticker=STOCK_TICKER,
    years_of_data=YEARS_OF_DATA,
    lasso_alphas=LASSO_ALPHAS_TO_TEST,
    look_back_periods=LOOK_BACK_PERIODS_TO_TEST,
    max_epochs_hyperband=MAX_EPOCHS_HYPERBAND,
    hyperband_iterations=HYPERBAND_ITERATIONS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    tuner_batch_size=TUNER_BATCH_SIZE,
    base_output_dir=BASE_PROJECT_DIR,
    use_differencing_options=[False]
):
    """
    Main orchestrator for the hyperparameter tuning process.

    This function iterates through specified model types, differencing options,
    LASSO alpha values, and look-back periods. For each combination:
    1. Data is preprocessed (downloaded, indicators calculated, features selected by LASSO, scaled).
       Caching is utilized by `DataPreprocessor` for raw data and indicators.
    2. `run_one_tuning_configuration` is called to perform hyperparameter search using KerasTuner.
    Results from all configurations are aggregated and saved to JSON files.
    The overall best configuration based on test MSE is identified and printed.

    Args:
        model_type_to_tune (str): Specifies which model(s) to tune.
                                  Options: 'att_lstm', 'standard_lstm', 'both'.
        stock_ticker (str): Stock ticker symbol.
        years_of_data (int): Number of years of historical data for preprocessing.
        lasso_alphas (list of float): LASSO alpha values for feature selection.
        look_back_periods (list of int): Sequence lengths for LSTM models.
        max_epochs_hyperband (int): Max epochs for Hyperband tuner.
        hyperband_iterations (int): Number of Hyperband iterations.
        early_stopping_patience (int): Patience for early stopping in training.
        tuner_batch_size (int): Batch size for training during tuning.
        base_output_dir (str): Root directory to save all tuning logs and summaries.
        use_differencing_options (list of bool): List of differencing options to test (e.g., [False, True]).
    """
    set_seeds(SEED) # Ensure reproducibility for the entire orchestration run
    overall_start_time = time.time()

    # Determine which model types to iterate over
    if model_type_to_tune.lower() == 'both':
        model_types_to_process = ['att_lstm', 'standard_lstm']
    elif model_type_to_tune.lower() in ['att_lstm', 'standard_lstm']:
        model_types_to_process = [model_type_to_tune.lower()]
    else:
        print(f"Warning: Invalid model_type_to_tune '{model_type_to_tune}'. Defaulting to 'att_lstm'.")
        model_types_to_process = ['att_lstm']

    all_run_results = [] # Stores results from all configurations across all model types

    # Ensure the base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Tuning results will be saved in: {os.path.abspath(base_output_dir)}")

    # Loop through each specified model type
    for model_type_iter in model_types_to_process:
        print(f"\n========== Starting Tuning for Model Type: {model_type_iter.upper()} ==========")
        model_specific_summary_path = os.path.join(base_output_dir, f"tuning_summary_{model_type_iter}.json")
        current_model_type_results = [] # Results for the current model_type

        # Loop through differencing options (e.g., use original prices, use differenced prices)
        for use_diff_iter in use_differencing_options:
            print(f"\n===== Processing with Differencing Option: {use_diff_iter} for Model: {model_type_iter.upper()} =====")

            # Data preprocessing is done once per (LASSO alpha + differencing option) combination
            # before iterating through look_back_periods.
            for lasso_val_iter in lasso_alphas:
                print(f"\n--- Preprocessing data with LASSO Alpha: {lasso_val_iter}, Differencing: {use_diff_iter} ---")
                # Initialize DataPreprocessor for the current LASSO alpha and differencing setting
                data_preprocessor = DataPreprocessor(
                    stock_ticker=stock_ticker,
                    years_of_data=years_of_data,
                    random_seed=SEED,
                    lasso_alpha=lasso_val_iter,
                    use_differencing=use_diff_iter # Pass current differencing option
                )

                preprocessed_full_data_df = pd.DataFrame() # Initialize to empty DataFrame
                selected_feature_names_list = []
                try:
                    # Preprocess data: returns scaled_df, target_scaler, selected_features_names, ...
                    preprocessed_full_data_df, _, selected_feature_names_list, _, _ = data_preprocessor.preprocess()

                    if preprocessed_full_data_df.empty:
                        print(f"Warning: Preprocessing for LASSO alpha {lasso_val_iter}, Diff: {use_diff_iter} resulted in empty data. Skipping configurations for this alpha.")
                        # Log this failure for this specific preprocessing attempt
                        failure_entry = {
                            "model_type": model_type_iter, "look_back_period": "N/A",
                            "lasso_alpha": lasso_val_iter, "use_differencing": use_diff_iter,
                            "status": "failed_preprocessing_empty_at_orchestrator",
                            "best_hyperparameters": None, "test_metrics": None, "duration_seconds": 0
                        }
                        current_model_type_results.append(failure_entry)
                        all_run_results.append(failure_entry)
                        continue # Move to the next LASSO alpha value

                    num_selected_features = len(selected_feature_names_list)
                    print(f"Data preprocessed for LASSO alpha {lasso_val_iter}, Diff: {use_diff_iter}. Shape: {preprocessed_full_data_df.shape}. Selected X features: {num_selected_features}")

                except Exception as e_preprocess:
                    print(f"CRITICAL ERROR during data preprocessing for LASSO alpha {lasso_val_iter}, Diff: {use_diff_iter}: {e_preprocess}")
                    import traceback
                    traceback.print_exc()
                    failure_entry = {
                        "model_type": model_type_iter, "look_back_period": "N/A",
                        "lasso_alpha": lasso_val_iter, "use_differencing": use_diff_iter,
                        "status": f"failed_preprocessing_exception_at_orchestrator: {str(e_preprocess)}",
                        "best_hyperparameters": None, "test_metrics": None, "duration_seconds": 0
                    }
                    current_model_type_results.append(failure_entry)
                    all_run_results.append(failure_entry)
                    continue # Move to the next LASSO alpha value

                # Inner loop: Iterate through specified look-back periods
                for look_back_val_iter in look_back_periods:
                    # Prepare configuration parameters for the tuning run
                    current_config_params = {
                        'max_epochs_hyperband': max_epochs_hyperband,
                        'hyperband_iterations': hyperband_iterations,
                        'early_stopping_patience': early_stopping_patience,
                        'tuner_batch_size': tuner_batch_size,
                        'use_differencing': use_diff_iter # Pass differencing status to run_one_tuning
                    }

                    # Call the function to run tuning for this single, specific configuration
                    # Pass a copy of the preprocessed data to avoid modifications across runs
                    # num_features argument is not strictly needed by run_one_tuning_configuration as it derives from X_seq
                    single_run_result = run_one_tuning_configuration(
                        model_type=model_type_iter,
                        look_back_period=look_back_val_iter,
                        lasso_alpha=lasso_val_iter,
                        preprocessed_data_df=preprocessed_full_data_df.copy(),
                        config_params=current_config_params
                    )
                    # single_run_result["use_differencing"] is already added inside run_one_tuning_configuration

                    current_model_type_results.append(single_run_result)
                    all_run_results.append(single_run_result)

                    # Save intermediate summary for the current model type after each configuration
                    try:
                        with open(model_specific_summary_path, 'w') as f_json:
                            # Use default=str for any non-serializable objects (like numpy types sometimes)
                            json.dump(current_model_type_results, f_json, indent=4, default=str)
                        print(f"Intermediate summary for {model_type_iter} saved to {model_specific_summary_path}")
                    except Exception as e_json_save:
                        print(f"Error saving intermediate summary for {model_type_iter}: {e_json_save}")

        print(f"========== Finished All LASSO/LookBack Configurations for Model Type: {model_type_iter.upper()} ==========")
        if not current_model_type_results:
            print(f"No results were generated for model type {model_type_iter}.")
        else:
            # Final save for this model type's complete summary
            try:
                with open(model_specific_summary_path, 'w') as f_json:
                    json.dump(current_model_type_results, f_json, indent=4, default=str)
                print(f"Final tuning summary for {model_type_iter} saved to {model_specific_summary_path}")
            except Exception as e_json_save_final:
                print(f"Error saving final summary for {model_type_iter}: {e_json_save_final}")

    # Save overall summary including results from all model types and configurations
    overall_summary_filename = os.path.join(base_output_dir, "tuning_summary_ALL_CONFIGS.json")
    try:
        with open(overall_summary_filename, 'w') as f_json_all:
            json.dump(all_run_results, f_json_all, indent=4, default=str)
        print(f"\nOverall tuning summary for all configurations saved to {overall_summary_filename}")
    except Exception as e_json_save_overall:
        print(f"Error saving overall summary: {e_json_save_overall}")

    # Identify and print the overall best configuration from all successful runs
    if all_run_results:
        # Filter for runs that completed successfully and have valid test metrics
        successful_runs = [
            r for r in all_run_results
            if r.get("status") == "completed" and
               r.get("test_metrics") and
               r.get("test_metrics", {}).get("mse") is not None
        ]
        if successful_runs:
            # Sort by test MSE (lower is better)
            best_overall_run = min(successful_runs, key=lambda r: r["test_metrics"]["mse"])
            print("\n--- Overall Best Configuration Found Across All Runs ---")
            print(f"  Model Type: {best_overall_run['model_type']}")
            print(f"  Look Back Period: {best_overall_run['look_back_period']}")
            print(f"  LASSO Alpha: {best_overall_run['lasso_alpha']}")
            print(f"  Use Differencing: {best_overall_run.get('use_differencing', 'N/A')}") # get() for safety
            print(f"  Test MSE: {best_overall_run['test_metrics']['mse']:.6f}")
            print(f"  Test MAE: {best_overall_run['test_metrics']['mae']:.6f}")
            print(f"  Best Hyperparameters: {best_overall_run['best_hyperparameters']}")
            print(f"  Tuner logs at: {best_overall_run.get('tuner_directory', 'N/A')}")
        else:
            print("\nNo successful runs completed with valid test metrics to determine an overall best configuration.")
    else:
        print("\nNo results were generated from any tuning configuration.")

    total_orchestration_duration = time.time() - overall_start_time
    print(f"\n--- Main Tuning Orchestrator Finished in {total_orchestration_duration // 3600:.0f}h "
          f"{(total_orchestration_duration % 3600) // 60:.0f}m {total_orchestration_duration % 60:.2f}s ---")


if __name__ == '__main__':
    print(f"--- Starting Full Hyperparameter Tuning Orchestration Script ---")
    # Log library versions for reproducibility and debugging
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"KerasTuner Version: {kt.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    # Check for GPU availability
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        print(f"Physical GPUs Available: {len(physical_gpus)}. Details: {physical_gpus}")
        try:
            for gpu in physical_gpus: # Attempt to set memory growth to avoid OOM issues on some setups
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set for GPUs.")
        except RuntimeError as e_gpu:
            print(f"Could not set memory growth for GPUs: {e_gpu} (might be already initialized or not supported).")
    else:
        print("No Physical GPUs Available. TensorFlow will use CPU.")

    # --- Configuration for the Current Tuning Run ---
    # Specify which model(s) to tune: 'att_lstm', 'standard_lstm', or 'both'.
    MODEL_TO_TUNE = 'att_lstm'
    # Example: MODEL_TO_TUNE = 'standard_lstm'
    # Example: MODEL_TO_TUNE = 'both'

    # Specify whether to test with differenced data, original data, or both.
    # Each boolean in the list will trigger a full preprocessing and tuning cycle for that option.
    USE_DIFFERENCING_OPTIONS = [False] # E.g., only use non-differenced data
    # Example: USE_DIFFERENCING_OPTIONS = [True]  # E.g., only use differenced data
    # Example: USE_DIFFERENCING_OPTIONS = [False, True] # Test both options

    # Call the main orchestrator function with the desired configurations
    main_tuning_orchestrator(
        model_type_to_tune=MODEL_TO_TUNE,
        stock_ticker=STOCK_TICKER,
        years_of_data=YEARS_OF_DATA, # Uses global config
        lasso_alphas=LASSO_ALPHAS_TO_TEST, # Uses global config
        look_back_periods=LOOK_BACK_PERIODS_TO_TEST, # Uses global config
        max_epochs_hyperband=MAX_EPOCHS_HYPERBAND, # Uses global config
        hyperband_iterations=HYPERBAND_ITERATIONS, # Uses global config
        early_stopping_patience=EARLY_STOPPING_PATIENCE, # Uses global config
        tuner_batch_size=TUNER_BATCH_SIZE, # Uses global config
        base_output_dir=BASE_PROJECT_DIR, # Uses global config
        use_differencing_options=USE_DIFFERENCING_OPTIONS
    )

    print("\n--- Orchestration Script Finished. Check console output and result files. ---")
