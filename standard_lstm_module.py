import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# It's good practice to import any necessary sibling modules if needed,
# but for a standalone LSTM model, it might only need DataPreprocessor for its own testing.
# For now, let's assume DataPreprocessor would be used similarly to how it's used in att_lstm_module.py's main block.
from data_preprocessing_module import DataPreprocessor


class StandardLSTMModel:
    def __init__(self, input_shape, look_back=60, random_seed=42, model_params=None):
        self.input_shape = input_shape  # (timesteps, features)
        self.look_back = look_back
        self.model = None
        self.random_seed = random_seed
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.model_params = model_params if model_params else {}

        # Default parameters, can be overridden by model_params
        self.lstm_units_1 = self.model_params.get('lstm_units_1', 64)
        self.lstm_units_2 = self.model_params.get('lstm_units_2', 32) # For a potential second LSTM layer
        self.num_lstm_layers = self.model_params.get('num_lstm_layers', 1)
        self.dense_units_1 = self.model_params.get('dense_units_1', 32)
        self.num_dense_layers = self.model_params.get('num_dense_layers', 1) # Only one dense layer by default for standard LSTM
        self.learning_rate = self.model_params.get('learning_rate', 0.001)
        self.dropout_rate_lstm = self.model_params.get('dropout_rate_lstm', 0.2)
        self.dropout_rate_dense = self.model_params.get('dropout_rate_dense', 0.2)
        self.activation_dense = self.model_params.get('activation_dense', 'relu')

    def build_model(self, hp=None): # Accept hp object for potential tuning
        if hp: # Use hyperparameters from tuner if provided
            # Learning Rate
            learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

            # LSTM Layer Configuration
            num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=4, step=1) # Max 3-4 as per plan
            lstm_units_list = [] # Renamed to avoid conflict with potential hp name 'lstm_units'
            for i in range(num_lstm_layers):
                max_units = 512 if i == 0 else (256 if i == 1 else (128 if i == 2 else 64))
                min_units = 32
                step_units = 32
                current_max_units = max(min_units, max_units)
                lstm_units_list.append(hp.Int(f'lstm_units_{i+1}', min_value=min_units, max_value=current_max_units, step=step_units))

            dropout_rate_lstm = hp.Float('dropout_rate_lstm', min_value=0.0, max_value=0.5, step=0.1)

            # Dense Layer Configuration
            num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, step=1)
            dense_units_list = [] # Renamed
            for i in range(num_dense_layers):
                max_units_dense = 256 if i == 0 else (128 if i == 1 else 64)
                min_units_dense = 32
                step_units_dense = 32
                current_max_dense_units = max(min_units_dense, max_units_dense)
                dense_units_list.append(hp.Int(f'dense_units_{i+1}', min_value=min_units_dense, max_value=current_max_dense_units, step=step_units_dense))

            dropout_rate_dense = hp.Float('dropout_rate_dense', min_value=0.0, max_value=0.5, step=0.1)
            activation_dense = hp.Choice('activation_dense', values=['relu', 'tanh', 'elu', 'swish'])

        else: # Use instance attributes (from self.model_params or their defaults)
            learning_rate = self.learning_rate
            num_lstm_layers = self.num_lstm_layers
            lstm_units_list = [self.lstm_units_1]
            if num_lstm_layers > 1: lstm_units_list.append(self.model_params.get('lstm_units_2', 64))
            if num_lstm_layers > 2: lstm_units_list.append(self.model_params.get('lstm_units_3', 32))
            if num_lstm_layers > 3: lstm_units_list.append(self.model_params.get('lstm_units_4', 32))

            dropout_rate_lstm = self.dropout_rate_lstm

            num_dense_layers = self.num_dense_layers
            dense_units_list = [self.dense_units_1]
            if num_dense_layers > 1: dense_units_list.append(self.model_params.get('dense_units_2', 32))
            if num_dense_layers > 2: dense_units_list.append(self.model_params.get('dense_units_3', 16))

            dropout_rate_dense = self.dropout_rate_dense
            activation_dense = self.activation_dense

        inputs = Input(shape=self.input_shape)
        x = inputs

        # LSTM layers
        for i in range(num_lstm_layers):
            # For Standard LSTM: only the last LSTM layer should have return_sequences=False
            # All preceding LSTM layers must have return_sequences=True if stacked.
            return_seq = True if i < (num_lstm_layers - 1) else False
            current_lstm_units = lstm_units_list[i] if i < len(lstm_units_list) else lstm_units_list[-1]

            x = LSTM(units=current_lstm_units,
                     return_sequences=return_seq,
                     dropout=dropout_rate_lstm,
                     recurrent_dropout=dropout_rate_lstm)(x)

        # Dense layers for prediction
        for i in range(num_dense_layers):
            current_dense_units = dense_units_list[i] if i < len(dense_units_list) else dense_units_list[-1]
            x = Dense(units=current_dense_units, activation=activation_dense)(x)
            x = Dropout(dropout_rate_dense)(x)

        outputs = Dense(1, name="output_layer")(x) # Output a single value

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

        if not hp:
            self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, early_stopping_patience=10, reduce_lr_patience=5, reduce_lr_factor=0.2):
        if self.model is None:
            print("Warning: self.model is None in train(). Calling build_model() without hp to initialize.")
            self.build_model()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            verbose=1,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            verbose=1,
            min_lr=1e-6
        )

        print("Training Standard LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
            shuffle=False
        )
        print("Standard LSTM model training complete.")
        return history

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not built or trained. Call build_model() and train() first.")
        return self.model.predict(X_test)

# Example Usage (for testing the module)
if __name__ == '__main__':
    print("\n--- StandardLSTMModel Module Test ---")
    # 1. Get preprocessed data
    data_preprocessor = DataPreprocessor(stock_ticker='GOOGL', years_of_data=2, random_seed=42)
    # Using default lasso_alpha and use_differencing for this test
    processed_df, data_scaler, _, _, _ = data_preprocessor.preprocess()

    if processed_df is None or processed_df.empty:
        print("Failed to preprocess data. Aborting StandardLSTMModel test.")
    else:
        print(f"Shape of preprocessed_df: {processed_df.shape}")
        target_column_name = processed_df.columns[-1]
        target_column_index = processed_df.columns.get_loc(target_column_name)
        print(f"Target column for LSTM: {target_column_name} at index {target_column_index}")

        look_back = 60

        def create_sequences_for_test(data_df, target_idx, lb):
            X_data, y_data = [], []
            raw_data = data_df.values
            for i in range(len(raw_data) - lb):
                X_data.append(raw_data[i:(i + lb), :])
                y_data.append(raw_data[i + lb, target_idx])
            return np.array(X_data), np.array(y_data)

        X, y = create_sequences_for_test(processed_df, target_column_index, look_back)

        if X.shape[0] == 0:
            print("Not enough data to create sequences. Aborting.")
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
                print("Not enough data after splitting. Aborting.")
            else:
                input_shape = (X_train.shape[1], X_train.shape[2])

                # Initialize and build model with default parameters from the class
                std_lstm_test_model = StandardLSTMModel(
                    input_shape=input_shape,
                    look_back=look_back,
                    random_seed=42,
                    model_params={'lstm_units_1': 50, 'dense_units_1': 25, 'num_lstm_layers':1, 'num_dense_layers':1} # Smaller units for faster test
                )
                std_lstm_test_model.build_model() # Build with params from constructor
                print("\nStandard LSTM Model Summary (for test):")
                std_lstm_test_model.model.summary()

                print("\nTraining Standard LSTM model for test...")
                history = std_lstm_test_model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=5, # Reduced epochs for test
                    batch_size=16,
                    early_stopping_patience=2,
                    reduce_lr_patience=1
                )

                print("\nEvaluating Standard LSTM model on test set...")
                predictions_scaled = std_lstm_test_model.predict(X_test).flatten()

                from sklearn.metrics import mean_squared_error, mean_absolute_error
                import matplotlib.pyplot as plt
                import os

                mse_scaled = mean_squared_error(y_test, predictions_scaled)
                mae_scaled = mean_absolute_error(y_test, predictions_scaled)
                rmse_scaled = np.sqrt(mse_scaled)

                print("\n--- Standard LSTM Model Performance (Scaled Data) ---")
                print(f"Test MSE (scaled): {mse_scaled:.6f}")
                print(f"Test MAE (scaled): {mae_scaled:.6f}")
                print(f"Test RMSE (scaled): {rmse_scaled:.6f}")

                print("\nGenerating visualizations for Standard LSTM module test...")
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Standard LSTM Model Training and Validation Loss (Module Test)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True)
                loss_plot_filename = "std_lstm_module_loss_curve.png"
                plt.savefig(loss_plot_filename)
                print(f"Loss curve plot saved as {loss_plot_filename} in {os.path.abspath('.')}")
                plt.close()

                sample_size = min(50, len(y_test))
                plt.figure(figsize=(12, 6))
                plt.plot(y_test[:sample_size], label='Actual Values (Scaled)', marker='.')
                plt.plot(predictions_scaled[:sample_size], label='Predicted Values (Scaled)', linestyle='--')
                plt.title(f'Standard LSTM Predictions vs Actuals (First {sample_size} Test Samples - Scaled)')
                plt.xlabel('Sample Index')
                plt.ylabel('Value (Scaled)')
                plt.legend()
                plt.grid(True)
                preds_plot_filename = "std_lstm_module_preds_vs_actuals.png"
                plt.savefig(preds_plot_filename)
                print(f"Predictions vs actuals plot saved as {preds_plot_filename} in {os.path.abspath('.')}")
                plt.close()

    print("\n--- End of StandardLSTMModel Module Test ---")
