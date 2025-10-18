import pandas as pd
import numpy as np
import torch
from backtesting import Backtest, Strategy
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

from alphabarrier.transformer import (
    load_and_engineer,
    TransformerClassifier,
    SEQUENCE_LENGTH,
    NHEAD,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
)

device = torch.device(
    "mps"
    if torch.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_FILE = "./final_transformer_model.pt"
SCALER_FILE = "./final_scaler.pkl"
DATA_FILE_1M = "./training_data/XBTUSD_1.csv"
INITIAL_CASH = 100_000
COMMISSION_PCT = 0.001  # 0.1% commission for entry and exit

def prepare_data_for_backtest(model_path, data_path="./training_data", scaler_path="scaler.pkl"):
    """
    Loads data, generates model predictions, and formats the output
    for the backtesting.py library.
    """
    print("Loading and engineering features for backtest...")
    df = load_and_engineer(data_path)

    feature_cols = [
        col
        for col in df.columns
        if (
            "RSI" in col
            or "ATR" in col
            or "OBV" in col
            or "MACD" in col
            or "ADX" in col
            or "log_return" in col
            or "momentum" in col
            or "volume_" in col
            or "price_vwap" in col
            or "bbands_ratio" in col
            or "volatility_" in col
            or "CDL_" in col
            or "ratio" in col
            or "halving" in col
            or "_sin" in col
            or "_cos" in col
            or "resistance" in col
            or "support" in col
        )
    ]

    # --- Scaling ---
    print(f"Loading scaler from {scaler_path}...")
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_path}. Setting standard scaler.")
        scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    print("Loading pre-trained model...")
    model = TransformerClassifier(
        n_features=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        dropout=DROPOUT,
        n_classes=3,
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print("Please ensure you have a trained model file.")
        return pd.DataFrame()  # Return empty DataFrame if model not found
    model.eval()

    print("Creating input sequences for the model...")
    feature_data = df_scaled[feature_cols].values
    num_sequences = len(df) - SEQUENCE_LENGTH
    X = np.empty(
        (num_sequences, SEQUENCE_LENGTH, len(feature_cols)), dtype=np.float32
    )

    for i in tqdm(range(num_sequences), desc="Creating sequences"):
        X[i] = feature_data[i : i + SEQUENCE_LENGTH]

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    print("Generating model predictions...")
    predictions = []
    BATCH_SIZE = 256
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), BATCH_SIZE), desc="Model Inference"):
            batch = X_tensor[i : i + BATCH_SIZE]
            batch_preds = model(batch)
            # Get the predicted class (0, 1, or 2)
            predictions.append(torch.argmax(batch_preds, dim=1).cpu().numpy())

    predictions = np.concatenate(predictions)

    # Prepare the final dataframe for the backtesting library
    # We use the original, unscaled dataframe `df` for backtesting prices
    df_with_preds = df.iloc[SEQUENCE_LENGTH:].copy()
    assert len(df_with_preds) == len(
        predictions
    ), "Mismatch between predictions and dataframe length!"
    df_with_preds["Signal"] = predictions

    # Map the raw, unscaled columns to the required names for the backtester
    bt_df = pd.DataFrame(
        {
            "Open": df_with_preds["XBTUSD_open"],
            "High": df_with_preds["XBTUSD_high"],
            "Low": df_with_preds["XBTUSD_low"],
            "Close": df_with_preds["XBTUSD_close"],
            "Volume": df_with_preds["XBTUSD_volume"],
            "Signal": df_with_preds["Signal"],  # the model's prediction
        },
        index=df_with_preds.index,
    )

    return bt_df


def identity(series):
    """A simple function that returns the input series. Used with the I() wrapper."""
    return series


class TransformerStrategy(Strategy):
    # --- Strategy Parameters ---
    long_entry_signal = 2  # Corresponds to 'UP' (class 2)
    short_entry_signal = 0  # Corresponds to 'DOWN' (class 0)
    exit_signal = 1  # Corresponds to 'SIDEWAYS' (class 1)

    # --- Triple Barrier Parameters (for stop-loss/take-profit) ---
    stop_loss_pct = 0.03
    profit_take_pct = 0.06

    # --- Signal smoothing window ---
    consistency_window = 3  # require same signal for last 3 bars

    def init(self):
        """
        Initialize the strategy.
        We use the I() function to get the raw 'Signal' values from our data.
        """
        self.signal = self.I(identity, self.data.Signal)
        print("Transformer Strategy Initialized.")
        print(f"Long entry on signal: {self.long_entry_signal}")
        print(f"Short entry on signal: {self.short_entry_signal}")
        print(f"Position exit on signal: {self.exit_signal}")

    def next(self):
        """
        Defines the trading logic for each bar (timestep).
        """
        price = self.data.Close[-1]
        current_signal = self.signal[-1]

        # --- Position Management ---
        # If we have an open position, check for exit signals
        if self.position:
            # Exit if model predicts 'SIDEWAYS' or reverses its signal
            if (self.position.is_long and current_signal != self.long_entry_signal) or (
                self.position.is_short and current_signal != self.short_entry_signal
            ):
                self.position.close()

        if len(self.signal) >= self.consistency_window:
            recent_signals = self.signal[-self.consistency_window:]

            # stable long singal, 'UP'
            if (
                not self.position
                and np.all(recent_signals == self.long_entry_signal)
            ):
                sl = price * (1 - self.stop_loss_pct)
                tp = price * (1 + self.profit_take_pct)
                self.buy(sl=sl, tp=tp)

            # stable short signal, 'DOWN'
            elif (
                not self.position
                and np.all(recent_signals == self.short_entry_signal)
            ):
                sl = price * (1 + self.stop_loss_pct)
                tp = price * (1 - self.profit_take_pct)
                self.sell(sl=sl, tp=tp)


if __name__ == "__main__":
    print(f"Using device: {device}")

    bt_df = prepare_data_for_backtest(MODEL_FILE, scaler_path=SCALER_FILE)

    if bt_df.empty:
        print("\nBacktest cannot run. Please check for errors above.")
    else:
        print("\nStarting Backtest...")
        bt = Backtest(
            bt_df,
            TransformerStrategy,
            cash=INITIAL_CASH,
            commission=COMMISSION_PCT,
            trade_on_close=True,
            exclusive_orders=True,
        )

        stats = bt.run()

        print("\n--- Backtest Results ---")
        print(stats)

        print("\n--- Strategy Parameters ---")
        print(stats["_strategy"])

        print("\nGenerating plot...")
        bt.plot()