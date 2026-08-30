import pandas as pd
import numpy as np
import torch
from backtesting import Backtest, Strategy
import joblib

from alphabarrier.transformer import (
    DataConfig,
    load_and_engineer,
    TransformerClassifier,
    get_device,
    get_feature_columns,
)

device = get_device()

MODEL_FILE = "./artifacts/final_transformer_model.pt"
SCALER_FILE = "./artifacts/final_scaler.pkl"
DIRECTION_CLASSIFIER_FILE = "./artifacts/direction_classifier.pkl"
DATA_PATH = "./training_data/kraken"
INITIAL_CASH = 10_000_000  # avoids integer-BTC order rejection at six-figure prices
COMMISSION_PCT = 0.001  # 0.1% commission for entry and exit

def prepare_data_for_backtest(
    model_path,
    data_path=DATA_PATH,
    scaler_path=SCALER_FILE,
    direction_classifier_path=DIRECTION_CLASSIFIER_FILE,
):
    """
    Loads data, generates model predictions, and formats the output
    for the backtesting.py library.
    """
    print("Loading model metadata...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(f"Error: model file not found at {model_path}.")
        return pd.DataFrame()

    if "state_dict" not in checkpoint:
        raise ValueError(
            "This checkpoint predates artifact metadata. Retrain with the current training pipeline."
        )
    data_config = DataConfig(**checkpoint["data_config"])
    feature_cols = checkpoint["feature_cols"]

    print("Loading and engineering causal features for backtest...")
    df = load_and_engineer(data_path, include_target=False, data_config=data_config)
    missing_features = sorted(set(feature_cols) - set(get_feature_columns(df)))
    if missing_features:
        raise ValueError(f"Backtest data is missing trained features: {missing_features}")

    # --- Scaling ---
    print(f"Loading scaler from {scaler_path}...")
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"Error: scaler file not found at {scaler_path}.")
        return pd.DataFrame()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    print("Loading pre-trained model...")
    model = TransformerClassifier(
        n_features=len(feature_cols),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Successfully loaded model from {model_path}")
    model.eval()

    print("Creating input sequences for the model...")
    feature_data = df_scaled[feature_cols].values
    sequence_length = data_config.sequence_length
    num_sequences = len(df) - sequence_length + 1
    X = np.empty(
        (num_sequences, sequence_length, len(feature_cols)), dtype=np.float32
    )

    for i in range(num_sequences):
        X[i] = feature_data[i : i + sequence_length]

    print("Generating model predictions...")
    probability_batches = []
    BATCH_SIZE = 256
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(X[i : i + BATCH_SIZE]).to(device)
            logits = model(batch)
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())

    probabilities = np.concatenate(probability_batches)
    predictions = probabilities.argmax(axis=1)
    direction_feature_cols = checkpoint.get("direction_feature_cols")
    ensemble_weight = checkpoint.get("direction_ensemble_transformer_weight")
    if direction_feature_cols is not None and ensemble_weight is not None:
        try:
            direction_classifier = joblib.load(direction_classifier_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Hybrid checkpoint requires {direction_classifier_path}"
            ) from error
        signal_rows = df_scaled.iloc[sequence_length - 1 :]
        direction_probability_up = direction_classifier.predict_proba(
            signal_rows[direction_feature_cols]
        )[:, 1]
        transformer_probability_up = probabilities[:, 2] / (
            probabilities[:, 0] + probabilities[:, 2]
        )
        hybrid_probability_up = (
            ensemble_weight * transformer_probability_up
            + (1.0 - ensemble_weight) * direction_probability_up
        )
        predictions = np.where(hybrid_probability_up >= 0.5, 2, 0)
        print(
            "Using hybrid direction signal: "
            f"{1.0 - ensemble_weight:.0%} direction head / "
            f"{ensemble_weight:.0%} Transformer"
        )

    # Prepare the final dataframe for the backtesting library
    # We use the original, unscaled dataframe `df` for backtesting prices
    df_with_preds = df.iloc[sequence_length - 1 :].copy()
    assert len(df_with_preds) == len(
        predictions
    ), "Mismatch between predictions and dataframe length!"
    df_with_preds["Signal"] = predictions

    # A trained checkpoint contains the untouched evaluation boundary. Keeping
    # only that period prevents the displayed backtest from mixing in-sample
    # returns into an out-of-sample claim.
    holdout_start = checkpoint.get("holdout_start")
    if holdout_start is not None:
        df_with_preds = df_with_preds.loc[pd.Timestamp(holdout_start) :]

    atr = df_with_preds["XBTUSD_feature_atr_pct"] * df_with_preds["XBTUSD_close"]
    profit_distance = data_config.profit_take_atr_mult * atr
    stop_distance = data_config.stop_loss_atr_mult * atr

    # Map the raw, unscaled columns to the required names for the backtester
    bt_df = pd.DataFrame(
        {
            "Open": df_with_preds["XBTUSD_open"],
            "High": df_with_preds["XBTUSD_high"],
            "Low": df_with_preds["XBTUSD_low"],
            "Close": df_with_preds["XBTUSD_close"],
            "Volume": df_with_preds["XBTUSD_volume"],
            "Signal": df_with_preds["Signal"],  # the model's prediction
            "LongStop": df_with_preds["XBTUSD_close"] - stop_distance,
            "LongProfit": df_with_preds["XBTUSD_close"] + profit_distance,
            "ShortStop": df_with_preds["XBTUSD_close"] + stop_distance,
            "ShortProfit": df_with_preds["XBTUSD_close"] - profit_distance,
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
                self.buy(sl=self.data.LongStop[-1], tp=self.data.LongProfit[-1])

            # stable short signal, 'DOWN'
            elif (
                not self.position
                and np.all(recent_signals == self.short_entry_signal)
            ):
                self.sell(sl=self.data.ShortStop[-1], tp=self.data.ShortProfit[-1])


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
