import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import pandas_ta as ta
from tqdm.auto import tqdm
import math
from scipy.stats import zscore

# --- Adjustable Parameters & Hyperparameters ---
ENABLED_SYMBOLS = ['XBTUSD']
SEQUENCE_LENGTH = 48
BATCH_SIZE = 128
LR = 2e-4
EPOCHS = 50
HIDDEN_SIZE = 512
NUM_LAYERS = 4
DROPOUT = 0.3
N_SPLITS = 5
NHEAD = 8  # Transformer heads
device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# --- Triple Barrier Method Parameters ---
HORIZON_MULTIPLIER = 2.5  # Dynamic horizon multiplier
PROFIT_TAKE_ATR_MULT = 1.8  # ATR multiplier for profit take
STOP_LOSS_ATR_MULT = 1.2   # ATR multiplier for stop loss

# --- Triple Barrier Labeling ---
def get_triple_barrier_labels(prices, atr_series, horizon_mult, pt_mult, sl_mult):
    """
    Applies dynamic triple barrier method using ATR for barrier thresholds.
    - prices: DataFrame with 'high', 'low', 'close' columns
    - atr_series: Average True Range series
    - horizon_mult: Multiplier for ATR to determine max holding period
    - pt_mult: Profit-take multiplier of ATR
    - sl_mult: Stop-loss multiplier of ATR
    
    Returns a pandas Series with labels:
    - 1: Profit-take hit first
    - -1: Stop-loss hit first
    - 0: Time barrier hit first
    """
    outcomes = pd.Series(index=prices.index, dtype=float)
    
    print("Calculating Dynamic Triple Barrier Labels...")
    for i in tqdm(range(len(prices) - 1)):
        entry_price = prices['close'].iloc[i]
        atr = atr_series.iloc[i]
        
        # Dynamic barrier levels based on ATR
        pt_level = entry_price + (pt_mult * atr)
        sl_level = entry_price - (sl_mult * atr)
        
        # Dynamic horizon based on volatility
        horizon = max(4, min(48, int(horizon_mult * (atr / prices['close'].iloc[i]))))
        if i + horizon >= len(prices):
            horizon = len(prices) - i - 1
            
        path = prices.iloc[i + 1 : i + 1 + horizon]
        
        if len(path) == 0:
            continue
            
        # Check barrier touches
        pt_touched = (path['high'] >= pt_level).any()
        sl_touched = (path['low'] <= sl_level).any()
        
        # Determine first barrier hit
        if pt_touched and sl_touched:
            pt_idx = (path['high'] >= pt_level).idxmax()
            sl_idx = (path['low'] <= sl_level).idxmax()
            outcomes.iloc[i] = 1 if pt_idx <= sl_idx else -1
        elif pt_touched:
            outcomes.iloc[i] = 1
        elif sl_touched:
            outcomes.iloc[i] = -1
        else:
            outcomes.iloc[i] = 0
            
    return outcomes

# --- Feature Engineering ---
def get_days_since_last_halving(timestamp):
    """
    Calculate days since the last Bitcoin halving event.
    """
    halving_dates = [
        pd.Timestamp('2009-01-03'),  # Genesis
        pd.Timestamp('2012-11-28'),  # 1st halving
        pd.Timestamp('2016-07-09'),  # 2nd halving
        pd.Timestamp('2020-05-11'),  # 3rd halving
        pd.Timestamp('2024-04-19'),  # 4th halving
    ]
    
    last_halving = None
    for halving_date in halving_dates:
        if halving_date <= timestamp:
            last_halving = halving_date
        else:
            break
    
    if last_halving is None:
        return 0  # Before first halving
    
    return (timestamp - last_halving).days

def add_support_resistance_features(df):
    """Add multi-timeframe support/resistance"""
    
    # Multiple timeframe pivot detection
    for window in [12, 36, 84, 168]:  # 20h, 50h, 100h, 200h lookbacks
        # Recent highs/lows as key levels
        df[f'resistance_{window}'] = df['high'].rolling(window).max()
        df[f'support_{window}'] = df['low'].rolling(window).min()
        
        # Distance to these levels (normalized)
        df[f'dist_to_resistance_{window}'] = (df[f'resistance_{window}'] - df['close']) / df['close']
        df[f'dist_to_support_{window}'] = (df['close'] - df[f'support_{window}']) / df['close']
        
        # How many times price has tested these levels
        df[f'resistance_tests_{window}'] = df['high'].rolling(window).apply(
            lambda x: sum(x >= x.max() * 0.995)  # Within 0.5% of resistance
        )
        df[f'support_tests_{window}'] = df['low'].rolling(window).apply(
            lambda x: sum(x <= x.min() * 1.005)  # Within 0.5% of support  
        )
    
    # Volume-weighted levels (stronger S/R)
    for window in [50, 100]:
        # Volume-weighted average price at recent highs/lows
        high_volume = df['volume'] * (df['high'] == df['high'].rolling(window).max())
        low_volume = df['volume'] * (df['low'] == df['low'].rolling(window).min())
        
        df[f'volume_weighted_resistance_{window}'] = (
            (df['high'] * high_volume).rolling(window).sum() / 
            high_volume.rolling(window).sum()
        ).fillna(df['high'])
        
        df[f'volume_weighted_support_{window}'] = (
            (df['low'] * low_volume).rolling(window).sum() / 
            low_volume.rolling(window).sum()
        ).fillna(df['low'])
    
    return df

def add_technical_features(df):
    """Add core technical features with multi-timeframe S/R"""
    # Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.cci(length=20, append=True)
    df.ta.mom(length=10, append=True)

    # Trend
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True) 
    df.ta.sma(length=50, append=True)
    df.ta.adx(length=14, append=True)

    # Volatility
    df.ta.bbands(length=20, std=2.0, append=True)
    df.ta.atr(length=14, append=True)

    # Volume
    df.ta.obv(append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.vwap(anchor="D", append=True)
    
    # Candlestick patterns
    df.ta.cdl_pattern(name="all", append=True)
    
    # Add multi-timefrime S/R features
    df = add_support_resistance_features(df)
    
    # Volatility features
    df['volatility_ratio'] = df['ATRr_14'] / df['close'].rolling(50).std()
    df['volatility_regime'] = df['ATRr_14'].rolling(100).rank(pct=True)
    
    # Momentum features
    for lag in [1, 3, 6, 12, 24]:
        df[f'log_return_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
        df[f'momentum_{lag}'] = df['close'] / df['close'].shift(lag) - 1
        
    # Volume features
    df['volume_z'] = zscore(df['volume'])
    df['volume_spike'] = (df['volume'] / df['volume'].rolling(20).mean()).clip(0, 5)

    # Price relationships
    df['price_vwap_delta'] = (df['close'] - df['VWAP_D']) / df['VWAP_D']
    df['bbands_ratio'] = (df['close'] - df['BBL_20_2.0_2.0']) / (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0'])
    
    return df

# --- Data Loading & Feature Engineering ---
def load_and_engineer(path_dir):
    dfs = []
    for sym in ENABLED_SYMBOLS:
        path = os.path.join(path_dir, f"{sym}_240.csv")
        df = pd.read_csv(
            path,
            names=['time', 'open', 'high', 'low', 'close', 'volume', 'trades'],
            converters={'time': lambda x: pd.to_datetime(pd.to_numeric(x), unit='s')},
            index_col='time'
        )
        
        # Add technical features
        df = add_technical_features(df)
        df = df.add_prefix(f"{sym}_")
        dfs.append(df)

    combined = pd.concat(dfs, axis=1, join='inner')
    
    # Bitcoin halving cycle feature
    days_since_halving = combined.index.map(get_days_since_last_halving)
    combined['halving_cycle_progress'] = (days_since_halving % 1461) / 1461  # Normalized 4-year cycle
    combined['halving_cycle_sin'] = np.sin(2 * np.pi * combined['halving_cycle_progress'])
    combined['halving_cycle_cos'] = np.cos(2 * np.pi * combined['halving_cycle_progress'])
    
    # Time features
    idx = combined.index
    for period, name in zip([24, 7, 12], ['hour', 'day_of_week', 'month']):
        angle = 2 * np.pi * getattr(idx, name) / period
        combined[f'{name}_sin'] = np.sin(angle)
        combined[f'{name}_cos'] = np.cos(angle)
    
    # clean data
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)
    
    # get target variable using Triple Barrier Method
    price_data = combined[['XBTUSD_high', 'XBTUSD_low', 'XBTUSD_close']].rename(
        columns={'XBTUSD_high': 'high', 'XBTUSD_low': 'low', 'XBTUSD_close': 'close'}
    )
    atr_series = combined['XBTUSD_ATRr_14']
    
    labels = get_triple_barrier_labels(
        price_data, 
        atr_series,
        HORIZON_MULTIPLIER,
        PROFIT_TAKE_ATR_MULT,
        STOP_LOSS_ATR_MULT
    )
    
    # Map labels to classes
    label_mapping = {-1: 0, 0: 1, 1: 2}
    combined['target'] = labels.map(label_mapping)

    # Final cleanup
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)
    
    print("\nTarget Class Distribution:")
    print(combined['target'].value_counts(normalize=True))
    
    return combined

# --- Dataset Class ---
class TimeSeriesDataset(Dataset):
    def __init__(self, df, features, target_col, seq_len):
        self.X = df[features].values
        self.y = df[target_col].values
        self.seq_len = seq_len
        self.indices = [i for i in range(seq_len, len(self.X))]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        start_idx = actual_idx - self.seq_len
        x = self.X[start_idx:actual_idx]
        y = self.y[actual_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- Transformer Model ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, nhead, dropout, n_classes=3):
        super().__init__()
        self.embedding = nn.Linear(n_features, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, n_classes)
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        # Input shape: (batch_size, seq_len, n_features)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Take last timestep
        return self.fc(x)

# --- Training Functions ---
def train_loop(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        total_loss += criterion(preds, yb).item()
    return total_loss / len(loader)

@torch.no_grad()
def calculate_metrics(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb)
        preds = torch.argmax(preds, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['DOWN', 'SIDEWAYS', 'UP'], zero_division=0)
    return accuracy, report

# --- Main Execution ---
if __name__ == '__main__':
    # Load data
    df = load_and_engineer('./training_data')

    # Feature selection
    feature_cols = [col for col in df.columns if (
        'RSI' in col or 
        'ATR' in col or 
        'OBV' in col or 
        'MACD' in col or 
        'ADX' in col or 
        'log_return' in col or 
        'momentum' in col or 
        'volume_' in col or 
        'price_vwap' in col or 
        'bbands_ratio' in col or 
        'volatility_' in col or 
        'CDL_' in col or 
        'ratio' in col or 
        'halving' in col or 
        '_sin' in col or 
        '_cos' in col or
        'resistance' in col or
        'support' in col
    )]
    
    target_col = 'target'
    close_prices = df['XBTUSD_close']
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []
    
    print(f"--- Starting Time Series Cross-Validation with {N_SPLITS} Folds ---")
    
    # print(f"features found: {feature_cols}")
    print(f"Total features selected: {len(feature_cols)}")
    
    for fold, (train_index, val_index) in enumerate(tscv.split(df)):
        print(f"\n===== FOLD {fold + 1}/{N_SPLITS} =====")
        
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        
        # Scaling with leakage prevention
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_df[feature_cols])
        val_features = scaler.transform(val_df[feature_cols])
        
        # Create DataFrames with scaled features
        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()
        train_df_scaled.loc[:, feature_cols] = train_features
        val_df_scaled.loc[:, feature_cols] = val_features.astype(np.float64)
        
        # Create datasets
        train_ds = TimeSeriesDataset(train_df_scaled, feature_cols, target_col, SEQUENCE_LENGTH)
        val_ds = TimeSeriesDataset(val_df_scaled, feature_cols, target_col, SEQUENCE_LENGTH)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        
        # Model setup
        model = TransformerClassifier(
            n_features=len(feature_cols),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            nhead=NHEAD,
            dropout=DROPOUT,
            n_classes=3
        ).to(device)
        
        # Class-balanced loss
        counts = train_df['target'].value_counts().sort_index()
        class_weights = torch.tensor([1.0/np.sqrt(counts[i]) for i in range(3)], dtype=torch.float).to(device)
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 7
        wait = 0
        best_model_path = f'best_transformer_fold_{fold+1}.pt'
        
        print(f"Training with class weights: {class_weights.cpu().numpy()}")
        for epoch in range(1, EPOCHS+1):
            train_loss = train_loop(model, train_loader, criterion, optimizer)
            val_loss = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(best_model_path))
        accuracy, report = calculate_metrics(model, val_loader)
        
        print(f"\nFold {fold+1} Results:")
        print(f"Accuracy: {accuracy:.2%}")
        print("Classification Report:\n", report)
        
        fold_results.append({
            'accuracy': accuracy,
            'report': report
        })
    
    # Final summary
    print("\n===== CROSS-VALIDATION SUMMARY =====")
    avg_acc = np.mean([res['accuracy'] for res in fold_results])
    print(f"Average Accuracy: {avg_acc:.2%}")

    print("\n\n===== Starting Final Model Training on Full Dataset =====")

    # Split data into a final training and validation set, most recent data for validation.
    val_split_idx = int(len(df) * 0.9)
    train_final_df = df.iloc[:val_split_idx]
    val_final_df = df.iloc[val_split_idx:]
    print(f"Final training set size: {len(train_final_df)}")
    print(f"Final validation set size: {len(val_final_df)}")

    print("Preparing final datasets and scaler...")
    final_scaler = StandardScaler()
    train_final_df_scaled = train_final_df.copy()
    train_final_df_scaled.loc[:, feature_cols] = final_scaler.fit_transform(train_final_df[feature_cols])

    val_final_df_scaled = val_final_df.copy()
    val_final_df_scaled.loc[:, feature_cols] = final_scaler.transform(val_final_df[feature_cols])

    joblib.dump(final_scaler, 'final_scaler.pkl')
    print("Final scaler saved to 'final_scaler.pkl'")

    # Create Datasets and DataLoaders
    full_train_ds = TimeSeriesDataset(train_final_df_scaled, feature_cols, target_col, SEQUENCE_LENGTH)
    final_val_ds = TimeSeriesDataset(val_final_df_scaled, feature_cols, target_col, SEQUENCE_LENGTH)

    full_train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    final_val_loader = DataLoader(final_val_ds, batch_size=BATCH_SIZE)


    # Initialize a new model, optimizer, and loss function
    print("Initializing a new model for final training...")
    final_model = TransformerClassifier(
        n_features=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        dropout=DROPOUT,
        n_classes=3
    ).to(device)

    # Use class weights calculated on the final training portion
    final_train_counts = train_final_df['target'].value_counts().sort_index()
    final_class_weights = torch.tensor([1.0 / np.sqrt(final_train_counts.get(i, 1)) for i in range(3)], dtype=torch.float).to(device)
    final_class_weights = final_class_weights / final_class_weights.sum()
    final_criterion = nn.CrossEntropyLoss(weight=final_class_weights)

    final_optimizer = optim.AdamW(final_model.parameters(), lr=LR, weight_decay=1e-5)
    final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, 'min', patience=3, factor=0.5)

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 7
    wait = 0
    final_model_path = 'final_transformer_model.pt'
    
    print(f"Training final model for up to {EPOCHS} epochs with early stopping...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_loop(final_model, full_train_loader, final_criterion, final_optimizer)
        val_loss = evaluate(final_model, final_val_loader, final_criterion)
        final_scheduler.step(val_loss)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(final_model.state_dict(), final_model_path)
            print(f"Validation loss improved. Saving model to '{final_model_path}'")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    
    print(f"\nFinal model training complete. Best model saved to '{final_model_path}'")
