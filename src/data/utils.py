import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


def fill_missing_time(df, datetime_col, method="interpolate"):
    freq = "1h"

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)

    # Build new index
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_range)

    # Fill missing values according to the method
    if method == "interpolate":
        df = df.interpolate()
    elif method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown fill method: {method}")

    df = df.reset_index().rename(columns={"index": datetime_col})
    df = df.sort_values(by=datetime_col).reset_index(drop=True)

    return df


def load_and_resample(df, time_col, freq="1h", agg="mean"):
    print(
        f"📅 Converting '{time_col}' to datetime and resampling with frequency '{freq}' using '{agg}' aggregation."
    )
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    df = df.sort_index()

    df_resampled = getattr(df.resample(freq), agg)()
    df_resampled = df_resampled.reset_index()  # <-- bring back datetime as column

    print(f"✅ Resampling complete. Resulting shape: {df_resampled.shape}")
    return df_resampled


def split_dataframe(df, splits):
    print(f"🔀 Splitting DataFrame with ratios {splits}")
    total_len = len(df)
    train_end = int(total_len * splits[0])
    val_end = train_end + int(total_len * splits[1])
    print(
        f"📊 Split sizes: Train={train_end}, Val={val_end - train_end}, Test={total_len - val_end}"
    )
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def one_hot_encode_columns(df, columns_to_encode, drop_first=False, prefix_sep="_"):
    print(f"🔣 One-hot encoding columns: {columns_to_encode}")
    existing_cols = [col for col in columns_to_encode if col in df.columns]
    missing_cols = [col for col in columns_to_encode if col not in df.columns]

    if missing_cols:
        print("❌ Columns not found in DataFrame and skipped:", missing_cols)
    if existing_cols:
        print("✅ Columns to encode:", existing_cols)
        df = pd.get_dummies(
            df, columns=existing_cols, drop_first=drop_first, prefix_sep=prefix_sep
        )
        print(f"🎯 One-hot encoding complete. New shape: {df.shape}")
    else:
        print("⚠️ No columns found for one-hot encoding.")

    return df


def scale_uni_data(data):
    print("📏 Scaling univariate data using MinMaxScaler.")
    df_train, df_val, df_test = data
    for name, df in zip(["train", "val", "test"], [df_train, df_val, df_test]):
        datetime_cols = df.select_dtypes(
            include=["datetime64[ns]", "object"]
        ).columns.tolist()
        if datetime_cols:
            print(f"🧹 Dropping non-numeric columns from {name} set: {datetime_cols}")
            df.drop(columns=datetime_cols, inplace=True)

    scaler = MinMaxScaler()

    columns = df_train.columns
    idx_train = df_train.index
    idx_val = df_val.index
    idx_test = df_test.index

    df_train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train), columns=columns, index=idx_train
    )
    df_val_scaled = pd.DataFrame(
        scaler.transform(df_val), columns=columns, index=idx_val
    )
    df_test_scaled = pd.DataFrame(
        scaler.transform(df_test), columns=columns, index=idx_test
    )

    print("✅ Scaling complete. Shapes:")
    print(f"   Train: {df_train_scaled.shape}")
    print(f"   Val:   {df_val_scaled.shape}")
    print(f"   Test:  {df_test_scaled.shape}")

    return scaler, df_train_scaled, df_val_scaled, df_test_scaled


def scale_multi_data(data, target_col):
    print(f"📏 Scaling multivariate data (target: '{target_col}') using MinMaxScaler.")
    df_train, df_val, df_test = data

    for name, df in zip(["train", "val", "test"], [df_train, df_val, df_test]):
        datetime_cols = df.select_dtypes(
            include=["datetime64[ns]", "object"]
        ).columns.tolist()
        if datetime_cols:
            print(f"🧹 Dropping non-numeric columns from {name} set: {datetime_cols}")
            df.drop(columns=datetime_cols, inplace=True)

    # Separate target and features
    print("🔧 Splitting features and target...")
    X_train, y_train = df_train.drop(columns=[target_col]), df_train[[target_col]]
    X_val, y_val = df_val.drop(columns=[target_col]), df_val[[target_col]]
    X_test, y_test = df_test.drop(columns=[target_col]), df_test[[target_col]]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    print("⚙️ Fitting and transforming training set...")
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)

    print("📐 Transforming validation and test sets...")
    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)

    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)

    df_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=df_train.index
    )
    df_train_scaled[target_col] = y_train_scaled

    df_val_scaled = pd.DataFrame(
        X_val_scaled, columns=X_val.columns, index=df_val.index
    )
    df_val_scaled[target_col] = y_val_scaled

    df_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=df_test.index
    )
    df_test_scaled[target_col] = y_test_scaled

    print("✅ Scaling complete. Shapes:")
    print(
        f"   Train: {df_train_scaled.shape}, Val: {df_val_scaled.shape}, Test: {df_test_scaled.shape}"
    )
    # Print the actual min/max of the original target
    original_y_min = y_train.min().values[0]
    original_y_max = y_train.max().values[0]

    print(
        f"📊 Original target range: min={original_y_min:.2f}, max={original_y_max:.2f}"
    )
    print(
        f"🎯 Scaler fitted range: min={target_scaler.data_min_[0]:.2f}, max={target_scaler.data_max_[0]:.2f}"
    )

    return (
        target_scaler,
        df_train_scaled,
        df_val_scaled,
        df_test_scaled,
    )


def build_traditional_sequences(df, lookback, horizon, target_col="load"):
    print(
        f"🧱 Building traditional sequences (lookback={lookback}, horizon={horizon}, target='{target_col}')"
    )
    X, y = [], []

    df_values = df.values
    target_idx = df.columns.get_loc(target_col)
    total_sequences = len(df) - lookback - horizon

    for i in range(lookback, len(df) - horizon + 1, horizon):
        if i % 1000 == 0 or i == lookback:
            print(
                f"  🔄 Progress: {i - lookback + 1}/{total_sequences} sequences",
                end="\r",
            )
        x_seq = df_values[i - lookback : i]
        y_seq = df_values[i : i + horizon, target_idx]
        X.append(x_seq)
        y.append(y_seq)

    print(f"\n✅ Sequence creation complete. Total sequences: {len(X)}")
    return np.array(X), np.array(y)


def build_periodic_sequences(df, lookback, horizon, n):
    print(
        f"⚡ Building periodic inputs aligned with traditional sequences (lookback={lookback}, horizon={horizon}, n={n})"
    )

    values = df.values  # shape (T, features)
    total_len = len(values)
    X_per = []

    # This ensures the same starting point as traditional loop
    start_idx = max(lookback, n * 24)
    end_idx = total_len - horizon

    print(f"📊 Total samples to process: {end_idx - start_idx}")

    for idx in range(start_idx, end_idx):
        p_input = []
        valid = True

        for i in range(1, n + 1):
            lookup_idx = idx - i * 24
            if lookup_idx < 0:
                valid = False
                break
            p_input.append(values[lookup_idx])

        if valid:
            X_per.append(np.stack(p_input))  # shape (n, features)

        if (idx - start_idx) % 1000 == 0:
            print(f"🔄 Processed {idx - start_idx + 1}/{end_idx - start_idx}", end="\r")

    print(f"\n✅ Periodic sequences built: {len(X_per)}")
    return np.array(X_per)


def smooth_and_clean_target(
    df, lookback, freq="1h", target_col="load", time_col="datetime"
):
    print(f"🧼 Smoothing and cleaning target column '{target_col}'...")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).rename(columns={target_col: "cntr"})

    print(f"⏱ Resampling to {freq} and summing...")
    df = df.resample(freq.lower()).sum().reset_index()
    df["cntr"] = df["cntr"].round(2)

    # === Step 1: Detect and replace outliers with median ===
    Q1, Q3 = df["cntr"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df["cntr"] < lower_bound) | (df["cntr"] > upper_bound)
    num_outliers = outliers.sum()

    median_val = df["cntr"].median()
    df.loc[outliers, "cntr"] = median_val
    print(f"🚫 Replaced {num_outliers} outliers with median value: {median_val:.2f}")

    # === Step 2: Apply smoothing after outlier correction ===
    print(f"📈 Applying {lookback}-window Simple Moving Average...")
    df["cntr"] = df["cntr"].rolling(window=lookback, min_periods=1).mean()

    # === Step 3: Rename and finish ===
    df = df.rename(columns={"cntr": target_col})
    print(f"✅ Target cleaned and ready.")
    return df


def add_time_features(
    df, datetime_col="datetime", holidays=None, include_hour_features=True
):
    print(df.columns)
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    df["day_of_week"] = df[datetime_col].dt.dayofweek
    df["month"] = df[datetime_col].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if include_hour_features:
        df["hour"] = df[datetime_col].dt.hour
        df["is_night"] = (
            df["hour"].isin(list(range(0, 6)) + list(range(22, 24))).astype(int)
        )
    else:
        df["is_night"] = 0  # default to 0 if not included

    if holidays is not None:
        holiday_dates = pd.to_datetime(holidays).dt.date
        df["is_holiday"] = df[datetime_col].dt.date.isin(holiday_dates).astype(int)

    return df


def preprocess_weather_data(
    path,
    index_col="time",
    freq="h",
    drop_threshold=0.15,
    interpolation="time",
    limit_direction="both",
    verbose=False,
):
    print(f"🌤️ Preprocessing weather data from '{path}'...")

    df = pd.read_csv(path, parse_dates=[index_col])
    df.set_index(index_col, inplace=True)
    print("📅 Parsed and set index to datetime.")

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range)
    df.index.name = index_col
    print(f"📏 Ensured full datetime range with {len(df)} time steps.")

    missing_ratio = df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    if verbose and drop_cols:
        print(
            f"🧹 Dropping {len(drop_cols)} columns with >{int(drop_threshold*100)}% missing values: {drop_cols}"
        )
    df = df.drop(columns=drop_cols)

    df = df.interpolate(method=interpolation, limit_direction=limit_direction)
    print(f"🔧 Interpolated missing values with method='{interpolation}'")

    print("✅ Weather data preprocessing complete.")
    return df


def join_calendar_and_weather(load_df, weather_df, freq="h", timestamp_col="datetime"):
    # Parse and index the datetime column
    load_df[timestamp_col] = pd.to_datetime(load_df[timestamp_col])
    load_df = load_df.set_index(timestamp_col).sort_index()

    # Automatically infer full datetime range
    start_date = load_df.index.min()
    end_date = load_df.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Reindex and interpolate missing values in load data
    load_df = load_df.reindex(full_range)
    load_df = load_df.interpolate(method="time", limit_direction="both")
    load_df.index.name = timestamp_col

    # Prepare weather_df
    weather_df.index = pd.to_datetime(weather_df.index)
    weather_df = weather_df.sort_index()

    # Join on datetime index
    merged_df = load_df.join(weather_df, how="inner")

    return merged_df.reset_index().rename(columns={"index": timestamp_col})


def join_calendar_and_weather(load_df, weather_df, freq="h", timestamp_col="datetime"):
    print("🔗 Joining load and weather data...")

    load_df[timestamp_col] = pd.to_datetime(load_df[timestamp_col])
    load_df = load_df.set_index(timestamp_col).sort_index()
    print("📅 Parsed and sorted load data timestamps.")

    start_date = load_df.index.min()
    end_date = load_df.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    print(f"📏 Full date range inferred: {start_date} to {end_date}")

    load_df = load_df.reindex(full_range)
    load_df = load_df.interpolate(method="time", limit_direction="both")
    load_df.index.name = timestamp_col
    print("🔧 Interpolated missing values in load data")

    weather_df.index = pd.to_datetime(weather_df.index)
    weather_df = weather_df.sort_index()
    print("🌦️ Prepared and sorted weather data index.")

    merged_df = load_df.join(weather_df, how="inner")
    print(f"✅ Joined DataFrame shape: {merged_df.shape}")

    return merged_df.reset_index().rename(columns={"index": timestamp_col})


def get_data_loaders(
    df,
    lookback,
    horizon,
    batch_size,
    split_ratio,
    target_col="load",
    uni=True,
    df_raw=None,  # optional: use for raw target scaler + unscaled y_true
):
    print("📦 Starting data loader preparation...")

    use_raw_scaler = df_raw is not None
    target_scaler = None

    # === Handle optional raw scaler and y_true ===
    if use_raw_scaler:
        print(
            "🎯 Splitting raw data and fitting scaler only on train set to avoid leakage..."
        )
        df_raw_train, df_raw_val, df_raw_test = split_dataframe(df_raw, split_ratio)

        target_scaler = MinMaxScaler()
        target_scaler.fit(df_raw_train[[target_col]])
    else:
        df_raw_train = df_raw_val = df_raw_test = None

    # === Scaling and split ===
    if uni:
        print("🔄 Using univariate scaling...")
        scaler, train_df, val_df, test_df = scale_uni_data(
            split_dataframe(df, split_ratio)
        )
    else:
        print("🔄 Using multivariate scaling with one-hot encoding...")
        calendar_cols = [
            "day_of_week",
            "month",
            "is_weekend",
            "hour",
            "is_night",
            "is_holiday",
        ]
        df = one_hot_encode_columns(df, calendar_cols, drop_first=True)
        scaler, train_df, val_df, test_df = scale_multi_data(
            split_dataframe(df, split_ratio), target_col
        )

    # === Sequences ===
    print("🔧 Building traditional sequences...")
    X_train, y_train = build_traditional_sequences(
        train_df, lookback, horizon, target_col
    )
    X_val, y_val = build_traditional_sequences(val_df, lookback, horizon, target_col)
    X_test, y_test = build_traditional_sequences(test_df, lookback, horizon, target_col)

    # === Raw target values (optional, from unscaled df_raw) ===
    if use_raw_scaler:
        print("📎 Extracting and scaling raw unscaled targets for each set...")
        _, y_train_true = build_traditional_sequences(
            df_raw_train, lookback, horizon, target_col
        )
        _, y_val_true = build_traditional_sequences(
            df_raw_val, lookback, horizon, target_col
        )
        _, y_test_true = build_traditional_sequences(
            df_raw_test, lookback, horizon, target_col
        )

        # === Scale raw targets using target_scaler ===
        def scale_y_true(y_arr):
            return target_scaler.transform(y_arr.reshape(-1, 1)).reshape(y_arr.shape)

        y_train_true = scale_y_true(y_train_true)
        y_val_true = scale_y_true(y_val_true)
        y_test_true = scale_y_true(y_test_true)

    # === Torch Tensors ===
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    X_train, y_train = to_tensor(X_train), to_tensor(y_train)
    X_val, y_val = to_tensor(X_val), to_tensor(y_val)
    X_test, y_test = to_tensor(X_test), to_tensor(y_test)

    if use_raw_scaler:
        y_train_true = to_tensor(y_train_true)
        y_val_true = to_tensor(y_val_true)
        y_test_true = to_tensor(y_test_true)

    # === DataLoaders ===
    print("📤 Creating DataLoaders...")
    if use_raw_scaler:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train, y_train_true),
            batch_size=batch_size,
            shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val, y_val_true),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test, y_test_true),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
        )

    print("✅ DataLoaders ready.")
    return (
        target_scaler if use_raw_scaler else scaler,
        (train_loader, val_loader, test_loader),
        X_train.shape[1:],
    )


def to_loader(X_s, X_p, y, batch_size):
    X_s = torch.tensor(X_s, dtype=torch.float32)
    X_p = torch.tensor(X_p, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_s, X_p, y), batch_size=batch_size, shuffle=False)


def stack_dataframes_long(load_dir, time_col, freq):
    files = [
        f for f in os.listdir(load_dir) if f.startswith("mm") and f.endswith(".csv")
    ]
    lengths = []
    temp_dfs = []

    # First pass: get lengths for all
    for idx, fname in enumerate(sorted(files)):
        fpath = os.path.join(load_dir, fname)
        df = pd.read_csv(fpath)
        df = fill_missing_time(df, time_col)
        df = load_and_resample(df, time_col, freq)
        lengths.append(len(df))
        temp_dfs.append((idx, df))

    # Compute median and find indices above or equal to median
    lengths = np.array(lengths)
    median_length = int(np.median(lengths))
    keep_idx = [i for i, l in enumerate(lengths) if l >= median_length]

    print(f"Dataset lengths: {lengths}")
    print(f"Median length: {median_length}")
    print(f"Keeping consumers: {keep_idx}")

    # Only include those with sufficient length; add consumer_id, select cols
    long_dfs = []
    for idx, df in temp_dfs:
        if idx in keep_idx:
            df = df.copy()
            df["consumer_id"] = idx
            long_dfs.append(df[["load", "consumer_id"]])

    # Concatenate all into long format
    big_df = pd.concat(long_dfs, ignore_index=True)
    print(f"Shape after stacking (long format): {big_df.shape}")
    print(big_df.head())

    return big_df, median_length


def get_universal_data_loaders(
    df,
    lookback,
    horizon,
    batch_size,
    split_ratio,
    target_col="load",
    consumer_col="consumer_id",
):
    print("📦 Starting universal data loader preparation (long format)...")

    consumer_scalers = {}
    all_X, all_y, all_ids, split_labels = [], [], [], []

    for cid, sub_df in df.groupby(consumer_col):
        arr = sub_df[[target_col]].values  # Shape (n, 1)
        n = len(arr)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        scaler = MinMaxScaler()
        scaler.fit(arr[:train_end])
        consumer_scalers[cid] = scaler

        # Split *indices* on the full sub_df
        splits = {
            "train": sub_df.iloc[:train_end].copy(),
            "val": sub_df.iloc[train_end:val_end].copy(),
            "test": sub_df.iloc[val_end:].copy(),
        }

        # Scale splits in-place
        for split_name, split_df in splits.items():
            if len(split_df) < lookback + horizon:
                continue  # skip splits too short for sequence

            split_df[target_col] = scaler.transform(split_df[[target_col]])
            X, y = build_traditional_sequences(
                split_df[[target_col]].copy(), lookback, horizon, target_col=target_col
            )
            print(f"CID={cid} {split_name}: X shape={X.shape} y shape={y.shape}")
            all_X.append(X)
            all_y.append(y)
            all_ids.extend([cid] * len(X))
            split_labels.extend([split_name] * len(X))

    if not all_X:
        raise ValueError("No sequences could be built for any consumer/split!")

    # Stack all collected arrays
    X_all = np.vstack(all_X)  # [num_samples, lookback, num_features]
    y_all = np.vstack(all_y)  # [num_samples, horizon]
    ids_all = np.array(all_ids)
    split_labels = np.array(split_labels)

    # For PyTorch
    X_all = torch.tensor(X_all, dtype=torch.float32)
    y_all = torch.tensor(y_all, dtype=torch.float32)
    ids_all = torch.tensor(ids_all, dtype=torch.long)

    train_idx = np.where(split_labels == "train")[0]
    val_idx = np.where(split_labels == "val")[0]
    test_idx = np.where(split_labels == "test")[0]

    train_ds = TensorDataset(X_all[train_idx], y_all[train_idx], ids_all[train_idx])
    val_ds = TensorDataset(X_all[val_idx], y_all[val_idx], ids_all[val_idx])
    test_ds = TensorDataset(X_all[test_idx], y_all[test_idx], ids_all[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Input shape for model: {X_all.shape[1:]}")
    return consumer_scalers, (train_loader, val_loader, test_loader), X_all.shape[1:]
