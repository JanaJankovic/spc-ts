import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_and_resample(df, time_col, freq="1h", agg="mean"):
    print(
        f"📅 Converting '{time_col}' to datetime and resampling with frequency '{freq}' using '{agg}' aggregation."
    )
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    df = df.sort_index()
    df_resampled = getattr(df.resample(freq), agg)()
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

    return (
        {"target": target_scaler, "feature": feature_scaler},
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

    for i in range(lookback, len(df) - horizon):
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


def build_periodic_sequences(df_all, n, horizon):
    print(f"⚡ Fast periodic sequence builder: n={n}, horizon={horizon}")

    values = df_all.values
    total_len = len(values)
    X_per = []

    offset = n * 24 + horizon
    total = total_len - offset
    print(f"📊 Total samples to process: {total}")

    for idx in range(offset, total_len - horizon):
        p_input = []
        valid = True

        for i in range(1, n + 1):
            start_idx = idx - i * 24 + 1  # +1 = skip current hour
            end_idx = start_idx + horizon

            if end_idx > total_len:
                valid = False
                break

            vals = values[start_idx:end_idx]
            if len(vals) < horizon:
                valid = False
                break

            p_input.append(vals.mean(axis=0))

        if valid:
            X_per.append(np.stack(p_input))  # shape (n, features)

        if (idx - offset) % 1000 == 0:
            print(f"🔄 Processed {idx - offset + 1}/{total}", end="\r")

    print(f"\n✅ Periodic sequences built: {len(X_per)}")
    return np.array(X_per)


def smooth_and_clean_target(
    df, lookback, freq="1h", target_col="load", time_col="datetime"
):
    print(f"🧼 Smoothing and cleaning target column '{target_col}'...")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).rename(columns={target_col: "cntr"})

    print(f"⏱ Resampling to {freq} and summing...")
    df = df.resample(freq.lower()).sum().reset_index()
    df["cntr"] = df["cntr"].round(2)

    print(f"📈 Applying {lookback}-window Simple Moving Average...")
    df["SMA"] = df["cntr"].rolling(window=lookback).mean().bfill()

    Q1, Q3 = df["cntr"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = (df["cntr"] < (Q1 - 1.5 * IQR)) | (df["cntr"] > (Q3 + 1.5 * IQR))
    num_outliers = outliers.sum()

    average_max = df["cntr"].nlargest(lookback).mean()
    df.loc[outliers, "cntr"] = average_max

    print(
        f"🚫 Replaced {num_outliers} outliers with average of top {lookback} values: {average_max:.2f}"
    )
    df = df.rename(columns={"cntr": target_col})

    print(f"✅ Target cleaned and ready.")
    return df


def add_time_features(df, datetime_col="ts", holidays=None, include_hour_features=True):
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
    else:
        df["is_holiday"] = 0

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
    # Load and parse datetime column
    df = pd.read_csv(path, parse_dates=[index_col])
    df.set_index(index_col, inplace=True)

    # Ensure full time range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range)
    df.index.name = index_col

    # Drop columns with too many missing values
    missing_ratio = df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    if verbose and drop_cols:
        print(
            f"🧹 Dropping {len(drop_cols)} columns with too much missing data: {drop_cols}"
        )
    df = df.drop(columns=drop_cols)

    # Interpolate remaining missing values
    df = df.interpolate(method=interpolation, limit_direction=limit_direction)

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
    freq,
    horizon,
    batch_size,
    split_ratio,
    target_col="vrednost",
    uni=True,
):
    print("📦 Starting data loader preparation...")
    scalers = {"target": None, "feature": None}

    if uni:
        print("🔄 Using univariate scaling...")
        scaler, train_df, val_df, test_df = scale_uni_data(
            split_dataframe(df, split_ratio)
        )
        scalers["target"] = scaler
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
        scalers, train_df, val_df, test_df = scale_multi_data(
            split_dataframe(df, split_ratio), target_col
        )

    print("🔧 Building traditional sequences...")
    X_train, y_train = build_traditional_sequences(
        train_df, lookback, horizon, target_col
    )
    X_val, y_val = build_traditional_sequences(val_df, lookback, horizon, target_col)
    X_test, y_test = build_traditional_sequences(test_df, lookback, horizon, target_col)

    print(f"📐 Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"📐 Shapes - X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"📐 Shapes - X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print("📤 Creating DataLoaders...")
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
        scalers,
        (train_loader, val_loader, test_loader),
        X_train.shape[1:],
    )


def to_loader(X_s, X_p, y, batch_size):
    X_s = torch.tensor(X_s, dtype=torch.float32)
    X_p = torch.tensor(X_p, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_s, X_p, y), batch_size=batch_size, shuffle=False)
