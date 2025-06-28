import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_and_resample(df, time_col, freq="1h", agg="mean"):
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    df = df.sort_index()
    df_resampled = getattr(df.resample(freq), agg)()
    return df_resampled


def split_dataframe(df, splits):
    total_len = len(df)
    train_end = int(total_len * splits[0])
    val_end = train_end + int(total_len * splits[1])
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def one_hot_encode_columns(df, columns_to_encode, drop_first=False, prefix_sep="_"):
    existing_cols = [col for col in columns_to_encode if col in df.columns]
    missing_cols = [col for col in columns_to_encode if col not in df.columns]

    if missing_cols:
        print("❌ Columns not found in DataFrame and skipped:", missing_cols)
    if existing_cols:
        df = pd.get_dummies(
            df, columns=existing_cols, drop_first=drop_first, prefix_sep=prefix_sep
        )

    return df


def scale_uni_data(data):
    df_train, df_val, df_test = data
    for df in [df_train, df_val, df_test]:
        for col in df.select_dtypes(include=["datetime64[ns]", "object"]).columns:
            df.drop(columns=col, inplace=True)
    scaler = MinMaxScaler()

    # Save column names and index
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

    return scaler, df_train_scaled, df_val_scaled, df_test_scaled


def scale_multi_data(data, target_col):
    df_train, df_val, df_test = data
    for df in [df_train, df_val, df_test]:
        for col in df.select_dtypes(include=["datetime64[ns]", "object"]).columns:
            df.drop(columns=col, inplace=True)
    # Separate target and features
    X_train, y_train = df_train.drop(columns=[target_col]), df_train[[target_col]]
    X_val, y_val = df_val.drop(columns=[target_col]), df_val[[target_col]]
    X_test, y_test = df_test.drop(columns=[target_col]), df_test[[target_col]]

    # Initialize scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit on training data only
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)

    # Transform val/test using same scalers
    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)

    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)

    # Reconstruct DataFrames with original columns and index
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

    # Return everything
    return (
        {"target": target_scaler, "feature": feature_scaler},
        df_train_scaled,
        df_val_scaled,
        df_test_scaled,
    )


def build_traditional_sequences(df, lookback, horizon, target_col="load"):
    X, y = [], []

    df_values = df.values
    target_idx = df.columns.get_loc(target_col)

    for i in range(lookback, len(df) - horizon):
        x_seq = df_values[i - lookback : i]
        y_seq = df_values[i : i + horizon, target_idx]
        X.append(x_seq)
        y.append(y_seq)

    return np.array(X), np.array(y)


def build_periodic_sequences(df_all, timestamps, n, freq, horizon):
    if len(df_all) != len(timestamps):
        raise ValueError("Length of df_all and timestamps must match.")

    df_all = df_all.copy()
    df_all.index = pd.to_datetime(timestamps)

    X_per = []

    for idx in range(n * 24 + horizon, len(df_all) - horizon):
        t = df_all.index[idx]
        current_hour = t.hour

        p_input = []
        valid = True

        for i in range(1, n + 1):
            base_time = (t - pd.Timedelta(days=i)).replace(hour=current_hour)
            horizon_range = pd.date_range(
                start=base_time + pd.to_timedelta(freq), periods=horizon, freq=freq
            )

            if not all(ts in df_all.index for ts in horizon_range):
                valid = False
                break

            vals = df_all.loc[horizon_range].values
            p_input.append(vals.mean(axis=0))  # collapse (h, f) into (f,)

        if valid:
            X_per.append(np.stack(p_input))  # (n, f)

    return np.array(X_per)


def smooth_and_clean_target(df, lookback, freq="1h", target_col="vrednost"):
    df = df.set_index("ts").rename(columns={target_col: "cntr"})
    df.rename_axis("Date_Time", inplace=True)

    # Resample and round
    df = df.resample(freq.lower()).sum().reset_index()
    df["cntr"] = df["cntr"].round(2)

    # Smooth with rolling mean (backward fill for early NaNs)
    df["SMA"] = df["cntr"].rolling(window=lookback).mean().bfill()

    # IQR outlier smoothing
    Q1, Q3 = df["cntr"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = (df["cntr"] < (Q1 - 1.5 * IQR)) | (df["cntr"] > (Q3 + 1.5 * IQR))
    average_max = df["cntr"].nlargest(lookback).mean()
    df.loc[outliers, "cntr"] = average_max

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
    scalers = {"target": None, "feature": None}

    if uni:
        scaler, train_df, val_df, test_df = scale_uni_data(
            split_dataframe(df, split_ratio)
        )
        scalers["target"] = scaler

    else:
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

    X_train, y_train = build_traditional_sequences(
        train_df, lookback, horizon, target_col
    )
    X_val, y_val = build_traditional_sequences(val_df, lookback, horizon, target_col)
    X_test, y_test = build_traditional_sequences(test_df, lookback, horizon, target_col)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    return scalers, train_loader, val_loader, test_loader


def to_loader(X_s, X_p, y, batch_size):
    X_s = torch.tensor(X_s, dtype=torch.float32)
    X_p = torch.tensor(X_p, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_s, X_p, y), batch_size=batch_size, shuffle=False)
