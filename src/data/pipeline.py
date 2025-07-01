import src.data.utils as utils
import pandas as pd
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
HOLIDAYS_PATH = os.path.join(DATA_DIR, "slovenia_holidays.csv")
WEATHER_PATH = os.path.join(DATA_DIR, "slovenia_weather.csv")

SPLIT_RATIO = (0.6, 0.1, 0.3)


def cnn_lstm_pipeline(
    load_path,
    lookback,
    horizon,
    batch,
    time_col="datetime",
    target_col="load",
    freq="1h",
    use_calendar=True,
    use_weather=True,
):
    df = pd.read_csv(load_path)

    if freq != "1h":
        df = utils.load_and_resample(df, time_col, freq)

    if use_calendar:
        holidays = pd.read_csv(HOLIDAYS_PATH)
        df = utils.add_time_features(
            df,
            datetime_col=time_col,
            holidays=holidays["holiday"],
            include_hour_features=(freq == "1h"),
        )

    if use_weather:
        weather_df = utils.preprocess_weather_data(WEATHER_PATH, time_col, freq=freq)
        df = utils.join_calendar_and_weather(
            df, weather_df, freq=freq, timestamp_col=time_col
        )

    if not use_calendar and not use_weather:
        return utils.get_data_loaders(
            df,
            lookback,
            freq,
            horizon,
            batch,
            SPLIT_RATIO,
            uni=True,
            target_col=target_col,
        )

    return utils.get_data_loaders(
        df,
        lookback,
        freq,
        horizon,
        batch,
        SPLIT_RATIO,
        uni=False,
        target_col=target_col,
    )


def di_rnn_pipeline(load_path, m, n, horizon, batch_size, target_col="load"):
    df = pd.read_csv(load_path)

    scaler, train_df, val_df, test_df = utils.scale_uni_data(
        utils.split_dataframe(df, SPLIT_RATIO)
    )

    # Sequential component
    X_s_train, y_s_train = utils.build_traditional_sequences(train_df, m, horizon, target_col)
    X_s_val, y_s_val = utils.build_traditional_sequences(val_df, m, horizon, target_col)
    X_s_test, y_s_test = utils.build_traditional_sequences(test_df, m, horizon, target_col)

    # Periodic component
    X_p_train = utils.build_periodic_sequences(train_df, m, horizon, n)
    X_p_val = utils.build_periodic_sequences(val_df, m, horizon, n)
    X_p_test = utils.build_periodic_sequences(test_df, m, horizon, n)

    # Convert to DataLoaders
    train_loader = utils.to_loader(X_s_train, X_p_train, y_s_train, batch_size)
    val_loader = utils.to_loader(X_s_val, X_p_val, y_s_val, batch_size)
    test_loader = utils.to_loader(X_s_test, X_p_test, y_s_test, batch_size)

    return scaler, (train_loader, val_loader, test_loader)



def base_residual_pipeline(
    load_path,
    lookback,
    horizon,
    batch,
    time_col="datetime",
    target_col="load",
    freq="1h",
    use_calendar=True,
    use_weather=True,
):
    df = pd.read_csv(load_path)
    df = utils.smooth_and_clean_target(df, lookback, freq, target_col, time_col)

    if use_calendar:
        holidays = pd.read_csv(HOLIDAYS_PATH)
        df = utils.add_time_features(
            df,
            time_col,
            holidays["holiday"],
            include_hour_features=(freq == "1h"),
        )

    if use_weather:
        weather_df = utils.preprocess_weather_data(WEATHER_PATH, "datetime", freq=freq)
        df = utils.join_calendar_and_weather(df, weather_df, freq=freq)

    if not use_calendar and not use_weather:
        return utils.get_data_loaders(
            df,
            lookback,
            freq,
            horizon,
            batch,
            SPLIT_RATIO,
            uni=True,
            target_col=target_col,
        )

    return utils.get_data_loaders(
        df,
        lookback,
        freq,
        horizon,
        batch,
        SPLIT_RATIO,
        uni=False,
        target_col=target_col,
    )
