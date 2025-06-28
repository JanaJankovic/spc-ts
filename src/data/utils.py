import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def split_dataframe(df, splits):
    total_len = len(df)
    train_end = int(total_len * splits[0])
    val_end = train_end + int(total_len * splits[1])
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy()
    )


def scale_data(df_train, df_val, df_test):
    scaler = MinMaxScaler()
    df_train['scaled'] = scaler.fit_transform(df_train[['vrednost']])
    df_val['scaled']   = scaler.transform(df_val[['vrednost']])
    df_test['scaled']  = scaler.transform(df_test[['vrednost']])
    return scaler, pd.concat([df_train, df_val, df_test])