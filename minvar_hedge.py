# %%
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from tqdm import trange

pd.set_option('mode.chained_assignment', None)
# pd.options.mode.chained_assignment = 'raise'

# %%
def df_transform(df):
    df["option_changed"] = (df[["exdate", "strike price"]].shift(-1) != df[["exdate", "strike price"]]).all(axis=1)
    df["T"] = df["time to maturity"] / 365
    df["del_f"] = df["option price"].diff()
    df["del_S"] = df["underlying price"].diff() 
    df["err_del"] = df["del_f"] - df["delta"] * df["del_S"]
    df["regr_term"] = (df["vega"] / np.sqrt(df["T"])) * (df["del_S"] / df["underlying price"])
    df["regr_y"] = df["err_del"] / df["regr_term"]
    df["del_mv_term"] = df["vega"]/(df["underlying price"] * df["T"])
    df["del_quad"] = -1
    df["del_mv"] = -1
    df["err_mv"] = -1

    return df

# %%
def calc_gain(df):
    gain = 1 - ((df["err_mv"] ** 2).sum() / (df["err_del"] ** 2).sum())

    return gain
    
# %%
def mv_hedge(filepath, est_days):
    df = pd.read_csv(filepath)
    df = df_transform(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["regr_y"], how="all")

    num_rows, _ = df.shape
    row_acc = 0
    month_acc = -1

    # for row in trange(1, 1000):
    for row in trange(1, num_rows - 1):
        if df["option_changed"].iloc[row]:
            row_acc = 0
            month_acc = -1
        else:
            row_acc = row_acc + 1

        if row_acc == est_days:
            month_acc = 0
        elif row_acc > est_days:
            month_acc = month_acc + 1
        else:
            pass

        if month_acc == 31:
            month_acc = 0

        if month_acc == 0:
           X = df["delta"].iloc[row - est_days - 1:row]
           y = df["regr_y"].iloc[row - est_days - 1:row]
           u = Polynomial.fit(X, y, 2)

        if row_acc >= est_days:
            df["del_quad"].iloc[row] = u(df["delta"].iloc[row])
        
        # print(row, row_acc, month_acc)

    df["del_mv"] = df["delta"] + df["del_mv_term"] * df["del_quad"]
    df["err_mv"] = df["err_del"] - df["regr_term"] * df["del_quad"]

    cols = ['quote_date',
    'root',
    'exdate',
    'strike price',
    'underlying price',
    'option price',
    'delta',
    'gamma',
    'theta',
    'vega',
    'rho',
    'implied volatility',
    'time to maturity',
    'T',
    'del_f',
    'del_S',
    'option_changed',
    'regr_term',
    'regr_y',
    'del_quad',
    'del_mv_term',
    'del_mv',
    'err_mv',
    'err_del']
    df = df[cols]

    return df

# %%
def main():
    filepath = "train_data/SPX_C.csv" # TODO: Add option to run puts as well "train_data/SPX_P.csv"
    est_days = 36 * 30
    df = mv_hedge(filepath, est_days)
    df = df[df["del_quad"] != -1]
    fin_num_rows, _ = df.shape
    df = df.reset_index().drop([0, fin_num_rows - 1]) # TODO: Check if needed
    gain = calc_gain(df)
    df.to_csv("mv_hedge_C_gain_" + str(round(gain, 2)) + ".csv")

# %%
if __name__ == "__main__":
    main()