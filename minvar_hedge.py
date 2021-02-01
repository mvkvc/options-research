# %%
from os import X_OK
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from tqdm import trange

# %%
filepath = "train_data/SPX_C.csv" # "train_data/SPX_P.csv"
est_days = 36 * 30

# %%
def est_constants():
    # Estimate a, b, c with regression
    pass

# %%
def calc_del_mv(S, T, del_bs, vega_bs, a, b, c):
    del_mv = del_bs + (vega_bs / (S * np.sqrt(T))) * (a + b * del_bs + c * del_bs ** 2)
    return del_mv

def calc_sse():
    pass

def calc_gain():
    pass

# %%
def mv_hedging():

    with open(filepath) as f:
        num_rows = sum(1 for line in f)

    df = pd.read_csv(filepath) # TODO: Add terms for estimating a, b, c, in df
    df["option_changed"] = (df[["exdate", "strike price"]].shift(-1) != df[["exdate", "strike price"]]).all(axis=1)
    df["T"] = df["time to maturity"] / 365
    df["del_f"] = df["option price"].diff()
    df["del_S"] = df["underlying price"].diff()
    df["regr_y"] = (df["del_f"] - df["delta"] * df["del_S"]) / ((df["vega"]/np.sqrt(df["T"])) * (df["del_S"]/df["underlying_price"]))
    df["del_mv"] = -1
    df["err_mv"] = -1

    row_acc = 0
    month_acc = 0
    
    for row in trange(1, num_rows - 1):

        if df["option_changed"].iloc[row]:
            row_acc = -1
            month_acc = -1
        else:
            row_acc = row_acc + 1
        print(row_acc)

        if row_acc == est_days:
            month_acc = 0
        elif row_acc > est_days:
            month_acc = month_acc + 1
        else:
            pass

        if month_acc == 31:
            month_acc = 0

        if month_acc == 0:
           # TODO: Calculate a, b, c with est_days
           X = df["delta"].iloc[row - est_days - 1:row]
           y = df["regr_y"].iloc[row - est_days - 1:row]
           u = Polynomial(X, y, 2)

        if row_acc >= est_days:
            # TODO: Apply calculated formula for error and record in row
            df["del_mv"].iloc[row] = -1
            df["err_mv"].iloc[row] = -1

    return

# %%
def main():
    pass

# %%
if __name__ == "__main__":
    main()