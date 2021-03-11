# %%
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
from tqdm import trange

pd.set_option("mode.chained_assignment", None)

#%%
path = "train_data/SPX_P.csv"
# path = "train_data/SPX_C.csv"
params = {
    "type": "P",
    # "type": "C",
    "bal_per": 1,
    "lookback_mths": 12,
    "T_days": 252,
    "excel_rows": 10000,
}

#%%
print("Importing data")
df_raw = pd.read_csv(path)
df = df_raw.copy()

#%%
print("Transforming data")
df["del_S"] = df["underlying price"].diff().shift(-params["bal_per"])
df["next_S"] = df["underlying price"].shift(-params["bal_per"])
df["del_f"] = df["option price"].diff().shift(-params["bal_per"])
df["next_f"] = df["option price"].shift(-params["bal_per"])

df["option_changed"] = (
    df[["exdate", "strike price"]].shift(-params["bal_per"])
    != df[["exdate", "strike price"]]
).any(axis=1)
df["changed_fltr"] = df["option_changed"].shift(-params["bal_per"])

# Scale next_S, del_S by S
df["scal_next_S"] = df["next_S"] / df["underlying price"]
df["scal_del_S"] = df["scal_next_S"] - 1

# Scale f, del_f by next_S
# df["scal_f"] = df["option price"] / df["underlying price"]
# df["scal_next_f"] = df["next_f"] / df["underlying price"]
df["scal_del_f"] = (df["next_f"] - df["option price"]) / df["underlying price"]

# # Scale f, del_f, by next_f
# df["option price"] = df["option price"] / df["prev_f"]
# df["prev_f"] = 1
# df["del_f"] = (
#     df["option price"] - df["prev_f"]
# )

# Check impact if commented
# df["T"] = df["time to maturity"]
df["T"] = df["time to maturity"] / params["T_days"]
df["scal_err_del"] = df["scal_del_f"] - df["delta"] * df["scal_del_S"]
df["err_del"] = df["del_f"] - df["delta"] * df["del_S"]
df["regr_term"] = (df["vega"] / np.sqrt(df["T"])) * df["scal_del_S"]
# df["regr_y"] = (
#     df["err_del"] / df["regr_term"]
# )  # TODO: Swith with below based on results
df["regr_y"] = df["scal_err_del"] / df["regr_term"]
df["mth_yr"] = pd.to_datetime(df["quote_date"]).dt.to_period("M").astype(str)
mth_ids = df["mth_yr"].unique().astype(str)
mth_dict = dict(zip(list(mth_ids), range(0, len(mth_ids))))
df["mth_id"] = df["mth_yr"].map(mth_dict)

#%%
print("Filtering data")
df.dropna(subset=["regr_term", "regr_y"], inplace=True)
df = df[df["changed_fltr"] == False]
df = df[df["option price"] > 0]
df = df[df["next_f"] > 0]
df = df[df["time to maturity"] > 14]
if params["type"] == "C":
    df = df[(df["delta"] > 0.05) & (df["delta"] < 0.95)]
elif params["type"] == "P":
    df = df[(df["delta"] < -0.05) & (df["delta"] > -0.95)]
else:
    raise ValueError("Incorrect input for option type.")
df.reset_index(drop=True, inplace=True)

#%%
print("Fitting curves")
df_fit = df.copy()
df_fit["a"] = np.nan
df_fit["b"] = np.nan
df_fit["c"] = np.nan
len_mths = len(mth_dict.values())

for mth in trange(params["lookback_mths"], len_mths):
    last_mths = list(range(max(0, mth - params["lookback_mths"]), mth))
    fit_rows = df_fit[df_fit["mth_id"].isin(last_mths)].index
    mth_rows = df_fit[df_fit["mth_id"] == mth].index

    poly_fit = polyfit(
        df_fit["delta"].iloc[fit_rows],
        df_fit["regr_y"].iloc[fit_rows],
        deg=2,
    )

    df_fit["a"].iloc[mth_rows] = poly_fit[0]
    df_fit["b"].iloc[mth_rows] = poly_fit[1]
    df_fit["c"].iloc[mth_rows] = poly_fit[2]

df_fit["quad_fnc"] = (
    df_fit["c"] + df_fit["b"] * df_fit["delta"] + df_fit["a"] * df_fit["delta"] ** 2
)

df_fit["mv_delta"] = (
    df_fit["delta"]
    + (df_fit["vega"] / (df_fit["underlying price"] * np.sqrt(df_fit["T"])))
    * df_fit["quad_fnc"]
)

df_fit = df_fit[
    df_fit["mth_id"] >= params["lookback_mths"]
]  # Remove rows that are missing prediction

#%%
print("Calculating error")
# df_fit["scal_err_mv"] = (
#     df_fit["scal_del_f"] - df_fit["mv_delta"] * df_fit["del_S"]
# )

df_fit["err_mv"] = (
    df_fit["del_f"]
    - df_fit["mv_delta"] * df_fit["del_S"]
    - df_fit["regr_term"] * df_fit["quad_fnc"]
)

df_fit["adj_err_del"] = df_fit["err_del"] * df_fit["underlying price"]
df_fit["adj_err_mv"] = df_fit["err_mv"] * df_fit["underlying price"]

gain = 1 - ((df_fit["err_mv"] ** 2).sum() / (df_fit["err_del"] ** 2).sum())

adj_gain = 1 - ((df_fit["adj_err_mv"] ** 2).sum() / (df_fit["adj_err_del"] ** 2).sum())

gain_mae = 1 - (abs(df_fit["adj_err_mv"]).sum() / abs(df_fit["adj_err_del"]).sum())

cols = [
    "quote_date",
    "root",
    "exdate",
    "strike price",
    "underlying price",
    "option price",
    "gamma",
    "vega",
    "rho",
    "implied volatility",
    "time to maturity",
    "T",
    "del_f",
    "del_S",
    "next_f",
    "next_S",
    "scal_next_S",
    "scal_del_S",
    "scal_del_f",
    "regr_term",
    "regr_y",
    "a",
    "b",
    "c",
    "quad_fnc",
    "delta",
    "mv_delta",
    "scal_err_del",
    "err_del",
    "err_mv",
    "mth_yr",
    "mth_id",
]
df_fit = df_fit[cols]

# df.dropna(subset=["err_mv", "regr_y"], inplace=True)

#%%
print("Pushing to excel")
df_fit.iloc[0 : params["excel_rows"]].to_excel(
    "results/" + params["type"] + "_" + str(round(gain, 2)) + ".xlsx"
)

#%%
print("Saving coeff plot")
df_plt = df_fit[["mth_id", "mth_yr", "a", "b", "c"]]
df_plt.drop_duplicates(subset="mth_id", inplace=True)
print(len(df_plt))
df_plt["-b"] = -df_plt["b"]
df_plt = df_plt[["mth_yr", "a", "-b", "c"]]
df_plt.plot(figsize=(10, 5), grid=True)
plt.savefig("results/polyfit_coef_" + str(round(gain, 2)) + ".png")
