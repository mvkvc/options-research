# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd

pd.set_option("mode.chained_assignment", None)

#%%
tickers = []

#%%
# path = "train_data/SPX_P.csv" # TODO: Combine put and call files
path = "train_data/SPX_C.csv"
params = {
    # "type": "P",
    "type": "C",
    "bal_per": 1,
    "lookback_mths": 12,
    "T_days": 252,
    "excel_rows": 10000,
}

#%%
df = pd.read_csv(path)

#%%
df["option_changed"] = (
        df[["exdate", "strike price"]].shift(-params["bal_per"])
        != df[["exdate", "strike price"]]
).any(axis=1)

df["next_S"] = df["underlying price"].shift(-params["bal_per"])
df["next_f"] = df["option price"].shift(-params["bal_per"])

df["del_S"] = df["underlying price"].diff().shift(-params["bal_per"])
df["del_f"] = df["option price"].diff().shift(-params["bal_per"]) # TODO: Sample 100 random and confirm they line up

#%%
# Scale next_S, del_S by S
df["scal_S"] = 1
df["scal_next_S"] = df["next_S"] / df["underlying price"]
df["scal_del_S"] = df["scal_next_S"] - df["scal_S"]

# Scale f, del_f by next_S
df["scal_f"] = df["option price"] / df["underlying price"]
df["scal_del_f"] = (df["next_f"] - df["option price"]) / df["underlying price"]

#%%
df["T"] = df["time to maturity"] / params["T_days"]
df["err_del"] = df["scal_del_f"] - df["delta"] * df["scal_del_S"]
df["regr_term"] = (df["vega"] / np.sqrt(df["T"])) * df["scal_del_S"]
df["regr_y"] = df["err_del"] / df["regr_term"]

#%%
df["mth_yr"] = pd.to_datetime(df["quote_date"]).dt.to_period("M").astype(str)
mth_ids = df["mth_yr"].unique().astype(str)
mth_dict = dict(zip(list(mth_ids), range(0, len(mth_ids))))
df["mth_id"] = df["mth_yr"].map(mth_dict)

#%%
df[["regr_term", "regr_y"]].replace(
    [np.inf, -np.inf], np.nan
)  # TODO: Evaluate removing inf now or later
df.dropna(subset=["regr_term", "regr_y"], inplace=True) #TODO: Count removed vs remaining

#%%
df = df[df["option_change"] == False]
df = df[df["time to maturity"] >= 14]

if params["type"] == "C":
    df = df[(df["delta"] > 0.05) & (df["delta"] < 0.95)]
elif params["type"] == "P":
    df = df[(df["delta"] < -0.05) & (df["delta"] > -0.95)]
else:
    raise ValueError("Incorrect option type.")

if len(tickers) > 0:
    df = df[df["root"].isin(tickers)]

# df = df[df["option price"] > 0]
# df = df[df["next_f"] > 0] # TODO: Test without

#%%
df["a"] = np.nan
df["b"] = np.nan
df["c"] = np.nan
len_mths = len(mth_dict.values())

for mth in range(params["lookback_mths"], len_mths):
    last_mths = list(range(max(0, mth - params["lookback_mths"]), mth))
    fit_rows = df[df["mth_id"].isin(last_mths)].index
    mth_rows = df[df["mth_id"] == mth].index

    # no_outl = df["regr_y"].iloc[fit_rows]
    # outl_mean = np.mean(no_outl)
    # outl_std = np.std(no_outl)
    #
    # tol_std = 2.576
    # no_outl = no_outl[no_outl > outl_mean - tol_std * outl_std]
    # no_outl = no_outl[no_outl < outl_mean + tol_std * outl_std]
    # fit_rows = no_outl.index

    poly_fit = polyfit(
        df["delta"].iloc[fit_rows],
        df["regr_y"].iloc[fit_rows],
        deg=2,
    )

    df["a"].iloc[mth_rows] = poly_fit[0]
    df["b"].iloc[mth_rows] = poly_fit[1]
    df["c"].iloc[mth_rows] = poly_fit[2]

#%%
df["quad_fnc"] = df["c"] + df["b"] * df["delta"] + df["a"] * df["delta"] ** 2

df["mv_delta"] = df["delta"] + df["vega"] / (df["scal_S"] * np.sqrt(df["T"])) * df["quad_fnc"]

df = df[
    df["mth_id"] >= params["lookback_mths"]
]  # Remove rows that are missing prediction

# df.dropna(subset=["a", "b", "c"], inplace=True)

#%%
# df["scal_err_mv"] = (
#     df["scal_del_f"] - df["mv_delta"] * df["del_S"]
# )

df["err_mv"] = (
    df["del_f"] - df["mv_delta"] * df["del_S"] - df["regr_term"] * df["quad_fnc"]
)

gain = 1 - ((df["err_mv"] ** 2).sum() / (df["err_del"] ** 2).sum())

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
    "err_del",
    "err_mv",
    "mth_yr",
    "mth_id",
]
df = df[cols]

# df.dropna(subset=["err_mv", "regr_y"], inplace=True)

#%%
df.iloc[0 : params["excel_rows"]].to_excel(
    "results/" + params["type"] + "_" + str(round(gain, 2)) + ".xlsx"
)

#%%
df_plt = df[["mth_id", "mth_yr", "a", "b", "c"]]
df_plt.drop_duplicates(subset="mth_id", inplace=True)
df_plt["-b"] = -df_plt["b"]
df_plt = df_plt[["mth_yr", "a", "-b", "c"]]
df_plt.plot(figsize=(10, 5), grid=True)
plt.savefig("results/polyfit_coef_" + str(round(gain, 2)) + ".png")
