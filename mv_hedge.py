#%% Generate extract to check formulas
# rnd_id = np.random.randint(low=0, high=(len(df) - params["excel_rows"]), size=1)[0]
# df.iloc[rnd_id : (rnd_id + params["excel_rows"])].to_csv("check_data.csv")
# df["err_del"] = df["del_f"] - (df["delta"] * df["del_S"])
# tickers=[]
# if len(tickers) > 0:
# df = df[df["root"].isin(tickers)]

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd

# pd.set_option("mode.chained_assignment", None)

#%%
params = {
    "path": "train_data/SPX_C.csv",
    "type": "C",
    "bal_per": 1,
    "lookback_mths": 12,
    "T_days": 252,
    "excel_rows": 100000,
}

#%%
df = pd.read_csv(params["path"])

#%%
df["option_changed"] = (
    df[["exdate", "strike price"]].shift(-params["bal_per"])
    != df[["exdate", "strike price"]]
).any(axis=1)

#%%
df["date_diff"] = np.array(
    pd.to_datetime(df["quote_date"]).diff().shift(-params["bal_per"]), dtype=np.int16
)

#%%
df["next_S"] = df["underlying price"].shift(-params["bal_per"])
df["next_f"] = df["option price"].shift(-params["bal_per"])

#%%
df["del_S"] = df["underlying price"].diff().shift(-params["bal_per"])
df["del_f"] = df["option price"].diff().shift(-params["bal_per"])

#%% Scale next_S, del_S by S
df["scal_next_S"] = df["next_S"] / df["underlying price"]
df["scal_del_S"] = df["scal_next_S"] - 1

#%% Scale f, del_f by next_S
df["scal_f"] = df["option price"] / df["underlying price"]
df["scal_next_f"] = df["next_f"] / df["underlying price"]
df["scal_del_f"] = df["scal_next_f"] - df["scal_f"]

#%%
df["T"] = df["time to maturity"] / params["T_days"]
df["scal_err_del"] = df["scal_del_f"] - (df["delta"] * df["scal_del_S"])
df["regr_term"] = (df["vega"] / np.sqrt(df["T"])) * df["scal_del_S"]
df["regr_y"] = df["err_del"] / df["regr_term"]

#%%
df["mth_yr"] = pd.to_datetime(df["quote_date"]).dt.to_period("M").astype(str)
mth_ids = df["mth_yr"].unique().astype(str)
mth_dict = dict(zip(list(mth_ids), range(0, len(mth_ids))))
df["mth_id"] = df["mth_yr"].map(mth_dict)

#%%
df = df[:-1]

df = df[df["date_diff"] < 5]

df = df[df["option_changed"] == False]
df = df[df["time to maturity"] >= 14]

if params["type"] == "C":
    df = df[(df["delta"] > 0.05) & (df["delta"] < 0.95)]
elif params["type"] == "P":
    df = df[(df["delta"] < -0.05) & (df["delta"] > -0.95)]
else:
    raise ValueError("Incorrect option type.")

#%% Create parameter df
columns_param = ["delta", "maturity", "mth_id"]
df_param = pd.DataFrame(columns=[])


df["a"] = np.nan
df["b"] = np.nan
df["c"] = np.nan
len_mths = len(mth_dict.values())
df.reset_index(drop=True, inplace=True)

for mth in range(params["lookback_mths"], len_mths):
    last_mths = list(range(max(0, mth - params["lookback_mths"]), mth))
    fit_rows = df[df["mth_id"].isin(last_mths)].index
    mth_rows = df[df["mth_id"] == mth].index

    # poly_fit = polyfit(
    #     df["delta"].iloc[fit_rows],
    #     df["regr_y"].iloc[fit_rows],
    #     deg=2,
    # )

    poly_fit = polyfit(
        df["delta"].iloc[fit_rows],
        df["regr_y"].iloc[fit_rows],
        deg=2,
    )

    df["a"].iloc[mth_rows] = poly_fit[0]
    df["b"].iloc[mth_rows] = poly_fit[1]
    df["c"].iloc[mth_rows] = poly_fit[2]

#%%
df = df[df["mth_id"] >= params["lookback_mths"]]
df[["regr_term", "regr_y"]].replace(
    [np.inf, -np.inf], np.nan
)  # TODO: Evaluate removing inf now or later
df.dropna(subset=["regr_term", "regr_y", "a", "b", "c"], inplace=True)

#%%
df["quad_fnc"] = df["c"] + (df["b"] * df["delta"]) + (df["a"] * (df["delta"] ** 2))
# df["quad_fnc"] = df["a"] + (df["b"] * df["delta"]) + (df["c"] * (df["delta"] ** 2))

# TODO: Check formula
df["mv_delta"] = (
    df["delta"]
    + (df["vega"] / (df["underlying price"] * np.sqrt(df["T"]))) * df["quad_fnc"]
)


df["err_mv"] = (
    df["del_f"] - (df["delta"] * df["del_S"]) - (df["regr_term"] * df["quad_fnc"])
)


#%%
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
    "option_changed",
    "date_diff",
    "del_f",
    "del_S",
    "next_f",
    "next_S",
    "scal_next_S",
    "scal_del_S",
    "scal_next_f",
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
rnd_id = np.random.randint(low=0, high=(len(df) - params["excel_rows"]), size=1)[0]
df.iloc[rnd_id : (rnd_id + params["excel_rows"])].to_csv(
    "results/" + params["type"] + "_" + str(round(gain, 2)) + ".csv"
)

#%%
df_plt = df[["mth_id", "mth_yr", "a", "b", "c"]]
df_plt.drop_duplicates(subset="mth_id", inplace=True)
df_plt["-b"] = -df_plt["b"]
df_plt = df_plt[["mth_yr", "a", "-b", "c"]]
df_plt.plot(figsize=(10, 5), grid=True)
plt.savefig("results/polyfit_coef_" + str(round(gain, 2)) + ".png")

# %%
# TESTING HERE
# df["regr_y"].plot(figsize=(10, 5), grid=True)
# x = df[abs(df["regr_y"]) > 0.1][100000:200000]
# x.to_csv("test.csv")
