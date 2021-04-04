# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from tqdm import trange

pd.set_option("mode.chained_assignment", None)

#%%
params = {
    "path": ["train_data/SPX_C.csv", "train_data/SPX_P.csv"],
    "bal_per": 1,
    "lookback_mths": 12,
    "T_days": 252,
    "tickers": [],
    "excel_rows": 100000,
    "debug": 50,
}

#%%
df = pd.DataFrame()

for path in params["path"]:
    df_ph = pd.read_csv(path)
    df_ph["type"] = path[path.find(".csv") - 1]
    df = df.append(df_ph)

#%%
df["mth_yr"] = pd.to_datetime(df["quote_date"]).dt.to_period("M").astype(str)
mth_ids = df["mth_yr"].unique().astype(str)
mth_dict = dict(zip(list(mth_ids), range(0, len(mth_ids))))
df["mth_id"] = df["mth_yr"].map(mth_dict)

if params["debug"] > 0:
    df = df[df["mth_id"] < params["debug"] + params["lookback_mths"]]

len_mths = len(df["mth_id"].unique())

#%%
if len(params["tickers"]) > 0:
    df = df[df["root"].isin(params["tickers"])]

df["time to maturity"] = df["time to maturity"].astype("int32")

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
df["regr_y"] = df["scal_err_del"] / df["regr_term"]

#%%
df = df[:-1]

df = df[df["date_diff"] <= 4]

df = df[df["option_changed"] == False]
df = df[df["time to maturity"] >= 14]

df_1 = df[(df["type"] == "C") & (df["delta"] > 0.05) & (df["delta"] < 0.95)]
df_2 = df[(df["type"] == "P") & (df["delta"] < -0.05) & (df["delta"] > -0.95)]
df = df_1.append(df_2)

#%% Create parameter df
df.reset_index(drop=True, inplace=True)
df["a"] = np.inf
df["b"] = np.inf
df["c"] = np.inf

#%%
longst_mat = max(df["time to maturity"])
longst_mat
bucket_mat = [
    [14, 30],
    [31, 60],
    [61, 91],
    [92, 122],
    [123, 182],
    [183, 365],
    [366, longst_mat],
]
bucket_del = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def range_incl(a, b, add=1):
    return range(a, b + add)


#%%
for mth in trange(params["lookback_mths"], len_mths, desc="Fitting regr"):
    for delta in bucket_del:
        for maturity in bucket_mat:

            df_regr = df[round(df["delta"], 1) == delta].copy()
            df_regr = df_regr[
                df_regr["time to maturity"].isin(range_incl(maturity[0], maturity[1]))
            ]

            last_mths = list(range(max(0, mth - params["lookback_mths"]), mth))
            fit_rows = df_regr[df_regr["mth_id"].isin(last_mths)].index
            prd_rows = df_regr[df_regr["mth_id"] == mth].index

            if (len(fit_rows) > 0) & (len(prd_rows) > 0):
                poly = Polynomial([1, 1, 1])
                a, b, c = poly.fit(
                    df["delta"].iloc[fit_rows], df["regr_y"].iloc[fit_rows], deg=2
                )

                df["a"].iloc[prd_rows] = a
                df["b"].iloc[prd_rows] = b
                df["c"].iloc[prd_rows] = c

#%%
df = df[df["mth_id"] >= params["lookback_mths"]]

#%%
if params["excel_rows"] > 0:
    rnd_id = np.random.randint(low=0, high=(len(df) - params["excel_rows"]), size=1)[0]
    df.iloc[rnd_id : (rnd_id + params["excel_rows"])].to_csv("results/bfr_filtr.csv")

#%%
len1 = len(df)
df.replace(to_replace=[np.inf, -np.inf], value=[np.nan, np.nan], inplace=True)
df.dropna(inplace=True)
len2 = len(df)
print("Removed {} rows with inf, -inf, or n/a".format(len1 - len2))

#%%
df["quad_fnc"] = df["a"] + (df["b"] * df["delta"]) + (df["c"] * (df["delta"] ** 2))

df["mv_delta"] = (
    df["delta"]
    + (df["vega"] / (df["underlying price"] * np.sqrt(df["T"]))) * df["quad_fnc"]
)

df["err_mv"] = (
    df["del_f"] - (df["delta"] * df["del_S"]) - (df["regr_term"] * df["quad_fnc"])
)

df["err_del"] = df["del_f"] - (df["delta"] * df["del_S"])

#%%
gain = 1 - ((df["err_mv"] ** 2).sum() / (df["err_del"] ** 2).sum())
print("Resulting gain: {}%".format(round(gain * 100, 2)))

#%%
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
    "next_S",
    "next_f",
    "scal_next_S",
    "scal_next_f",
    "del_S",
    "del_f",
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
    "type",
    "mth_yr",
    "mth_id",
]
df = df[cols]

#%%
if params["excel_rows"] > 0:
    rnd_id = np.random.randint(low=0, high=(len(df) - params["excel_rows"]), size=1)[0]
    df.iloc[rnd_id : (rnd_id + params["excel_rows"])].to_csv(
        "results/" + "_" + str(round(gain, 2)) + ".csv"
    )

#%%
df_plt = df[["mth_id", "mth_yr", "a", "b", "c"]]
df_plt.drop_duplicates(subset="mth_id", inplace=True)
df_plt["-b"] = -df_plt["b"]
df_plt = df_plt[["mth_yr", "a", "-b", "c"]]
df_plt.plot(figsize=(10, 5), grid=True)
plt.savefig("results/polyfit_coef_" + str(round(gain, 2)) + ".png")
