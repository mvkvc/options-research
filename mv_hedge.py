# Test different setting for "time to maturity"/"T"
# Plot regression values a, b, c

# %%
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
from tqdm import trange

pd.set_option("mode.chained_assignment", None)

# %%
class mv_hedge:
    def __init__(self, params, path):
        print("Importing data")
        self.path = path
        self.params = params
        self.df_raw = pd.read_csv(path)
        self.df = self.df_raw.copy()

    def transform(self):
        print("Transforming data")
        # self.df.drop(columns=["theta"], inplace=True)
        bal_per = self.params["bal_per"]
        self.df["del_S"] = self.df["underlying price"].diff().shift(-bal_per)
        self.df["next_S"] = self.df["underlying price"].shift(-bal_per)
        self.df["del_f"] = self.df["option price"].diff().shift(-bal_per)
        self.df["next_f"] = self.df["option price"].shift(-bal_per)

        self.df["option_changed"] = (
            self.df[["exdate", "strike price"]].shift(-bal_per)
            != self.df[["exdate", "strike price"]]
        ).any(axis=1)
        self.df["changed_fltr"] = self.df["option_changed"].shift(-bal_per)

        # Scale next_S, del_S by S
        self.df["scal_next_S"] = self.df["next_S"] / self.df["underlying price"]
        self.df["scal_del_S"] = self.df["scal_next_S"] - 1

        # Scale f, del_f by next_S
        # self.df["scal_f"] = self.df["option price"] / self.df["underlying price"]
        # self.df["scal_next_f"] = self.df["next_f"] / self.df["underlying price"]
        self.df["scal_del_f"] = (self.df["next_f"] - self.df["option price"]) / self.df[
            "underlying price"
        ]

        # # Scale f, del_f, by next_f
        # self.df["option price"] = self.df["option price"] / self.df["prev_f"]
        # self.df["prev_f"] = 1
        # self.df["del_f"] = (
        #     self.df["option price"] - self.df["prev_f"]
        # )

        # Check impact if commented
        # self.df["T"] = self.df["time to maturity"]
        self.df["T"] = self.df["time to maturity"] / self.params["T_days"]
        self.df["scal_err_del"] = (
            self.df["scal_del_f"] - self.df["delta"] * self.df["scal_del_S"]
        )
        self.df["err_del"] = self.df["del_f"] - self.df["delta"] * self.df["del_S"]
        self.df["regr_term"] = (self.df["vega"] / np.sqrt(self.df["T"])) * self.df[
            "scal_del_S"
        ]
        # self.df["regr_y"] = (
        #     self.df["err_del"] / self.df["regr_term"]
        # )  # TODO: Swith with below based on results
        self.df["regr_y"] = self.df["scal_err_del"] / self.df["regr_term"]
        self.df["mth_yr"] = (
            pd.to_datetime(self.df["quote_date"]).dt.to_period("M").astype(str)
        )
        mth_ids = self.df["mth_yr"].unique().astype(str)
        self.mth_dict = dict(zip(list(mth_ids), range(0, len(mth_ids))))
        self.df["mth_id"] = self.df["mth_yr"].map(self.mth_dict)

    def fltr_rows(self):
        print("Filtering data")
        self.df.dropna(subset=["regr_term", "regr_y"], inplace=True)
        self.df = self.df[self.df["changed_fltr"] == False]
        self.df = self.df[self.df["option price"] > 0]
        self.df = self.df[self.df["next_f"] > 0]
        self.df = self.df[self.df["time to maturity"] > 14]
        if self.params["type"] == "C":
            self.df = self.df[(self.df["delta"] > 0.05) & (self.df["delta"] < 0.95)]
        elif self.params["type"] == "P":
            self.df = self.df[(self.df["delta"] < -0.05) & (self.df["delta"] > -0.95)]
        else:
            raise ValueError("Incorrect input for option type.")
        self.df.reset_index(drop=True, inplace=True)

    def fit_curve(self):
        print("Fitting curves")
        self.df_fit = self.df.copy()
        self.df_fit["a"] = np.nan
        self.df_fit["b"] = np.nan
        self.df_fit["c"] = np.nan
        len_mths = len(self.mth_dict.values())

        for mth in trange(self.params["lookback_mths"], len_mths):
            last_mths = list(range(max(0, mth - self.params["lookback_mths"]), mth))
            fit_rows = self.df_fit[self.df_fit["mth_id"].isin(last_mths)].index
            mth_rows = self.df_fit[self.df_fit["mth_id"] == mth].index

            poly_fit = polyfit(
                self.df_fit["delta"].iloc[fit_rows],
                self.df_fit["regr_y"].iloc[fit_rows],
                deg=2,
            )

            self.df_fit["a"].iloc[mth_rows] = poly_fit[0]
            self.df_fit["b"].iloc[mth_rows] = poly_fit[1]
            self.df_fit["c"].iloc[mth_rows] = poly_fit[2]

        self.df_fit["quad_fnc"] = (
            self.df_fit["c"]
            + self.df_fit["b"] * self.df_fit["delta"]
            + self.df_fit["a"] * self.df_fit["delta"] ** 2
        )

        self.df_fit["mv_delta"] = (
            self.df_fit["delta"]
            + (
                self.df_fit["vega"]
                / (self.df_fit["underlying price"] * np.sqrt(self.df_fit["T"]))
            )
            * self.df_fit["quad_fnc"]
        )

        self.df_fit = self.df_fit[
            self.df_fit["mth_id"] >= self.params["lookback_mths"]
        ]  # Remove rows that are missing prediction

    def calc_errr(self):
        print("Calculating error")
        # self.df_fit["scal_err_mv"] = (
        #     self.df_fit["scal_del_f"] - self.df_fit["mv_delta"] * self.df_fit["del_S"]
        # )

        self.df_fit["err_mv"] = (
            self.df_fit["del_f"]
            - self.df_fit["mv_delta"] * self.df_fit["del_S"]
            - self.df_fit["regr_term"] * self.df_fit["quad_fnc"]
        )

        self.df_fit["adj_err_del"] = (
            self.df_fit["err_del"] * self.df_fit["underlying price"]
        )
        self.df_fit["adj_err_mv"] = (
            self.df_fit["err_mv"] * self.df_fit["underlying price"]
        )

        self.gain = 1 - (
            (self.df_fit["err_mv"] ** 2).sum() / (self.df_fit["err_del"] ** 2).sum()
        )

        self.adj_gain = 1 - (
            (self.df_fit["adj_err_mv"] ** 2).sum()
            / (self.df_fit["adj_err_del"] ** 2).sum()
        )

        self.gain_mae = 1 - (
            abs(self.df_fit["adj_err_mv"]).sum() / abs(self.df_fit["adj_err_del"]).sum()
        )

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
        self.df_fit = self.df_fit[cols]

        self.df.dropna(subset=["mv_error", "regr_y"], inplace=True)

    def ret_excel(self):
        print("Pushing to excel")
        self.df_fit.iloc[0 : self.params["excel_rows"]].to_excel(
            "results/" + self.params["type"] + "_" + str(round(self.gain, 2)) + ".xlsx"
        )

    def save_plot(self):
        print("Saving coeff plot")
        df_plt = self.df_fit[["mth_id", "mth_yr", "a", "b", "c"]]
        df_plt.drop_duplicates(subset="mth_id", inplace=True)
        print(len(df_plt))
        df_plt["-b"] = -df_plt["b"]
        df_plt = df_plt[["mth_yr", "a", "-b", "c"]]
        df_plt.plot(figsize=(10, 5), grid=True)
        plt.savefig("results/polyfit_coef_" + str(round(self.gain, 2)) + ".png")


# %%
if __name__ == "__main__":
    # path = "train_data/SPX_P.csv"
    path = "train_data/SPX_C.csv"
    params = {
        # "type": "P",
        "type": "C",
        "bal_per": 1,
        "lookback_mths": 12,
        "T_days": 252,
        "excel_rows": 10000,
    }

    hedge = mv_hedge(params, path)
    hedge.transform()
    hedge.fltr_rows()
    hedge.fit_curve()
    hedge.calc_errr()
    hedge.ret_excel()
    hedge.save_plot()