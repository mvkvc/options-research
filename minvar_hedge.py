# %%
# import dask.dataframe as dd
from datetime import datetime
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import trange

# %%
filepath = "SPX_C_train_filterdelta.csv"
rebalance_interval = 14
starting_cash = 10000
interest_rate = 0.05
dividend_yield = 0.05
contract_multiplier = 1

# %%
def calc_del_bs(S, K, T, r, sigma, option_type="call"):

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        del_bs = np.exp(-dividend_yield * T) * sp.stats.norm.cdf(d1)

    elif option_type == "put":
        del_bs = np.exp(-dividend_yield * T) * (sp.stats.norm.cdf(d1) - 1)

    else:
        raise ValueError("Not a valid option type.")

    return del_bs


# %%
def calc_vega_bs(S, T, del_bs):

    vega_bs = S * np.sqrt(T) * (sp.stats.gamma.cdf(del_bs))

    return vega_bs


# %%
def calc_del_mv(S, T, del_bs, vega_bs, a=1, b=1, c=1):

    # TODO Check OOO
    del_mv = del_bs + (vega_bs / (S * np.sqrt(T))) * (a + b * del_bs + c * del_bs ^ 2)

    return del_mv


# %%
def sim_del_hedge(parameters):

    with open(filepath) as f:
        num_rows = sum(1 for line in f)

    df = pd.read_csv(filepath)
    port_error = []

    for row in trange(num_rows):

        row_acc = 1

        if row_acc == 1:

            start_row = row
            port_shares = 0
            port_cash = starting_cash

        if row_acc % rebalance_interval == 0:

            S = df.underlying_price[row]
            K = df.strike[row]
            date = datetime.strptime(df.quote_date[row], "%Y-%m-%d").date()
            expiration = datetime.strptime(df.expiration[row], "%Y-%m-%d").date()
            T = (expiration - date).days  # TODO Check format
            r = interest_rate
            sigma = np.sqrt(np.var(df.underlying_price[start_row:row]))

            del_bs = calc_del_bs(S, K, T, r, sigma)
            vega_bs = calc_vega_bs(S, T, del_bs)

            del_mv = calc_del_mv(S, T, del_bs, vega_bs)

            num_buysell = del_mv * contract_multiplier
            port_shares = port_shares + num_buysell
            port_cash = port_cash + num_buysell * df.row.S

        error = (
            df.underlying_price[row] - (port_cash + port_shares * df.strike[row])
        ) / df.underlying_price[row]
        port_error.append(error)

        # df[['strike', 'underlying_price']].iloc[50].values
        if df[K, data][row] == df[K, data][row + 1]:
            row_acc += 1
        else:
            # TODO
            row_acc = 1

    avg_error = np.average(port_error)

    return avg_error


# %%
def main(self, parameter_list):
    sim_del_hedge()


# %%
if __name__ == "__main__":
    main()
