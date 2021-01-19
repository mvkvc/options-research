import dask.dataframe as dd
import numpy as np
import scipy as sp
from scipy.stats import norm, gamma
from tqdm import trange

filepath = "*.csv"
rebalance_interval = 14
starting_cash = 10000
interest_rate = 0.05
contract_multiplier = 1


def calc_del_bs(S, K, T, r, sigma, option_type="call"):

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        del_bs = norm(d1)

    elif option_type == "put":
        del_bs = norm(d1) - 1

    else:
        raise ValueError("Not a valid option type.")

    return del_bs


def calc_vega_bs(S, T, del_bs):

    vega_bs = S * np.sqrt(T) * gamma(del_bs)

    return vega_bs


def calc_del_mv(S, T, del_bs, vega_bs, a=1, b=1, c=1):

    # TODO Check OOO
    del_mv = del_bs + (vega_bs / (S * np.sqrt(T))) * (a + b * del_bs + c * del_bs ^ 2)

    return del_mv


def sim_del_hedge(parameters):
    with open(filepath) as f:
        num_rows = sum(1 for line in f)

    # TODO Initiate delta hedging portfolio
    row_acc = 1
    pf_shares = 0
    pf_cash = starting_cash
    pf_error = []

    for row in trange(num_rows):

        if row_acc == 1:
            #Init delta hedging for new option
        else:
            pass

        if row_acc % rebalance_interval == 0:
            S = df.S[row]
            K = df.K[row]
            T = df.T[row]
            r = interest_rate
            sigma = np.var(df.S[start_row:row])

            del_bs = calc_del_bs(S, K, T, r, sigma)
            vega_bs = calc_vega_bs(S, T, del_bs)

            del_mv = calc_del_mv(S, T, del_bs, vega_bs)

            error = (df.P.row - (pf_cash + pf_shares * df.row.S)) / df.P.row
            pf_error.append(error)

            num_buysell = del_mv * contract_multiplier
            pf_shares = pf_shares + num_buysell
            pf_cash = pf_cash + num_buysell * df.row.S

        else:
            pass

        if df[K, data][row] == df[K, data][row + 1]:
            row_acc += 1
        else:
            # TODO Write results to np.series
            start_row = row + 1
            row_acc = 1


def main(self, parameter_list):
    pass


if __name__ == "__main__":
    main()
