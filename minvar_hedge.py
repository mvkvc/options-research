# %%
import numpy as np
import pandas as pd
from tqdm import trange

# %%
filepath = "train_data/SPX_C.csv" # "train_data/SPX_P.csv"

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
    
    row_acc   = 0
    month_acc = 0

    for row in trange(1, num_rows):

        compare = (
            df[["exdate", "strike price"]].iloc[row]
            == df[["exdate", "strike price"]].iloc[row - 1]
        )
        if compare.all():
            row_acc =+ 1
        else:
            row_acc = 0

        if row_acc < 1080:
            pass
        elif row_acc == 1080:
            month_acc = 0
        elif row_acc > 1080:
            month_acc += 1
        else:
            # TODO: Check how to address potential error

        if month_acc == 31:
            month_acc = 0


        if month_acc == 0:
           # TODO: Recalculate a, b, c
           pass


        if row_acc >= 1080:
            # TODO: Apply calculated formula for error and record in row
            pass

        print(row, row_acc, month_acc)

        # S = df.underlying_price[row]
        # K = df.strike[row]
        # r = interest_rate
        # today = datetime.strptime(df.quote_date[row], "%Y-%m-%d").date()
        # expiration = datetime.strptime(df.expiration[row], "%Y-%m-%d").date()
        # T = (expiration - today).days / 365

        # if row == start_row or (row - start_row) % rebalance_interval == 0:

        #     price_today = prices.index[prices["quote_date"] == today.strftime("%#m/%#d/%Y")].to_list()[0] # Replace # in strftime with - in Unix
        #     if price_today < vol_lookback:
        #         price_lookback = 0
        #     else:
        #         price_lookback = price_today - vol_lookback
        #     sigma = np.sqrt(np.var(prices.underlying_price[price_lookback:price_today+1]))

        #     del_bs = df.delta[row]
        #     # del_bs = calc_del_bs(S, K, T, r, sigma, dividend_yield)
        #     vega_bs = calc_vega_bs(S, T, del_bs, gamma_a)
        #     del_mv = calc_del_mv(S, T, del_bs, vega_bs)

        #     port_shares = port_shares + (del_mv * contract_multiplier)
        #     port_cash = port_cash - (del_mv * contract_multiplier) * S
    
        # df.port_val[row] = port_shares * S

        # if start_row == row:
        #     port_error = 0
        # else:
        #     chng_option = (df.option_price[row]  - df.option_price[row - 1])
        #     chng_port = (df.port_val[row] - df.port_val[row - 1])
        #     port_error = (chng_option - chng_port) / chng_option
        #     print(chng_option, chng_port)

        # df.port_error[row] = port_error
        # print(port_error)
        # print(row == start_row)

    return


# %%
mv_hedging()
# %%
def main():

    df = sim_del_hedge(
        filepath,
        filepath_prices,
        rebalance_interval,
        starting_cash,
        interest_rate,
        dividend_yield,
        contract_multiplier,
        vol_lookback,
    )
    df.to_csv("minvar_results.csv")


# %%
if __name__ == "__main__":
    main()
    hello(sys.argv[1], sys.argv[2])