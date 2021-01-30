# %%
# from datetime import datetime, timedelta
# import numpy as np
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
# from scipy import stats
# from tqdm import trange


contracts = "SPX_C_train_filterdelta.csv"
df = pd.read_csv(filepath=contracts)

imp_v = iv(price, S, K, t, r, flag)




df['Flag'] = 'C'

df_contracts = price_dataframe(df, flag_col='Flag', underlying_price_col='S', strike_col='K', annualized_tte_col='T',
                     riskfree_rate_col='R', sigma_col='IV', model='black_scholes', inplace=False)


# %%
filepath = "SPX_C_train_filterdelta.csv"
filepath_prices = "SPX_prices.csv"
rebalance_interval = 14
starting_cash = 10000 # No accounting for running out of money currently
interest_rate = 0.05
dividend_yield = 0.05
contract_multiplier = 1
vol_lookback = 50
gamma_a = 1

# %%
def calc_del_bs(S, K, T, r, sigma, dividend_yield, option_type="call"):

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        del_bs = np.exp(-dividend_yield * T) * stats.norm.cdf(d1)

    elif option_type == "put":
        del_bs = np.exp(-dividend_yield * T) * (stats.norm.cdf(d1) - 1)

    else:
        raise ValueError("Not a valid option type.")

    return del_bs


# %%
def calc_vega_bs(S, T, del_bs, gamma_a):

    vega_bs = S * np.sqrt(T) * (stats.gamma.cdf(del_bs, gamma_a))

    return vega_bs


# %%
def calc_del_mv(S, T, del_bs, vega_bs, a=1, b=1, c=1):

    del_mv = del_bs + (vega_bs / (S * np.sqrt(T))) * (a + b * del_bs + c * del_bs ** 2)

    return del_mv


# %%
def sim_del_hedge(
    filepath=filepath,
    filepath_prices=filepath_prices,
    rebalance_interval=rebalance_interval,
    starting_cash=starting_cash,
    interest_rate=interest_rate,
    dividend_yield=dividend_yield,
    contract_multiplier=contract_multiplier,
    vol_lookback=vol_lookback,
    gamma_a=gamma_a
):

    with open(filepath) as f:
        num_rows = sum(1 for line in f)

    df = pd.read_csv(filepath)
    prices = pd.read_csv(filepath_prices)
    df["port_val"] = 0
    df["port_error"] = 0
    prices = pd.read_csv(filepath_prices)

    start_row = 1
    port_cash = starting_cash
    port_shares = 0
    
    # for row in range(start_row, 10):
    for row in trange(start_row, num_rows):

        print(row)

        compare = (
            df[["expiration", "strike"]].iloc[row]
            == df[["expiration", "strike"]].iloc[row - 1]
        )
        if compare.all():
            pass
        else:
            start_row = row
            port_cash = starting_cash
            port_shares = 0

        S = df.underlying_price[row]
        K = df.strike[row]
        r = interest_rate
        today = datetime.strptime(df.quote_date[row], "%Y-%m-%d").date()
        expiration = datetime.strptime(df.expiration[row], "%Y-%m-%d").date()
        T = (expiration - today).days / 365

        if row == start_row or (row - start_row) % rebalance_interval == 0:

            price_today = prices.index[prices["quote_date"] == today.strftime("%#m/%#d/%Y")].to_list()[0] # Replace # in strftime with - in Unix
            if price_today < vol_lookback:
                price_lookback = 0
            else:
                price_lookback = price_today - vol_lookback
            sigma = np.sqrt(np.var(prices.underlying_price[price_lookback:price_today+1]))

            del_bs = df.delta[row]
            # del_bs = calc_del_bs(S, K, T, r, sigma, dividend_yield)
            vega_bs = calc_vega_bs(S, T, del_bs, gamma_a)
            del_mv = calc_del_mv(S, T, del_bs, vega_bs)

            port_shares = port_shares + (del_mv * contract_multiplier)
            port_cash = port_cash - (del_mv * contract_multiplier) * S
    
        df.port_val[row] = port_shares * S

        if start_row == row:
            port_error = 0
        else:
            chng_option = (df.option_price[row]  - df.option_price[row - 1])
            chng_port = (df.port_val[row] - df.port_val[row - 1])
            port_error = (chng_option - chng_port) / chng_option
            print(chng_option, chng_port)

        df.port_error[row] = port_error
        print(port_error)
        print(row == start_row)

    return df


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