"""
CNE-6 价格类因子计算
因子清单：Daily std, Cumulative range, Short Term reversal, Seasonality,
         Industry Momentum, Relative strength, Long term relative strength,
         BETA, Hist sigma, Historical alpha, Long term historical alpha
"""

import numpy as np
import pandas as pd
from datatools import get_price, get_index_K, get_trade_cal


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def weight_sum(data, half_life, normalize=True):
    window = len(data)
    w = 0.5**(np.arange(window) / half_life)[::-1]
    if normalize:
        w /= w.sum()
    return np.nansum(w * data)
    
# ── 主函数 ────────────────────────────────────────────────────────────────────

def calc_price_factors(start_date, end_date, allstocks):
    """
    计算 CNE-6 价格类因子

    Parameters
    ----------
    start_date, end_date : str  'YYYY-MM-DD'
    allstocks : list[str]  股票池

    Returns
    -------
    DataFrame  主键 ts_code + cal_date
    """
    # ── 提前加载足够历史数据（覆盖最长窗口 1040 日 + 季节性 5 年月度）──
    load_start = (pd.Timestamp(start_date) - pd.DateOffset(month=13)).strftime('%Y-%m-%d')

    print('加载价格数据...')
    price = get_price(codes=allstocks, start_date=load_start, end_date=end_date, fields=['close', 'pre_close'])

    print('加载指数数据...')
    hs300 = get_index_K(codes=['000300.SH'], start_date=load_start, end_date=end_date, fields=['close', 'pre_close'])

    # ── 构建矩阵 ──────────────────────────────────────────────────────────────
    dates = np.sort(price['trade_date'].unique())
    universe = np.sort(price['ts_code'].unique())

    close_pivot = price.pivot(index='trade_date', columns='ts_code', values='close').reindex(index=dates, columns=universe)
    pre_pivot = price.pivot(index='trade_date', columns='ts_code', values='pre_close').reindex(index=dates, columns=universe)

    ret_mat = np.log(close_pivot/pre_pivot)          # (T, N)

    hs300_close = hs300.set_index('trade_date')['close'].reindex(dates)
    hs300_pre   = hs300.set_index('trade_date')['pre_close'].reindex(dates)
    ret_mkt = np.log(hs300_close / hs300_pre)          # (T,)

    n_dates, n_stocks = ret_mat.shape
    print(f'矩阵: {n_dates} 交易日 × {n_stocks} 只股票')

    # ================================================================
    # 1. Daily std — 252 日窗口，半衰期 42 日
    # ================================================================
    print('计算 Daily std...')
    daily_std = ret_mat.ewm(halflife=42).std().loc[start_date:end_date]

    # ================================================================
    # 2. Cumulative range — Z(T) 为过去 T 个月累积对数收益，T=1..12
    # ================================================================
    print('计算 Cumulative range...')

    rolling_cum = ret_mat.expanding().sum()
    # 方法一
    # result = {}
    # for date in get_trade_cal(start_date=start_date, end_date=end_date):
    #     today = rolling_cum.index.get_loc(date)
    #     values = rolling_cum.iloc[today] - rolling_cum.iloc[[today - i * 21 for i in range(1,13)]]
    #     result[date] = values.max() - values.min()
    # cum_range = pd.DataFrame(result).T.sort_index()

    ### 方法二
    idx_past = np.arange(len(ret_mat))[np.newaxis, :] - (np.arange(1, 13) * 21)[:, np.newaxis]
    arr = rolling_cum.values
    arr_past = np.where(idx_past[:, :, np.newaxis] >= 0, arr[idx_past], np.nan)
    arr_today = arr[np.newaxis, :, :]
    window_rets = arr_today - arr_past
    factor_values = np.nanmax(window_rets, axis=0) - np.nanmin(window_rets, axis=0)
    cum_range = pd.DataFrame(factor_values, index=ret_mat.index, columns=ret_mat.columns).loc[start_date:end_date]

    # ================================================================
    # 3. Short Term reversal — 21 日窗口，半衰期 5 日
    # ================================================================
    print('计算 Short Term reversal...')
    st_rev = ret_mat.rolling(window=21, min_periods=1).mean()
    st_rev = st_rev.rolling(window=21, min_periods=1).apply(lambda x: weight_sum(x, 5, normalize=False), raw=True).loc[start_date:end_date]

    # ================================================================
    # 4. Seasonality — 过去 5 年同月收益率均值
    # ================================================================
    print('计算 Seasonality...')
    season = {}
    for date in get_trade_cal(start_date=start_date, end_date=end_date):
        r_y = []
        for i in range(1, 6):
            his = ret_mat.index.get_indexer([date-pd.Timedelta(days=365*i)], method='ffill')[0]
            r_y.append(ret_mat.iloc[his: his+21].sum())
        season[date] = pd.concat(r_y, axis=1).mean(axis=1)
    season = pd.DataFrame(season).T

    # ================================================================
    # 5. Industry Momentum（暂不做行业中性化）
    # ================================================================
    print('计算 Industry Momentum...')
    #indmom_mat = ret_mat.rolling(window=126, min_periods=1).apply(lambda x: weight_sum(x, 21, normalize=False), raw=True).loc[start_date:end_date]
    indmom_mat = ret_mat.ewm(halflife=21, adjust=False).sum().loc[start_date:end_date]

    # ================================================================
    # 6. BETA + Hist sigma + Historical alpha（同一 CAPM 回归，252日 / 63日）
    # ================================================================
    print('计算 BETA / Hist sigma / Historical alpha...')
    r_mkt2 = pd.DataFrame((ret_mkt ** 2).values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index,columns=ret_mat.columns)
    r_mkt_r = ret_mat*ret_mkt.values[:, None]
    r_mkt_N = pd.DataFrame(ret_mkt.values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index, columns=ret_mat.columns)

    # ---------- 2. 计算各项指数加权和 ----------
    mean_wm2 = r_mkt2.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 63), raw=True)
    mean_wmr = r_mkt_r.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 63), raw=True)
    mean_wm = r_mkt_N.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 63), raw=True)
    mean_wr = ret_mat.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 63), raw=True)

    # ---------- 3. 计算 Beta & Alpha ----------
    var_m = mean_wm2 - mean_wm ** 2
    cov_mr = mean_wmr- mean_wm * mean_wr

    beta = cov_mr / var_m
    beta = beta.loc[start_date:end_date]
    alpha = mean_wr - beta * mean_wm
    alpha = alpha.loc[start_date:end_date]

    var_S = ret_mat.rolling(252, min_periods=1).var()
    var_M = r_mkt_N.rolling(252, min_periods=1).var()
    cov_SM = ret_mat.rolling(252, min_periods=1).cov(ret_mkt)

    # 2. 利用方差分解公式：Var(S - A - B*M) = Var(S - B*M)
    res_var = var_S + beta**2 * var_M - 2 * beta * cov_SM
    hist_sigma = np.sqrt(res_var.clip(lower=0)).loc[start_date:end_date]

    # ================================================================
    # 7. Relative strength — 252 日 / 126 日，11 日窗口均值
    # ================================================================
    print('计算 Relative strength...')
    rs = ret_mat.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 126, normalize=False), raw=True)
    rs = rs.rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    # ================================================================
    # 8. Long term relative strength — 1040 日 / 260 日，滞后 273 日，11 日均值，取负
    # ================================================================
    long_start = (pd.Timestamp(start_date) - pd.DateOffset(year=5)).strftime('%Y-%m-%d')
    print('加载长期价格数据...')
    price = get_price(codes=allstocks, start_date=long_start, end_date=end_date, fields=['close', 'pre_close'])

    print('加载长期指数数据...')
    hs300 = get_index_K(codes=['000300.SH'], start_date=long_start, end_date=end_date, fields=['close', 'pre_close'])

    # ── 构建矩阵 ──────────────────────────────────────────────────────────────
    close_pivot = price.pivot(index='trade_date', columns='ts_code', values='close').reindex(index=dates, columns=universe)
    pre_pivot = price.pivot(index='trade_date', columns='ts_code', values='pre_close').reindex(index=dates, columns=universe)
    ret_mat = np.log(close_pivot/pre_pivot)          # (T, N)

    hs300_close = hs300.set_index('trade_date')['close'].reindex(dates)
    hs300_pre   = hs300.set_index('trade_date')['pre_close'].reindex(dates)
    ret_mkt = np.log(hs300_close / hs300_pre)    

    print('计算 Long term relative strength...')
    long_term_rs = ret_mat.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260, normalize=False), raw=True)
    long_term_rs = - long_term_rs.rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    # ================================================================
    # 9. Long term historical alpha — 1040 日 / 260 日，滞后 273 日，11 日均值，取负
    # ================================================================
    print('计算 Long term historical alpha...')
    r_mkt2 = pd.DataFrame((ret_mkt ** 2).values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index,columns=ret_mat.columns)
    r_mkt_r = ret_mat*ret_mkt.values[:, None]
    r_mkt_N = pd.DataFrame(ret_mkt.values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index, columns=ret_mat.columns)

    mean_wm2 = r_mkt2.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260), raw=True)
    mean_wmr = r_mkt_r.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260), raw=True)
    mean_wm = r_mkt_N.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260), raw=True)
    mean_wr = ret_mat.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260), raw=True)

    var_m = mean_wm2 - mean_wm ** 2
    cov_mr = mean_wmr- mean_wm * mean_wr

    long_term_alpha = mean_wr - cov_mr / var_m * mean_wm
    long_term_alpha = - long_term_alpha.rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    # ================================================================
    # 组装输出
    # ================================================================
    print('组装结果...')
    factor_dfs = [daily_std, cum_range, st_rev, season, indmom_mat, beta, hist_sigma, alpha, rs, long_term_rs, long_term_alpha]   # 你的 N 个 df
    factor_names = ['daily_std','cum_range', 'st_rev', 'season', 'indmom_mat', 'beta', 'hist_sigma', 'alpha', 'rs', 'long_term_rs', 'long_term_alpha']     # 对应因子名称
    result = pd.concat(factor_dfs, axis=1, keys=factor_names).stack(level=1).reset_index(names=["trade_date", "ts_code"])
    print('done')
    return result


# ── 使用示例 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    allstocks = pd.read_csv('data/allstock.csv')
    allstocks = allstocks[~allstocks['ts_code'].str.contains('BJ')].ts_code.tolist()

    df = calc_price_factors(start_date='2024-01-01', end_date='2025-12-31', allstocks=allstocks)