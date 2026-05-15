"""
CNE-6 价格类因子计算
因子清单：Daily std, Cumulative range, Short Term reversal, Seasonality,
         Industry Momentum, Relative strength, Long term relative strength,
         BETA, Hist sigma, Historical alpha, Long term historical alpha
"""

import numpy as np
import pandas as pd
from numba import njit
from datatools import get_price, get_basic, get_index_K, get_trade_cal


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def make_weights(window, half_life):
    """半衰指数权重，归一化到和=1"""
    w = 0.5**(np.arange(window) / half_life)
    return w / w.sum()


@njit(cache=True)
def _ewrs_core(arr_clean, nan_f, q, qk, k, T, N):
    """numba 核心：IIR 递推 + NaN 计数"""
    V = np.empty((T, N))
    V[0] = arr_clean[0]
    C = np.empty((T, N))
    C[0] = nan_f[0]
    for t in range(1, T):
        v = arr_clean[t] + q * V[t - 1]
        c = nan_f[t] + C[t - 1]
        if t >= k:
            v -= qk * arr_clean[t - k]
            c -= nan_f[t - k]
        V[t] = v
        C[t] = c
    return V, C


def exp_wt_rolling_sum(arr, w):
    """滚动加权求和（numba IIR 递推版，O(T×N)）"""
    T, N = arr.shape
    k = len(w)
    q = w[1] / w[0]
    qk = q ** k
    nan_mask = np.isnan(arr)
    arr_clean = np.where(nan_mask, 0.0, arr).astype(np.float32)
    nan_f = nan_mask.astype(np.float32)
    V, C = _ewrs_core(arr_clean, nan_f, q, qk, k, T, N)
    out = w[0] * V
    out[C > 0] = np.nan
    out[:k - 1] = np.nan
    return out


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
    load_start = (pd.Timestamp(start_date) - pd.DateOffset(years=7)).strftime('%Y-%m-%d')

    print('加载价格数据...')
    price = get_price(start_date=load_start, end_date=end_date, fields=['close', 'pre_close'])
    price = price[price['ts_code'].isin(allstocks)].copy()

    print('加载指数数据...')
    hs300 = get_index_K(codes=['000300.SH'], start_date=load_start, end_date=end_date, fields=['close', 'pre_close'])
    hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])

    print('加载市值数据...')
    basic = get_basic(start_date=load_start, end_date=end_date, fields=['circ_mv'])
    basic = basic[basic['ts_code'].isin(allstocks)].copy()
    basic['trade_date'] = pd.to_datetime(basic['trade_date'])
    
    all_dates = get_trade_cal(start_date=start_date, end_date=end_date)

    # ── 构建矩阵 ──────────────────────────────────────────────────────────────
    dates = np.sort(price['trade_date'].unique())
    universe = np.sort(price['ts_code'].unique())

    close_pivot = price.pivot(index='trade_date', columns='ts_code', values='close').reindex(index=dates, columns=universe)
    pre_pivot = price.pivot(index='trade_date', columns='ts_code', values='pre_close').reindex(index=dates, columns=universe)

    ret_mat = np.log(close_pivot/pre_pivot)          # (T, N)

    hs300_close = hs300.set_index('trade_date')['close'].reindex(dates)
    hs300_pre   = hs300.set_index('trade_date')['pre_close'].reindex(dates)
    ret_mkt = np.log(hs300_close.values / hs300_pre.values)          # (T,)

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
    st_rev = ret_mat.rolling(window=21, min_periods=1).mean().ewm(halflife=5).sum().loc[start_date:end_date]

    # ================================================================
    # 4. Seasonality — 过去 5 年同月收益率均值
    # ================================================================
    print('计算 Seasonality...')
    season = {}
    for date in get_trade_cal(start_date=start_date, end_date=end_date):
        r_y = []
        for i in range(1, 6):
            his = ret_mat.index.get_indexer([date-pd.Timedelta(days=365*i)], method='pad')[0]
            r_y.append(ret_mat.iloc[his: his+21].sum())
        season[date] = pd.concat(r_y, axis=1).mean(axis=1)
    season = pd.DataFrame(season).T

    # ================================================================
    # 5. Industry Momentum（暂不做行业中性化）
    # ================================================================
    print('计算 Industry Momentum...')
    indmom_mat = ret_mat.ewm(halflife=21).sum().loc[start_date:end_date]

    # ================================================================
    # 6. BETA + Hist sigma + Historical alpha（同一 CAPM 回归，252日 / 63日）
    # ================================================================
    print('计算 BETA / Hist sigma / Historical alpha...')
    r_mkt2 = pd.DataFrame((ret_mkt ** 2).values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index,columns=ret_mat.columns)
    r_mkt_r = ret_mat*ret_mkt.values[:, None]
    r_mkt_N = pd.DataFrame(ret_mkt.values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index, columns=ret_mat.columns)

    # ---------- 2. 计算各项指数加权和 ----------
    mean_wm2 = r_mkt2.ewm(halflife=63).mean()
    mean_wmr = r_mkt_r.ewm(halflife=63).mean()
    mean_wm  = r_mkt_N.ewm(halflife=63).mean()
    mean_wr  = ret_mat.ewm(halflife=63).mean()

    # ---------- 3. 计算 Beta & Alpha ----------
    var_m = mean_wm2 - mean_wm ** 2
    cov_mr = mean_wmr- mean_wm * mean_wr

    beta = cov_mr / var_m.where(var_m > 1e-18, np.nan).loc[start_date:end_date]
    alpha = mean_wr - beta * mean_wm.loc[start_date:end_date]

    var_S = ret_mat.rolling(252, min_periods=1).var()
    var_M = r_mkt_N.rolling(252, min_periods=1).var()
    cov_SM = ret_mat.rolling(252, min_periods=1).cov(ret_mkt)

    # 2. 利用方差分解公式：Var(S - A - B*M) = Var(S - B*M)
    res_var = var_S + beta**2 * var_M - 2 * beta * cov_SM
    hist_sigma = np.sqrt(res_var.clip(lower=0)).loc[start_date:end_date]

    # ================================================================
    # 7. Relative strength — 252 日 / 126 日，滞后 11 日，11 日窗口均值
    # ================================================================
    print('计算 Relative strength...')
    rs = ret_mat.ewm(halflife=126).sum().rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    # ================================================================
    # 8. Long term relative strength — 1040 日 / 260 日，滞后 273 日，11 日均值，取负
    # ================================================================
    print('计算 Long term relative strength...')
    long_term_rs = - ret_mat.ewm(halflife=260).sum().rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    # ================================================================
    # 9. Long term historical alpha — 1040 日 / 260 日，滞后 273 日，11 日均值，取负
    # ================================================================
    print('计算 Long term historical alpha...')
    mean_wm2 = r_mkt2.ewm(halflife=260).mean()
    mean_wmr = r_mkt_r.ewm(halflife=260).mean()
    mean_wm  = r_mkt_N.ewm(halflife=260).mean()
    mean_wr  = ret_mat.ewm(halflife=260).mean()

    var_m = mean_wm2 - mean_wm ** 2
    cov_mr = mean_wmr- mean_wm * mean_wr

    long_term_alpha = mean_wr - cov_mr / var_m.where(var_m > 1e-18, np.nan) * mean_wm
    long_term_alpha = - long_term_alpha.rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    # ================================================================
    # 组装输出
    # ================================================================
    print('组装结果...')
    rows = np.repeat(dates, n_stocks)
    cols = np.tile(universe, n_dates)

    result = pd.DataFrame({
        'ts_code':              cols,
        'cal_date':             rows,
        'daily_std':            daily_std.ravel(),
        'cum_range':            cum_range.ravel(),
        'st_rev':               st_rev.ravel(),
        'seasonality':          season.ravel(),
        'industry_momentum':    indmom_mat.ravel(),
        'beta':                 beta.ravel(),
        'hist_sigma':           hist_sigma.ravel(),
        'historical_alpha':     alpha.ravel(),
        'relative_strength':    rs.ravel(),
        'long_term_rs':         long_term_rs.ravel(),
        'long_term_alpha':      long_term_alpha.ravel(),
    })

    # 截取目标日期范围
    result = result.sort_values(['ts_code', 'cal_date']).reset_index(drop=True)
    print('done')
    return result


# ── 使用示例 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    allstocks = pd.read_csv('data/allstock.csv')
    allstocks = allstocks[(allstocks['list_date'] < 20250101) & ~allstocks['ts_code'].str.contains('BJ')].ts_code.tolist()

    df = calc_price_factors(start_date='2025-01-01', end_date='2025-12-31', allstocks=allstocks)