import numpy as np
import pandas as pd
from datatools import get_price, get_basic, get_index_K


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def make_weights(window, half_life):
    """半衰指数权重，归一化到和=1"""
    tau = window / half_life * np.log(2)
    w = np.exp(-tau * (1 - np.arange(window) / window))
    return w / w.sum()


def exp_wt_rolling_sum(arr, w):
    """
    滚动加权求和（向量化版）
    arr: (T, N)  w: (k,)
    窗口第一行为 NaN 则该行输出 NaN
    """
    T, N = arr.shape
    k = len(w)
    out = np.full((T, N), np.nan)
    for i in range(k - 1, T):
        if np.isnan(arr[i - k + 1, 0]):
            continue
        out[i] = arr[i - k + 1:i + 1].T @ w
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
    hs300 = get_index_K(codes=['000300.SH'], start_date=load_start, end_date=end_date,fields=['close', 'pre_close'])
    hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])

    # ── 构建矩阵 ──────────────────────────────────────────────────────────────
    dates = np.sort(price['trade_date'].unique())
    universe = np.sort(price['ts_code'].unique())

    close_pivot = price.pivot(index='trade_date', columns='ts_code', values='close').reindex(index=dates, columns=universe)
    pre_pivot = price.pivot(index='trade_date', columns='ts_code', values='pre_close').reindex(index=dates, columns=universe)

    ret_mat = np.log(close_pivot.values / pre_pivot.values)          # (T, N)

    hs300_close = hs300.set_index('trade_date')['close'].reindex(dates)
    hs300_pre   = hs300.set_index('trade_date')['pre_close'].reindex(dates)
    ret_mkt = np.log(hs300_close.values / hs300_pre.values)          # (T,)

    n_dates, n_stocks = ret_mat.shape
    print(f'矩阵: {n_dates} 交易日 × {n_stocks} 只股票')

    # ================================================================
    # 1. Daily std — 252 日窗口，半衰期 42 日
    # ================================================================
    print('计算 Daily std...')
    w_dstd = make_weights(252, 42)
    w2 = w_dstd ** 2
    rw     = exp_wt_rolling_sum(np.ones_like(ret_mat), w2)
    rw2    = exp_wt_rolling_sum(ret_mat ** 2, w2)
    rw_sum = exp_wt_rolling_sum(ret_mat, w2)
    var_w  = rw2 / rw - (rw_sum / rw) ** 2
    daily_std = np.sqrt(np.clip(var_w, 0, None))

    # ================================================================
    # 2. Cumulative range — Z(T) 为过去 T 个月累积对数收益，T=1..12
    # ================================================================
    print('计算 Cumulative range...')
    monthly_ret = []
    monthly_dates = []
    for d in dates:
        m_start = d - pd.DateOffset(months=1)
        mask = (dates >= m_start) & (dates < d)
        if mask.sum() >= 15:
            monthly_ret.append(np.nansum(ret_mat[mask], axis=0))
            monthly_dates.append(d)
    monthly_ret = np.array(monthly_ret)  # (M, N)

    cum_range = np.full((n_dates, n_stocks), np.nan)
    for i, d in enumerate(dates):
        m_idx = [j for j, md in enumerate(monthly_dates) if md <= d]
        if len(m_idx) >= 12:
            window = monthly_ret[m_idx[-12:]]
            cum_z = np.cumsum(window, axis=0)
            cum_range[i] = np.nanmax(cum_z, axis=0) - np.nanmin(cum_z, axis=0)

    # ================================================================
    # 3. Short Term reversal — 21 日窗口，半衰期 5 日
    # ================================================================
    print('计算 Short Term reversal...')
    w_str = make_weights(21, 5)
    st_rev = exp_wt_rolling_sum(ret_mat, w_str)

    # ================================================================
    # 4. Seasonality — 过去 5 年同月收益率均值
    # ================================================================
    print('计算 Seasonality...')
    mret_df = pd.DataFrame(monthly_ret, index=pd.DatetimeIndex(monthly_dates), columns=universe)
    season = np.full((n_dates, n_stocks), np.nan)
    for i, d in enumerate(dates):
        y, m = d.year, d.month
        vals = []
        for y_back in range(1, 6):
            key = pd.Timestamp(year=y - y_back, month=m, day=1) + pd.DateOffset(months=1)
            if key in mret_df.index:
                vals.append(mret_df.loc[key].values)
        if vals:
            season[i] = np.nanmean(vals, axis=0)

    # ================================================================
    # 5. Industry Momentum
    # ================================================================
    print('计算 Industry Momentum...')
    ind = pd.read_csv('data/industry.csv')
    stock_ind = ind.set_index('ts_code')['l1_code']

    mktcap = basic.pivot(index='trade_date', columns='ts_code', values='circ_mv') \
                   .reindex(index=dates, columns=universe)
    mktcap_sqrt = np.sqrt(mktcap.values)

    w_ind = make_weights(126, 21)   # 6 个月 ≈ 126 日，半衰期 1 个月 ≈ 21 日
    rss_mat = exp_wt_rolling_sum(ret_mat, w_ind)  # 个股相对强度

    rsi_mat = np.full_like(rss_mat, np.nan)
    for i in range(n_dates):
        rss_today = pd.Series(rss_mat[i], index=universe)
        mc_today  = pd.Series(mktcap_sqrt[i], index=universe)
        df_tmp = pd.DataFrame({
            'rss': rss_today, 'mc': mc_today,
            'ind': stock_ind.reindex(universe)
        })
        rsi = df_tmp.groupby('ind').apply(
            lambda g: np.nansum(g['rss'] * g['mc']) / np.nansum(g['mc'])
        )
        rsi_mat[i] = df_tmp['ind'].map(rsi).values

    indmom_mat = -(rss_mat - rsi_mat)

    # ================================================================
    # 6. BETA + Hist sigma + Historical alpha（同一 CAPM 回归，252日 / 63日）
    # ================================================================
    print('计算 BETA / Hist sigma / Historical alpha...')
    w_beta = make_weights(252, 63)
    wm = (ret_mkt[:, None] * np.ones((1, n_stocks))) * w_beta[:, None]  # (252, N)

    sum_w    = exp_wt_rolling_sum(np.ones_like(ret_mat), w_beta ** 2)
    sum_wm   = exp_wt_rolling_sum(wm, w_beta)
    sum_wm2  = exp_wt_rolling_sum(wm ** 2, w_beta ** 2)
    sum_wr   = exp_wt_rolling_sum(ret_mat * w_beta[:, None], w_beta)
    sum_wmr  = exp_wt_rolling_sum(wm * ret_mat, w_beta ** 2)

    var_m  = sum_wm2 / sum_w - (sum_wm / sum_w) ** 2
    cov_mr = sum_wmr / sum_w - (sum_wm / sum_w) * (sum_wr / sum_w)
    beta = cov_mr / var_m
    beta = np.where(np.isnan(var_m) | (var_m < 1e-18), np.nan, beta)

    mu_m = sum_wm / sum_w
    mu_r = sum_wr / sum_w
    alpha = mu_r - beta * mu_m                       # Historical alpha

    resid = ret_mat - (alpha + beta * ret_mkt[:, None])
    sum_w_resid2 = exp_wt_rolling_sum(
        (resid * w_beta[:, None]) ** 2, np.ones(252)
    )
    hist_sigma = np.sqrt(sum_w_resid2 / sum_w)       # Hist sigma

    # ================================================================
    # 7. Relative strength — 252 日 / 126 日，滞后 11 日，11 日窗口均值
    # ================================================================
    print('计算 Relative strength...')
    w_rs = make_weights(252, 126)
    rs_raw = exp_wt_rolling_sum(ret_mat, w_rs)
    rs_shift = np.full_like(rs_raw, np.nan)
    rs_shift[11:] = rs_raw[:-11]
    rs = pd.DataFrame(rs_shift).rolling(11, min_periods=11).mean().values

    # ================================================================
    # 8. Long term relative strength — 1040 日 / 260 日，滞后 273 日，11 日均值，取负
    # ================================================================
    print('计算 Long term relative strength...')
    w_ltrs = make_weights(1040, 260)
    ltrs_raw = exp_wt_rolling_sum(ret_mat, w_ltrs)
    ltrs_shift = np.full_like(ltrs_raw, np.nan)
    ltrs_shift[273:] = ltrs_raw[:-273]
    long_term_rs = -pd.DataFrame(ltrs_shift).rolling(11, min_periods=11).mean().values

    # ================================================================
    # 9. Long term historical alpha — 1040 日 / 260 日，滞后 273 日，11 日均值，取负
    # ================================================================
    print('计算 Long term historical alpha...')
    w_ltha = make_weights(1040, 260)
    wm_lt = (ret_mkt[:, None] * np.ones((1, n_stocks))) * w_ltha[:, None]

    sum_w_lt   = exp_wt_rolling_sum(np.ones_like(ret_mat), w_ltha ** 2)
    sum_wm_lt  = exp_wt_rolling_sum(wm_lt, w_ltha)
    sum_wm2_lt = exp_wt_rolling_sum(wm_lt ** 2, w_ltha ** 2)
    sum_wr_lt  = exp_wt_rolling_sum(ret_mat * w_ltha[:, None], w_ltha)
    sum_wmr_lt = exp_wt_rolling_sum(wm_lt * ret_mat, w_ltha ** 2)

    var_m_lt  = sum_wm2_lt / sum_w_lt - (sum_wm_lt / sum_w_lt) ** 2
    cov_mr_lt = sum_wmr_lt / sum_w_lt - (sum_wm_lt / sum_w_lt) * (sum_wr_lt / sum_w_lt)
    beta_lt = cov_mr_lt / var_m_lt
    beta_lt = np.where(np.isnan(var_m_lt) | (var_m_lt < 1e-18), np.nan, beta_lt)

    mu_m_lt = sum_wm_lt / sum_w_lt
    mu_r_lt = sum_wr_lt / sum_w_lt
    ltha_raw = mu_r_lt - beta_lt * mu_m_lt
    ltha_shift = np.full_like(ltha_raw, np.nan)
    ltha_shift[273:] = ltha_raw[:-273]
    long_term_alpha = -pd.DataFrame(ltha_shift).rolling(11, min_periods=11).mean().values

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
        'beta':                 beta.ravel(),
        'hist_sigma':           hist_sigma.ravel(),
        'historical_alpha':     alpha.ravel(),
        'relative_strength':    rs.ravel(),
        'long_term_rs':         long_term_rs.ravel(),
        'long_term_alpha':      long_term_alpha.ravel(),
    })

    # Industry Momentum 单独合并（仅覆盖有市值数据的区间）
    im_df = pd.DataFrame({
        'ts_code':          np.tile(universe, n_dates),
        'cal_date':         np.repeat(dates, n_stocks),
        'industry_momentum': indmom_mat.ravel(),
    })
    result = result.merge(im_df, on=['ts_code', 'cal_date'], how='left')

    # 截取目标日期范围
    result = result[result['cal_date'].between(start_date, end_date)].copy()
    result = result.sort_values(['ts_code', 'cal_date']).reset_index(drop=True)
    print('完成')
    return result


# ── 使用示例 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    allstocks = pd.read_csv('data/allstock.csv')
    allstocks = allstocks[(allstocks['list_date'] < 20250101) & ~allstocks['ts_code'].str.contains('BJ')].ts_code.tolist()

    df = calc_price_factors(start_date='2025-01-01', end_date='2025-12-31', allstocks=allstocks)
