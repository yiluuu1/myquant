import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def get_price(codes=None, start_date='2023-03-01', end_date='2023-07-17', fq='post', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/daily_K'):

    """
    codes: 股票代码列表；
    start_date, end_date: 开始结束时间
    fq: 复权pre/post/None
    """
    # 筛选字段
    fields1 = None
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]
        fields1 = fields.copy()
        if fq is not None and 'adj_factor' not in fields:
            fields1 = fields + ['adj_factor']

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'stock-{d.strftime("%Y%m%d")}.parquet'), columns=fields1)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    # 复权操作
    if fq is not None:
        if fq == 'post':
            pass
        elif fq == 'pre':
            data1 = data.copy()
            latest_factor = data1.query('trade_date == trade_date.max()').set_index('ts_code')['adj_factor']
            data1['latest_factor'] = data1['ts_code'].map(latest_factor)
            data['adj_factor'] = data1.eval('adj_factor/latest_factor')
        
        for col in ['open', 'high', 'low', 'close','pre_close']:
            try:
                data[col] = data.eval(f'{col}*adj_factor')
            except:
                continue
        try:
            data['vol'] = data.eval('vol/adj_factor')
        except:
            pass
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)
    
def get_basic(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/daily_basic'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'basic-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def get_index_K(codes=['000300.SH'], start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/index/index_daily_K'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in codes:
        try:
            tmp = pd.read_csv(os.path.join(data_path, f'{d}.csv'), usecols=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    data = data[data['trade_date'].between(start_date, end_date)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)
    
def get_index_basic(codes=['000300.SH'], start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/index/index_daily_basic'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in codes:
        try:
            tmp = pd.read_csv(os.path.join(data_path, f'{d}_basic.csv'), usecols=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    data = data[data['trade_date'].between(start_date, end_date)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def get_moneyflow(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/moneyflow'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'moneyflow-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def get_rzrq(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/rzrq'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'margin-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def get_toplist(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/toplist'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'toplist-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def get_report_rc(codes=None, start_date=None, end_date=None, year = '2025', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/report_rc/raw_report'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code','report_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    try:
        data = pd.read_parquet(os.path.join(data_path, f'report-{year}Q4.parquet'), columns=fields)
        if codes is not None:
            data = data[data['ts_code'].isin(codes)]
        if start_date is not None and end_date is not None:
            data = data[data['report_date'].between(start_date, end_date)]
    except FileNotFoundError:
        pass
    data = data.sort_values(['ts_code', 'report_date'])
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def get_finance(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/finance/sheet', align_trade_date=True):
    def get_report_date(date_input):
        dt = pd.to_datetime(date_input)
        year = dt.year
        month = dt.month
        if month in [11, 12]:
            return f"{year}-09-30"
        elif month in [1, 2, 3, 4]:
            return f"{year-1}-09-30"
        elif month in [5, 6, 7, 8]:
            return f"{year}-03-31"
        elif month in [9, 10]:
            return f"{year}-06-30"
        else:
            return None  # 异常情况
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'end_date', 'ann_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=get_report_date(start_date), end=get_report_date(end_date), freq='QE'):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'sheet-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['ts_code', 'ann_date'])
    data = data.drop_duplicates(subset=['ts_code', 'ann_date'], keep='last')
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    data['ann_date'] = pd.to_datetime(data['ann_date'])
    if align_trade_date:
        data = data.set_index(['ts_code', 'ann_date']).reindex(
            pd.MultiIndex.from_product([data['ts_code'].unique(), pd.date_range(data['ann_date'].min(), end_date, freq='D')],
            names=['ts_code', 'ann_date'])).groupby(level='ts_code').ffill().reset_index()
        data = data[data['ann_date'].between(start_date, end_date)]
        trade_cal = pd.to_datetime(pd.read_csv('C:/Users/User/OneDrive - CUHK-Shenzhen/data/trade_cal.csv')['cal_date'].unique().tolist())
        data = data[data['ann_date'].isin(trade_cal)]
    data = data.rename(columns={'ann_date':'trade_date','end_date':'stat_date'})
    if fields is not None:
        data = data[fields]
    return data.reset_index(drop=True)
    
def get_finance_ttm(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/finance/sheet_ttm',align_trade_date=True):
    def get_report_date(date_input):
        dt = pd.to_datetime(date_input)
        year = dt.year
        month = dt.month
        if month in [11, 12]:
            return f"{year}-09-30"
        elif month in [1, 2, 3, 4]:
            return f"{year-1}-09-30"
        elif month in [5, 6, 7, 8]:
            return f"{year}-03-31"
        elif month in [9, 10]:
            return f"{year}-06-30"
        else:
            return None
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'end_date', 'ann_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=get_report_date(start_date), end=get_report_date(end_date), freq='QE'):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'sheet_ttm-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['ts_code', 'ann_date'])
    data = data.drop_duplicates(subset=['ts_code', 'ann_date'], keep='last')
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    data['ann_date'] = pd.to_datetime(data['ann_date'])
    if align_trade_date:
        data = data.set_index(['ts_code', 'ann_date']).reindex(
            pd.MultiIndex.from_product([data['ts_code'].unique(), pd.date_range(data['ann_date'].min(), end_date, freq='D')],
            names=['ts_code', 'ann_date'])).groupby(level='ts_code').ffill().reset_index()
        data = data[data['ann_date'].between(start_date, end_date)]
        trade_cal = pd.to_datetime(pd.read_csv('C:/Users/User/OneDrive - CUHK-Shenzhen/data/trade_cal.csv')['cal_date'].unique().tolist())
        data = data[data['ann_date'].isin(trade_cal)]
    data = data.rename(columns={'ann_date':'trade_date','end_date':'stat_date'})
    if fields is not None:
        data = data[fields]
    return data.reset_index(drop=True)
    
def get_report_roll(codes=None, start_date='2023-03-01', end_date='2023-07-17', year=[2025],  fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/report_rc/roll_data', align_trade_date=True):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'report_date','quarter']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for y in year:
        tmp = pd.read_parquet(os.path.join(data_path, f'report_roll-{y}.parquet'), columns=fields)
        if len(tmp)==0:
            continue
        if align_trade_date:
            tmp = tmp.set_index(['ts_code', 'report_date']).reindex(
                pd.MultiIndex.from_product([tmp['ts_code'].unique(), pd.date_range(tmp['report_date'].min(), end_date, freq='D')],
                names=['ts_code', 'report_date'])).groupby(level='ts_code').ffill().reset_index()
            tmp = tmp[tmp['report_date'].between(start_date, end_date)]
            trade_cal = pd.to_datetime(pd.read_csv('C:/Users/User/OneDrive - CUHK-Shenzhen/data/trade_cal.csv')['cal_date'].unique().tolist())
            tmp = tmp[tmp['report_date'].isin(trade_cal)]
        data.append(tmp)
    data = pd.concat(data).sort_values(['ts_code', 'report_date'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    data['report_date'] = pd.to_datetime(data['report_date'])
    data = data.rename(columns={'report_date':'trade_date'})
    if fields is not None:
        data = data[fields]
    return data.reset_index(drop=True)
    
def get_trade_cal(start_date='2023-03-01', end_date='2023-07-17', data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/trade_cal.csv'):
    data = pd.read_csv(data_path).sort_values('cal_date')
    data['cal_date'] = pd.to_datetime(data['cal_date'])
    return data[(data['cal_date'] >= start_date) & (data['cal_date'] <= end_date) & (data['is_open'] == 1)]['cal_date'].tolist()

def get_cyq(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/cyq'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'cyq-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def get_ETF(codes=None, start_date='2023-03-01', end_date='2023-07-17', fq='post', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/ETF'):

    # 筛选字段
    fields1 = None
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]
        fields1 = fields.copy()
        if fq is not None and 'adj_factor' not in fields:
            fields1 = fields + ['adj_factor']

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'etf-{d.strftime("%Y%m%d")}.parquet'), columns=fields1)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    # 复权操作
    if fq is not None:
        if fq == 'post':
            pass
        elif fq == 'pre':
            data1 = data.copy()
            latest_factor = data1.query('trade_date == trade_date.max()').set_index('ts_code')['adj_factor']
            data1['latest_factor'] = data1['ts_code'].map(latest_factor)
            data['adj_factor'] = data1.eval('adj_factor/latest_factor')
        
        for col in ['open', 'high', 'low', 'close','pre_close']:
            try:
                data[col] = data.eval(f'{col}*adj_factor')
            except:
                continue
        try:
            data['vol'] = data.eval('vol/adj_factor')
        except:
            pass
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)
    
def get_future(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/future'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'future-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)
    
def get_limit_list(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/limit_list'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'limit_list-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def get_technical(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/tech'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'Tech-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)
    
def get_industry_K(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='C:/Users/User/OneDrive - CUHK-Shenzhen/data/industry/sw'):
    # 筛选字段
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_parquet(os.path.join(data_path, f'Tech-{d.strftime("%Y%m%d")}.parquet'), columns=fields)
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).sort_values(['trade_date','ts_code'])
    if codes is not None:
        data = data[data['ts_code'].isin(codes)]
    if fields is not None:
        data =  data[fields]
    return data.reset_index(drop=True)

def spilit_test(df_sample, factor_columns, change_freq, ret_col, n=10, cost = 2 * 0.0015):
    df_sample = df_sample.sort_values('trade_date').iloc[::change_freq]
    df_sample['group'] = df_sample.groupby('trade_date')[factor_columns].transform(
        lambda x: pd.qcut(x.rank(method='first'), n, labels=False, duplicates='drop'))
    # 计算每组每日的收益 Group 0 是预测最差的组，Group n_groups-1 是预测最好的组
    group_ret = df_sample.groupby(['trade_date', 'group'])[ret_col].mean().unstack()
    # 3. 处理换仓成本
    group_ret = group_ret - cost
    # 4. 计算多空对冲收益 (做多最好组，做空最差组)
    group_ret['Long_Short'] = group_ret[n - 1] - group_ret[0]
    # 5. 计算累计净值
    group_nav = (1 + group_ret).cumprod()
    
    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    # 子图1：各组净值曲线 (扇形图)
    n_groups = len(group_nav.columns) - 1 # 减去 Long_Short 列
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_groups))
    for i in range(0, n_groups, 1):
        axes[0].plot(group_nav.index, group_nav[i], label=f'Group {i} (第{i+1}层)', color=colors[i], linewidth=1.5)
    axes[0].set_title('分层回测净值曲线 (扇形图)', fontsize=14)
    axes[0].set_ylabel('累计净值')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    # 子图2：多空对冲净值曲线
    axes[1].plot(group_nav.index, group_nav['Long_Short'], label='多空对冲', color='green', linewidth=2)
    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_title('多空对冲净值曲线', fontsize=14)
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('累计净值')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()