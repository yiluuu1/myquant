import pandas as pd
import os


def get_price(codes=None, start_date='2023-03-01', end_date='2023-07-17', fq='post', fields=None, data_path='data/daily_K'):

    """
    codes: 股票代码列表；
    start_date, end_date: 开始结束时间
    fq: 复权pre/post/None
    fields:字段 ['open', 'high', 'low', 'close', 'vol', 'amount', 'adj_factor']
    """
    # 筛选字段
    fields1 = None
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]
        fields1 = fields.copy()
        if fq is not None:
            fields1 = fields + ['adj_factor']

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_feather(os.path.join(data_path, f'stock-{d.strftime("%Y%m%d")}.ftr'), columns=fields1)
            if isinstance(codes, list):
                tmp = tmp[tmp['ts_code'].isin(codes)]
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).reset_index(drop=True)

    # 复权操作
    if fq is not None:
        if fq == 'post':
            pass
        elif fq == 'pre':
            data1 = data.copy()
            latest_factor = data1.query('trade_date == trade_date.max()').set_index('ts_code')['adj_factor']
            data1['latest_factor'] = data1['ts_code'].map(latest_factor)
            data['adj_factor'] = data1.eval('adj_factor/latest_factor')
        
        for col in ['open', 'high', 'low', 'close']:
            try:
                data[col] = data.eval(f'{col}*adj_factor')
            except:
                continue
        try:
            data['vol'] = data.eval('vol/adj_factor')
        except:
            pass
    if fields is None:
        return data
    else:
        return data[fields]
    
def get_basic(codes=None, start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='data/daily_basic'):
    # 筛选字段
    fields1 = None
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]
        fields1 = fields.copy()

    # 提取数据
    data = []
    for d in pd.date_range(start=start_date, end=end_date):
        try:
            tmp = pd.read_feather(os.path.join(data_path, f'basic-{d.strftime("%Y%m%d")}.ftr'), columns=fields1)
            if isinstance(codes, list):
                tmp = tmp[tmp['ts_code'].isin(codes)]
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).reset_index(drop=True)
    
    if fields is None:
        return data
    else:
        return data[fields]

def get_index_K(codes=['000300.SH'], start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='data/index/index_daily_K'):
    # 筛选字段
    fields1 = None
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]
        fields1 = fields.copy()

    # 提取数据
    data = []
    for d in codes:
        try:
            tmp = pd.read_csv(os.path.join(data_path, f'{d}.csv'), usecols=fields1)
            tmp = tmp[tmp['trade_date'].between(start_date, end_date)]
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).reset_index(drop=True)
    
    if fields is None:
        return data
    else:
        return data[fields]
    
def get_index_basic(codes=['000300.SH'], start_date='2023-03-01', end_date='2023-07-17', fields=None, data_path='data/index/index_daily_basic'):
    # 筛选字段
    fields1 = None
    if fields is not None:
        fix_fields = ['ts_code', 'trade_date']
        fields = fix_fields + [f for f in fields if f not in fix_fields]
        fields1 = fields.copy()

    # 提取数据
    data = []
    for d in codes:
        try:
            tmp = pd.read_csv(os.path.join(data_path, f'{d}.csv'), columns=fields1)
            tmp = tmp[tmp['trade_date'].between(start_date, end_date)]
            data.append(tmp)
        except FileNotFoundError:
            continue
    data = pd.concat(data).reset_index(drop=True)
    
    if fields is None:
        return data
    else:
        return data[fields]