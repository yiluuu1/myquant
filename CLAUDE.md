# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

A股量化因子研究项目，基于 Tushare 数据源构建因子库（CNE6 风格）。核心流程：数据采集 → 因子计算 → 回测分析。

## 常用命令

```bash
# 运行数据采集（在 database.ipynb 中执行）
jupyter notebook database.ipynb

# 运行因子计算（在 factor_cal.ipynb 中执行）
jupyter notebook factor_cal.ipynb

# 探索性分析
jupyter notebook playground.ipynb
```

## 架构

### 数据层：`datatools.py`

统一数据访问接口，所有函数签名遵循相同模式：`codes, start_date, end_date, fields, data_path`。

- `get_price()` — 日线行情 + 复权（post/pre/None），自动读取 adj_factor
- `get_basic()` — 每日基础指标（市值、PE、PB、换手率等）
- `get_finance()` — 财报数据，按 ann_date 前向填充到交易日（使用 `data/trade_cal.csv` 过滤非交易日）
- `get_finance_ttm()` — TTM 滚动财务数据
- `get_report_roll()` — 分析师预测滚动统计（均值/标准差/覆盖数）
- `get_index_K()` / `get_index_basic()` — 指数行情/基础指标（CSV 格式）
- `get_moneyflow()` / `get_rzrq()` / `get_toplist()` — 资金流向、融资融券、龙虎榜

### 因子层：`factortools.py`

当前仅含 `MAD_winsorize()` — 基于 MAD 的去极值函数。

### 数据采集：`database.ipynb`

从 Tushare API 批量拉取数据，按日期存储为 feather 文件。采集范围通过 `data_cal`（交易日历子集）控制。

**注意**：使用自定义 API endpoint（非 Tushare 官方），token 直接写在 notebook 中。

### 数据存储：`data/`

| 目录 | 格式 | 内容 |
|------|------|------|
| `daily_K/` | feather | 日线行情，文件名 `stock-{YYYYMMDD}.ftr` |
| `daily_basic/` | feather | 每日基础指标，文件名 `basic-{YYYYMMDD}.ftr` |
| `finance/sheet/` | feather | 合并财报（资产负债表+现金流量表+利润表） |
| `finance/sheet_ttm/` | feather | TTM 滚动财报 |
| `moneyflow/` | feather | 资金流向 |
| `rzrq/` | feather | 融资融券明细 |
| `toplist/` | feather | 龙虎榜 |
| `report_rc/` | feather | 分析师预测原始数据 |
| `report_rc/roll_data/` | feather | 分析师预测滚动统计 |
| `index/` | CSV | 指数行情/基础指标 |
| `ETF/` | feather | ETF 日线 |
| `future/` | feather | 期货日线 |
| `cyq/` | feather | 筹码分布 |
| `limit_list/` | feather | 涨跌停板 |
| `holdertrade/` | feather | 股东增减持 |
| `month_recommend/` | CSV | 券商月度金股 |

## 关键约定

- **股票代码格式**：`{6位数字}.{SH/SZ/BJ}`（如 `600000.SH`），北交所（BJ）在因子计算中通常排除
- **日期格式**：函数参数用 `'YYYY-MM-DD'` 字符串，文件名用 `'YYYYMMDD'`
- **feather 文件读取**：用 `columns=` 参数按需加载列，避免内存浪费
- **财报前向填充**：`get_finance()` 和 `get_finance_ttm()` 按 ann_date 填充到日频，仅保留交易日
- **TTM 计算逻辑**：`Q4数据 + 当期 - 去年同期`，Q4 数据向后填充3个季度
- **allstock.csv 过滤**：因子计算时过滤上市不满一年的股票和北交所股票
