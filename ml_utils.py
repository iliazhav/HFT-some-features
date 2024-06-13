import pandas as pd
import numpy as np

from typing import List

def book_depth(lob, size, side, lvls_cnt: int = 10, fillna = True):
    side = side.lower()
    assert side in ('ask', 'bid'), f"unknown side: {side}"

    vol = 0
    fst_mask = pd.Series(np.zeros(lob.shape[0]))
    fst_mask.columns = [f'{side}s[0].amount']
    fst_mask.index = lob.index

    res = pd.Series(np.zeros(lob.shape[0]))
    res.index = lob.index

    for i in range(lvls_cnt):
        vol += (vol < size) * lob[f'{side}s[{i}].amount']
        fst_mask += (vol >= size)
        res[fst_mask == 1] = lob[f'{side}s[{i}].price']
        
    if fillna:
        res[res == 0] = lob[f'{side}s[{lvls_cnt-1}].price']
        
    return res

def spread(book_df):
    spread = pd.DataFrame()
    spread['spread'] = book_df['asks[0].price'] - book_df['bids[0].price']
    return spread

def weighted_midprice(book_df):
    weighted_midprice = pd.DataFrame()
    weighted_midprice['weighted_midprice'] = (book_df['asks[0].price'] * book_df['asks[0].amount'] + \
                         book_df['bids[0].price'] * book_df['bids[0].amount']) / \
                         (book_df['asks[0].amount'] + book_df['bids[0].amount'])
    return weighted_midprice

def vimba_up_level(book_df, up_level):
    vimba_up_level = pd.DataFrame()
    ask = sum([book_df[f'asks[{lvl}].amount'] for lvl in range(up_level)])
    vimba_up_level[f'vimba_up_{up_level}'] = ask / (sum([book_df[f'bids[{lvl}].amount'] for lvl in range(up_level)]) + ask)
    return vimba_up_level

def count_trades(trades_df, up_second):
    original_index = trades_df.index
    trades_per_seconds = pd.DataFrame()
    trades_df.index = pd.to_datetime(trades_df.index, errors='coerce')
    
    counts = trades_df.resample(f'{up_second}S').size()
    
    trades_per_seconds[f'counts_per_{up_second}_seconds'] = pd.DataFrame({f'counts_per_{up_second}_seconds': counts})
    
    trades_df.index = original_index

    trades_per_seconds.index = pd.to_datetime(trades_per_seconds.index).view('int64')

    return trades_per_seconds

def vimba_level(df, levels):
    vimba_at_level = pd.DataFrame()
    for lvl in levels:
        vimba_at_level[f'vimba_at_{lvl}'] = (df[f'asks[{lvl}].amount']) / (df[f'asks[{lvl}].amount'] + df[f'bids[{lvl}].amount'])
    return vimba_at_level

def zscore_calc(df, features, lookbacks: List[int]):
    df_zscore = pd.DataFrame()
    for feature in features:
        for lb in lookbacks:
            df_zscore[f'zscore_{feature}_{lb}'] = (df[feature] - df[feature].rolling(min_periods=lb // 2,
                                                                                     window=lb).mean()) / (
                                                          df[feature].rolling(min_periods=lb // 2,
                                                                              window=lb).std() + 1)
    return df_zscore

def vwap_all_by_size(df, up_size):
    vwap_side = pd.DataFrame()
    sum_vol = defaultdict(int)
    vwap = 0

    for side in ('ask', 'bid'):
        for lvl in range(num_of_levels):
            cur_vol_mult = np.maximum(np.minimum(up_size - sum_vol[side], df[f'{side}s[{lvl}].amount']), 0)
            sum_vol[side] += cur_vol_mult
            vwap += cur_vol_mult * df[f'{side}s[{lvl}].price']

    vwap_side[f'vwap_all_qty_{str(up_size).replace(".", "p")}'] = vwap / (sum_vol['ask'] + sum_vol['bid'])

    return vwap_side
