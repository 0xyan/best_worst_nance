import pandas as pd
import matplotlib.pyplot as plt
import requests
import schedule
import time
from pandas.plotting import table
import os
import matplotlib.dates as mdates

def main_function(timeframe, periods, min_vol = 30000000, startTime = None):
    
    def send(text):
        url = 'https://api.telegram.org/bot'+token+'/sendMessage?chat_id='+id_tg+'&text='+text+''
        resp = requests.get(url)
        r = resp.json()
        return
    
    def sendimage(img):
        url = 'https://api.telegram.org/bot'+token+'/sendPhoto'
        f = {'photo': open(img, 'rb')}
        d = {'chat_id': id_tg}
        resp = requests.get(url, files = f, data = d)
        r = resp.json()
        return r
    
    token = ''
    id_tg = ''


    #querying all the USDT pairs from Binance
    def pairs(filter='USDT'):
        url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
        r = requests.get(url)
        if r.status_code == 200:
            r = r.json()
            pairs_data = r['symbols']
            full_data_dic = {s['symbol']: s for s in pairs_data if filter in s['symbol']}
            return full_data_dic.keys()
        else:
            return r.status_code

    tickers = list(pairs('USDT'))

    #removing expirable futures
    tickers = [x for x in tickers if '_' not in x]
    
    #function to create dataframes with historic data
    def candles(ticker,tf,startTime,limit):
        url = 'https://fapi.binance.com/fapi/v1/klines'
        param = {'symbol': ticker, 'interval': tf, 'limit': limit}
        if startTime is not None:
            param['startTime'] = startTime
        r = requests.get(url, params = param)
        if r.status_code == 200:
            dfi = pd.DataFrame(r.json())
            #fetching only data that we need
            df = pd.DataFrame()
            df['time'] = dfi[0].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit = 'ms')
            df['close'] = dfi[4].astype(float)
            df['volume'] = dfi[7].astype(float)
            ret = 'return_' + ticker
            df[ret] = df['close'].pct_change(1)
            df[f'cum_{ret}'] = (df[ret] + 1).cumprod() - 1
            df.rename(columns = {'close':'close_' + ticker}, inplace = True)
            df.rename(columns = {'volume':'volume_' + ticker}, inplace = True)
            #set_index
            datetime_series = df['time']
            datetime_index = pd.DatetimeIndex(datetime_series.values)
            df = df.set_index(datetime_index)
            df.drop(["time"], axis = 1, inplace = True)
            df.dropna(inplace = True)
            return df
        else:
            return(print('doublecheck'))
    
    #dataframes to filter by last day's volume
    timeframe_daily = '1d'
    lmt = 3
    for i in tickers:
        try:
            globals()[f'{i}'] = candles(i,timeframe_daily, startTime,lmt)
        except:
            pass
        
    #building a vol spreadsheet
    df = pd.DataFrame(index = BTCUSDT.index)
    df['BTCUSDT'] = BTCUSDT['volume_BTCUSDT']
    for i in tickers:
        try:
            df[f'{i}'] = globals()[f'{i}'][f'volume_{i}']
        except:
            pass

    # VOLUME filtering
    volume_list = df.columns[(df.iloc[[0]] > min_vol).all()]
    tickers = volume_list.tolist()

    #20% of total
    percentage = 0.20
    top_percent = int(round(len(tickers) * percentage, 0))
    
    #dataframes with volume above x
    tf = timeframe
    lmt = periods
    for i in tickers:
        globals()[f'{i}'] = candles(i,tf,startTime,lmt)
    
    #df for correlation and beta
    df_returns = pd.DataFrame(index = ETHUSDT.index)
    df_returns['BTCUSDT'] = BTCUSDT['return_BTCUSDT']
    for i in tickers:
        df_returns[f'{i}'] = globals()[f'{i}'][f'return_{i}']
        
    # df for relative strength days
    df_rel_str = pd.DataFrame(index=ETHUSDT.index)
    df_rel_str['BTCUSDT'] = BTCUSDT['cum_return_BTCUSDT']
    for i in tickers:
        df_rel_str[f'{i}'] = globals()[f'{i}'][f'cum_return_{i}']

        
    #beta
    covariance = df_returns.cov()
    beta = covariance['BTCUSDT']/df_returns['BTCUSDT'].var()
    beta = beta.round(2)
    #correlation
    df_corr = df_returns.corr()
    correlation = df_corr['BTCUSDT']
    correlation = correlation.round(2)
    
    #best and worst performers
    last_value_series = df_rel_str[-1::].max()
    best_performers = round(last_value_series.nlargest(top_percent)*100, 2)
    worst_performers = round(last_value_series.nsmallest(top_percent)*100, 2)
    
    best_perf_list = best_performers.index.to_list()
    worst_perf_list = worst_performers.index.to_list()

    if timeframe == '5m':
        a = int((periods / 12))
        days_hours = 'hours'
    if timeframe == '15m':
        a = int((periods / 4))
        days_hours = 'hours'
    if timeframe == '1h':
        a = int(periods)
        days_hours = 'hours'
    if timeframe == '4h':
        a = int((periods * 4))
        days_hours = 'hours'
    if timeframe == '1d':
        a = int(periods)
        days_hours = 'days'

    
    #function to create tables
    def tables(series_top_tokens, beta_series, corr_series):
        
        #creating a dataframe for a table best perf
        dfp = pd.DataFrame(index = series_top_tokens.keys())
        dfp['Return %'] = series_top_tokens
        dfp['Beta'] = dfp.index.map(dict(zip(beta_series.index,beta_series)))
        dfp['Correlation'] = dfp.index.map(dict(zip(corr_series.index,corr_series)))
    
        #creating a table
        plt.clf()
        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis

        table(ax, dfp, loc = 'center')  # where df is your data frame
        plt.savefig('mytable.png', bbox_inches = 'tight')
        
        return
    
    #calling functions, renaming tables
    tables(best_performers, beta, correlation)
    try:
        os.rename('mytable.png', 'mytable_best.png')
    except:
        os.remove('mytable_best.png')
        os.rename('mytable.png', 'mytable_best.png')

    tables(worst_performers, beta, correlation)
    try:
        os.rename('mytable.png', 'mytable_worst.png')
    except:
        os.remove('mytable_worst.png')
        os.rename('mytable.png', 'mytable_worst.png')


    #function to create plots
    def charts(kind, top_bottom, df_rel_str, perf_list, timeframe):
        df_plot = pd.DataFrame(index = df_rel_str.index)
        df_plot['BTCUSDT'] = df_rel_str['BTCUSDT']
        for i in perf_list:
            try:
                df_plot[f'{i}'] = df_rel_str[f'{i}']
            except:
                pass

        plt.figure(figsize=(12, 8))
        plt.title(f'{kind} performers & BTC, last {a} {days_hours}, Binance futures, volume > ${int(min_vol/1000000)}m, {top_bottom} {int(percentage*100)}%')
        for asset in df_plot:
            if asset != 'BTCUSDT':
                plt.plot(df_plot.index, df_plot[asset], label = f'{asset} {round(df_plot[asset][-1]*100, 2)}%')
                plt.annotate(asset, xy=(0.95, df_rel_str[asset][-1]), xytext=(8, 0), 
                     xycoords=('axes fraction', 'data'), textcoords='offset points', size = 8, weight = 'bold')
                plt.legend(loc = 'best')
            else:
                plt.plot(df_plot.index, df_plot[asset], label = f'{asset} {round(df_plot[asset][-1]*100, 2)}%', linestyle = 'dashed')
                plt.annotate(asset, xy=(0.95, df_rel_str[asset][-1]), xytext=(8, 0), 
                     xycoords=('axes fraction', 'data'), textcoords='offset points', size = 8, weight = 'bold')
                plt.legend(loc = 'lower left')
        plt.xlabel('Datetime UTC', fontsize = 13)
        plt.ylabel('Return', fontsize = 13)
        plt.grid(axis='y')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))

        plt.savefig('mychart.png')

        return

    charts('Best', 'top', df_rel_str, best_perf_list, a)
    try:
        os.rename('mychart.png', 'mychart_best.png')
    except:
        os.remove('mychart_best.png')
        os.rename('mychart.png', 'mychart_best.png')

    charts('Worst', 'bottom', df_rel_str, worst_perf_list, a)
    try:
        os.rename('mychart.png', 'mychart_worst.png')
    except:
        os.remove('mychart_worst.png')
        os.rename('mychart.png', 'mychart_worst.png')

    
    # sending to a bot
    send('%23best_worst_perf')
    send(f'Best Performers, last {a} hours, in %: \n{best_performers.to_string()}')
    sendimage('mytable_best.png')
    sendimage('mychart_best.png')
    send(f'Worst Performers, last {a} hours, in %: \n{worst_performers.to_string()}')
    sendimage('mytable_worst.png')
    sendimage('mychart_worst.png')

    
    return

    
def last7d():
    main_function(timeframe = '1h', periods = 168)

