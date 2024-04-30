import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import table
import requests
import schedule
import time
import asyncio
import os
from binance import AsyncClient
from utils import send, sendimage, timeframe_formatter, tables, charts
from dotenv import load_dotenv

load_dotenv()

def binance_init():
    binance_api_key = os.getenv("BINANCE_API_KEY")
    binance_secret = os.getenv("BINANCE_SECRET")
    client = AsyncClient(binance_api_key, binance_secret)

    return client

def tg_init():
    token_tg = os.getenv("TELEGRAM_TOKEN")
    id_tg = os.getenv("TELEGRAM_ID")

    return token_tg, id_tg

async def pairs(client):
    symbols = []
    raw_data = await client.futures_exchange_info()
    for ticker in raw_data['symbols']:
        if ticker['quoteAsset'] == 'USDT' and ticker['contractType'] == 'PERPETUAL':
            symbols.append(ticker['symbol'])
    return symbols

#function to create dataframes with historic data
async def futures_continous_klines(client, symbol, interval, start_str=None, limit=None):
    df = pd.DataFrame()
    try:
        if start_str is not None and limit is not None:
            klines_data = await client.futures_continous_klines(pair=symbol, interval=interval, contractType='PERPETUAL', start_str=start_str, limit=limit)
        elif start_str is not None and limit is None:
            klines_data = await client.futures_continous_klines(pair=symbol, interval=interval, contractType='PERPETUAL', start_str=start_str)
        elif start_str is None and limit is not None:
            klines_data = await client.futures_continous_klines(pair=symbol, interval=interval, contractType='PERPETUAL', limit=limit)
        else:
            raise ValueError('start_str or limit has to be provided')
        
        dfi = pd.DataFrame(klines_data)
        df['time'] = dfi[0].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit = 'ms')
        df['close'] = dfi[4].astype(float)
        df[f'volume_{symbol}'] = dfi[7].astype(float)
        df['return'] = df['close'].pct_change(1)
        df[f'{symbol}'] = (df['return'] + 1).cumprod() - 1
        #set_index
        datetime_series = df['time']
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        df = df.set_index(datetime_index)
        df.dropna(inplace = True)
        df.drop(['time', 'close'], axis=1, inplace=True)
    except Exception as e:
        print(f'error processing {symbol}: {e}')
    
    return df

async def ticker_volume_filtering(client, tickers, min_vol):
    volumes = {}
    filtered_tickers = []

    async def process_ticker(ticker):
        try:
            df = await futures_continous_klines(client=client, symbol=ticker, interval='1d', limit=3)
            return ticker, df[f'volume_{ticker}'][0]
        except Exception as e:
            print(f'an issue with {ticker}: {e}')
        return ticker, None
    
    tasks = [process_ticker(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    for ticker, volume in results:
        if volume is not None:
            volumes[ticker] = volume

    for k,v in volumes.items():
        if v > min_vol:
            filtered_tickers.append(k)
    
    return filtered_tickers

async def df_creation(client, filtered_tickers, interval, start_str=None, limit=None):
    df_dict = {}

    async def process_ticker(ticker):
        try:
            df_dict[ticker] = await futures_continous_klines(client, ticker, interval, start_str, limit)
            return ticker, df_dict[ticker]
        except Exception as e:
            print(f'error occurred in df_creation, {ticker}: {e}')
    
    tasks = [process_ticker(ticker) for ticker in filtered_tickers]
    results = await asyncio.gather(*tasks)

    for ticker, df in results:
        if df is not None:
            df_dict[ticker] = df

    return df_dict


def best_worst_coins(filtered_tickers, df_dict, percentage=0.2):
    top_percentage = int(round(len(filtered_tickers) * percentage, 0))

    cum_perf = pd.DataFrame()
    for k,v in df_dict.items():
        cum_perf[k] = df_dict[k][k]

    #.max() here simply turns df into series
    last_value_series = cum_perf[-1::].max()
    best_performers = round(last_value_series.nlargest(top_percentage)*100, 2)
    worst_performers = round(last_value_series.nsmallest(top_percentage)*100, 2)
    best_perf_list = best_performers.index.to_list()
    worst_perf_list = worst_performers.index.to_list()
    best_perf_list.append('BTCUSDT')
    worst_perf_list.append('BTCUSDT')
    return best_perf_list, worst_perf_list

def best_worst_df(top_list, df_dict):
    df = pd.DataFrame()
    for ticker in top_list:
        try:
            df[ticker] = df_dict[ticker][ticker]
        except Exception as e:
            print(f'Error occurred while processing {ticker}: {e}')
    return df

def beta_correlation(top_tokens_series, df_dict):
    df_returns = pd.DataFrame()
    for ticker in top_tokens_series:
        try:
            df_returns[ticker] = df_dict[ticker]['return']
        except Exception as e:
            print(f'problem in beta_correlation with {ticker}: {e}')

    #beta
    covariance = df_returns.cov()
    beta = covariance['BTCUSDT']/df_returns['BTCUSDT'].var()
    beta = beta.round(2)
    #correlation
    df_corr = df_returns.corr()
    correlation = df_corr['BTCUSDT']
    correlation = correlation.round(2)
    return beta, correlation


async def main(min_vol, interval, percentage=0.2, start_str=None, limit=None):
    client = binance_init()
    token_tg, id_tg = tg_init()
    symbols = await pairs(client)
    filtered_tickers = await ticker_volume_filtering(client, symbols, min_vol)
    df_dict = await df_creation(client=client, filtered_tickers= filtered_tickers, interval=interval, start_str=start_str, limit=limit)
    best_list, worst_list = best_worst_coins(filtered_tickers, df_dict, percentage)
    best_df = best_worst_df(best_list, df_dict)
    worst_df = best_worst_df(worst_list, df_dict)
    best_beta, best_correlation = beta_correlation(best_df, df_dict)
    worst_beta, worst_correlation = beta_correlation(worst_df, df_dict)
    best_performers = round(best_df[-1::].max() * 100, 2)
    worst_performers = round(worst_df[1::].max() * 100, 2)

    tables(best_performers, best_beta, best_correlation)
    try:
        os.rename('mytable.png', 'mytable_best.png')
    except:
        os.remove('mytable_best.png')
        os.rename('mytable.png', 'mytable_best.png')

    tables(worst_performers, worst_beta, worst_correlation)
    try:
        os.rename('mytable.png', 'mytable_worst.png')
    except:
        os.remove('mytable_worst.png')
        os.rename('mytable.png', 'mytable_worst.png')

    a_periods, days_hours = timeframe_formatter(interval, limit)


    charts(a_periods, days_hours, 'Best', 'top', min_vol, percentage, best_df)
    try:
        os.rename('mychart.png', 'mychart_best.png')
    except:
        os.remove('mychart_best.png')
        os.rename('mychart.png', 'mychart_best.png')

    charts(a_periods, days_hours, 'Worst', 'bottom', min_vol, percentage, worst_df)
    try:
        os.rename('mychart.png', 'mychart_worst.png')
    except:
        os.remove('mychart_worst.png')
        os.rename('mychart.png', 'mychart_worst.png')


    # sending to a bot
    send(token_tg, id_tg, '%23best_worst_perf')
    send(token_tg, id_tg, f'Best Performers, last {a_periods} hours, in %: \n{best_performers}')
    #sendimage(token_tg, id_tg, 'mytable_best.png')
    sendimage(token_tg, id_tg, 'mychart_best.png')
    send(token_tg, id_tg, f'Worst Performers, last {a_periods} hours, in %: \n{worst_performers}')
    #sendimage(token_tg, id_tg, 'mytable_worst.png')
    sendimage(token_tg, id_tg, 'mychart_worst.png')

def two_days():
    asyncio.run(main(min_vol=80000000, interval='15m', percentage=0.2, start_str=None, limit=192))

def week():
    asyncio.run(main(min_vol=80000000, interval='1h', percentage=0.2, start_str=None, limit=168))

'''
def setup_schedule():
    schedule.every().day.at("09:00").do(two_days)
    schedule.every().monday.at("15:00").do(week)
    schedule.every().wednesday.at("15:00").do(week)
    schedule.every().friday.at("15:00").do(week)

    while True:
        schedule.run_pending()
        time.sleep(1)
'''

if __name__ == "__main__":
    week()