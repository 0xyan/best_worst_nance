import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import table


def send(token_tg, id_tg, text):
    url = 'https://api.telegram.org/bot'+token_tg+'/sendMessage?chat_id='+id_tg+'&text='+text+''
    resp = requests.get(url)
    r = resp.json()
    return

def sendimage(token_tg, id_tg, img):
    url = 'https://api.telegram.org/bot'+token_tg+'/sendPhoto'
    f = {'photo': open(img, 'rb')}
    d = {'chat_id': id_tg}
    resp = requests.get(url, files = f, data = d)
    r = resp.json()
    return

def timeframe_formatter(timeframe, periods):
    #candlesticks to hour format
    if timeframe == '5m':
        a = int((periods / 12))
        days_hours = 'hours'
    if timeframe == '15m':
        a = int((periods / 4))
        days_hours = 'hours'
    if timeframe == '1h':
        a = int(periods / 24)
        days_hours = 'days'
    if timeframe == '4h':
        a = int((periods / 6))
        days_hours = 'days'
    if timeframe == '1d':
        a = int(periods)
        days_hours = 'days'

    return a, days_hours

#function to create tables
def tables(series_top_tokens, beta_series, corr_series):
    #creating a dataframe for a table best perf
    dfp = pd.DataFrame(index = series_top_tokens.index)
    dfp['Return %'] = series_top_tokens.values
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

#function to create plots
def charts(a_periods, days_hours, kind, top_bottom, min_vol, percentage, top_df):

    plt.figure(figsize=(12, 8))
    plt.title(f'{kind} performers & BTC, last {a_periods} {days_hours}, Binance futures, volume > ${int(min_vol/1000000)}m, {top_bottom} {int(percentage*100)}%')
    for asset in top_df:
        if asset != 'BTCUSDT':
            plt.plot(top_df.index, top_df[asset], label = f'{asset} {round(top_df[asset][-1]*100, 2)}%')
            plt.annotate(asset, xy=(0.95, top_df[asset][-1]), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points', size = 8, weight = 'bold')
            plt.legend(loc = 'best')
        else:
            plt.plot(top_df.index, top_df[asset], label = f'{asset} {round(top_df[asset][-1]*100, 2)}%', linestyle = 'dashed')
            plt.annotate(asset, xy=(0.95, top_df[asset][-1]), xytext=(8, 0), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points', size = 8, weight = 'bold')
            plt.legend(loc = 'lower left')
    plt.xlabel('Datetime UTC', fontsize = 13)
    plt.ylabel('Return', fontsize = 13)
    plt.grid(axis='y')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))

    plt.savefig('mychart.png')

    return