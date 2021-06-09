import pandas as pd
import pickle

if __name__ == '__main__':
    table = pd.read_csv('/Users/Benji/PycharmProjects/investing/Euronext_Equities_EU_2017-05-18.csv',
                        skiprows=[1, 2, 3, 4], sep=';')

    map = {
        'Euronext Brussels': '.BR',
        'Euronext Paris': '.PA',
        'Alternext Paris': '.PA',
        'Euronext Paris, London': '.PA',
        'Euronext Amsterdam': '.AS',
        'Euronext Brussels, Amsterdam': '.BR',
        'Euronext Amsterdam, Brussels': '.AS',
        'Euronext Paris, Amsterdam': '.PA',
        'Euronext Lisbon': '.LS',
        'Euronext Amsterdam, Brussels, Paris': '.AS',
        'Euronext Amsterdam, Paris': '.AS',
        'Euronext Brussels, Paris': '.BR',
        'Alternext Brussels': '.BR',
        'Euronext Amsterdam, London': '.AS',
        'Euronext Paris, Brussels': '.PA',
        'Euronext Paris, Amsterdam, Brussels': '.PA',
    }

    k = map.keys()
    ls = []
    for i in table.index:
        t = table.loc[i]
        if t['Market'] in k:
            yahoo_ticker = t['Symbol'] + map[t['Market']]
            ls += [yahoo_ticker]

    pickle.dump(ls, open('euronext_tickers.pkl', 'wb'))
