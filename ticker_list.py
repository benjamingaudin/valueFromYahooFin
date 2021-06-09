import StringIO
import csv
import urllib2
from bs4 import BeautifulSoup
import pandas as pd


class Russell3000Ticker:
    def __init__(self):
        pass

    def get(self):
        # self.save_table()
        table = pd.read_csv('Russell_3000.csv')
        l = []
        t_index = table.index
        for i, j in enumerate(table['#']):
            try:
                int(j)
                l.append(t_index[i])
            except Exception as e:
                print e
                if not l == []:
                    break
        table = table.iloc[l]
        table = table[table.Sector.apply(lambda x: str(x)).
            apply(lambda x: 'Bond' not in x and 'Debt' not in x and 'Financ' not in x and 'Utilities' not in x and 'Banks' not in x)]

        quotes = table.Symbol
        ls = []
        for q in quotes:
            try:
                q = q.replace('.', '-')
            except Exception as e:
                print e
            ls.append(q)

        # d = {}
        # symbols = table.Symbol.values
        # descriptions = table.Description.values
        # industry = table.Industry.values
        # for s in xrange(len(symbols)):
        #     d[symbols[s]] = {'description': descriptions[s], 'industry': industry[s]}

        table.index = ls
        table.to_pickle('Russell3000skimmed2.pkl')
        return table[['Description', 'Industry']]

    @staticmethod
    def save_table():
        url = 'http://www.kibot.com/Files/2/Russell_3000_Intraday.txt'
        hdr = {'User-Agent': 'Mozilla/5.0'}
        req = urllib2.Request(url, headers=hdr)
        page = urllib2.urlopen(req)
        soup = BeautifulSoup(page)
        txt_file = soup.find('p').text[117:-4]
        csv_out = csv.writer(open('Russell_3000.csv', 'w'), delimiter=',')

        f = StringIO.StringIO(txt_file)
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            csv_out.writerow(row)

        table = pd.read_csv('Russell_3000.csv')
        del table['#']
        table.to_pickle('Russell_3000.pkl')


if __name__ == '__main__':
    print Russell3000Ticker().get()
