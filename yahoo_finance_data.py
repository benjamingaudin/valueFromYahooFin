from __future__ import division
import dryscrape
import webkit_server
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import pickle
import re
import datetime
import warnings
from find_file import find
from string2format import str2date, ele2nb, str2thousands
from time_stuff import countdown, print_lapse_of_time
import signal
import os.path
from multiprocessing import Pool
import time
import subprocess


class WebKitSession:
    def __init__(self, name=''):
        pass

        self.session = None
        self.server = None
        self.name = name
        self.open_session()

    def open_session(self):
        self.server = webkit_server.Server()
        server_conn = webkit_server.ServerConnection(server=self.server)
        driver = dryscrape.driver.webkit.Driver(connection=server_conn)
        self.session = dryscrape.Session(driver=driver)
        self.session.set_attribute('auto_load_images', False)
        WebKitSession.session = self.session
        WebKitSession.server = self.server
        print 'Open session {}'.format(self.name)

    def kill(self):
        if isinstance(self.server, webkit_server.Server):
            self.server.kill()
            self.session = None
            self.server = None
            print 'Session {} killed'.format(self.name)
        else:
            warnings.warn("kill() has no effect. Variable 'server' is not an instance of webkit_server.Server")


class GlobalNames:
    def __init__(self):
        pass

    income_statement = 'Income Statement'
    cash_flow = 'Cash Flow'
    balance_sheet = 'Balance Sheet'

    annual = 'Annual'
    quarterly = 'Quarterly'


class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException('Process takes too much time')


# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


class FetchStock:
    def __init__(self, quote, timeout=50, web_kit_session=None, statistics_only=False):
        kill_session_after_fetch = False
        self.quote = quote
        self.web_kit_session = web_kit_session
        if self.web_kit_session is None:
            self.web_kit_session = WebKitSession(self.quote)
            kill_session_after_fetch = True
        elif self.web_kit_session.session is None:
            self.web_kit_session.open_session()
        else:
            print 'session is opened'
        self.session = self.web_kit_session.session
        self.server = self.web_kit_session.server
        # keep this order for initializing the following variables
        try:
            self._visit_financials()
            signal.alarm(timeout)  # throws exception if alarm is not reset before timeout
            if not statistics_only:
                self.income_statement_annual_html = self._get_financials_html_table(GlobalNames.income_statement,
                                                                                    GlobalNames.annual)
                self.income_statement_quarterly_html = self._get_financials_html_table(GlobalNames.income_statement,
                                                                                       GlobalNames.quarterly)
                self.cash_flow_annual_html = self._get_financials_html_table(GlobalNames.cash_flow,
                                                                             GlobalNames.annual)
                self.cash_flow_quarterly_html = self._get_financials_html_table(GlobalNames.cash_flow,
                                                                                GlobalNames.quarterly)
                self.balance_sheet_annual_html = self._get_financials_html_table(GlobalNames.balance_sheet,
                                                                                 GlobalNames.annual)
                self.balance_sheet_quarterly_html = self._get_financials_html_table(GlobalNames.balance_sheet,
                                                                                    GlobalNames.quarterly)
                self.session.reset()
            self.statistics_html, self.price = self._get_statistics_html_div()
        except Exception as e:
            self.session.reset()
            if kill_session_after_fetch:
                self.web_kit_session.kill()
            raise e
        else:
            # Reset the alarm
            signal.alarm(0)

        print 'Got data for {}'.format(self.quote)

        self.session.reset()
        if kill_session_after_fetch:
            self.web_kit_session.kill()

    def fundamentals_to_pkl(self, folder):
        self.financials_to_pkl(GlobalNames.income_statement, GlobalNames.annual, folder)
        self.financials_to_pkl(GlobalNames.income_statement, GlobalNames.quarterly, folder)
        self.financials_to_pkl(GlobalNames.cash_flow, GlobalNames.annual, folder)
        self.financials_to_pkl(GlobalNames.cash_flow, GlobalNames.quarterly, folder)
        self.financials_to_pkl(GlobalNames.balance_sheet, GlobalNames.annual, folder)
        self.financials_to_pkl(GlobalNames.balance_sheet, GlobalNames.quarterly, folder)
        self.statistics_to_pkl(folder)

    def get_fundamentals_pandas(self):
        ia = self.get_financials_pandas_table(GlobalNames.income_statement, GlobalNames.annual)
        iq = self.get_financials_pandas_table(GlobalNames.income_statement, GlobalNames.quarterly)
        ca = self.get_financials_pandas_table(GlobalNames.cash_flow, GlobalNames.annual)
        cq = self.get_financials_pandas_table(GlobalNames.cash_flow, GlobalNames.quarterly)
        ba = self.get_financials_pandas_table(GlobalNames.balance_sheet, GlobalNames.annual)
        bq = self.get_financials_pandas_table(GlobalNames.balance_sheet, GlobalNames.quarterly)
        s = self.get_statistics_pandas_series()
        return {'income_statement_annual': ia, 'cash_flow_annual': ca, 'balance_sheet_annual': ba,
                'income_statement_quarterly': iq, 'cash_flow_quarterly': cq, 'balance_sheet_quarterly': bq,
                'statistics': s}

    def _visit_financials(self):
        url = 'http://finance.yahoo.com/quote/{0}/financials?p={0}'.format(self.quote)
        self.session.visit(url)

    def _get_financials_html_full(self, statement, frequency):
        if statement in [GlobalNames.cash_flow, GlobalNames.balance_sheet] and frequency == GlobalNames.annual:
            button = self.session.at_xpath('//*[normalize-space(text()) = "{}"]'.format(statement))
            button.click()
            button = self.session.at_xpath('//*[normalize-space(text()) = "{}"]'.format(frequency))
            button.click()
        if frequency == GlobalNames.quarterly:
            button = self.session.at_xpath('//*[normalize-space(text()) = "{}"]'.format(frequency))
            button.click()

        response = self.session.body()
        soup = BeautifulSoup(response, 'html.parser')
        return soup

    def _get_statistics_html_full(self):
        url = 'http://finance.yahoo.com/quote/{0}/key-statistics?p={0}'.format(self.quote)
        self.session.visit(url)
        response = self.session.body()
        soup = BeautifulSoup(response, 'html.parser')
        return soup

    def _get_financials_html_table(self, statement, frequency):
        soup = self._get_financials_html_full(statement, frequency)
        table_html = soup.find("table", class_="Lh(1.7) W(100%) M(0)")
        return table_html

    def _get_statistics_html_div(self):
        soup = self._get_statistics_html_full()
        div_html = soup.find("div", class_="Fl(start) W(50%)")
        price = float(soup.find("div", class_="D(ib) Fw(200) Mend(20px)").find("span").text.replace(',', ''))
        return div_html, price

    def get_statistics_pandas_series(self):
        div_html = self.statistics_html
        root = ET.fromstring(str(div_html))

        series = pd.Series()
        key = None
        for element in root.findall('.//tbody'):
            for child in element.findall('tr'):
                for gchild in child.findall('td'):
                    if gchild.find("span") is not None:
                        value = gchild.find("span").text
                    else:
                        value = gchild.text
                    if key is not None:
                        series[key] = value
                        key = None
                    if gchild.attrib == {}:
                        key = value
        series = series.set_value('price', self.price)
        return series

    def get_financials_pandas_table(self, statement, frequency):

        self.assertion_name(statement, frequency)

        if frequency == GlobalNames.annual:
            if statement == GlobalNames.income_statement:
                table_html = self.income_statement_annual_html
            elif statement == GlobalNames.cash_flow:
                table_html = self.cash_flow_annual_html
            elif statement == GlobalNames.balance_sheet:
                table_html = self.balance_sheet_annual_html
        elif frequency == GlobalNames.quarterly:
            if statement == GlobalNames.income_statement:
                table_html = self.income_statement_quarterly_html
            elif statement == GlobalNames.cash_flow:
                table_html = self.cash_flow_quarterly_html
            elif statement == GlobalNames.balance_sheet:
                table_html = self.balance_sheet_quarterly_html

        root = ET.fromstring(str(table_html))

        body = root[0]
        pandas_table = pd.DataFrame()
        current_entry = None
        values = None
        for child in body:
            for gchild in child:
                for ggchild in gchild:
                    if ggchild.text is not None:

                        if re.search('[a-zA-Z]', ggchild.text):
                            current_entry = ggchild.text
                            values = []
                        else:
                            if current_entry is not None:
                                if re.match('\d{1,2}/\d{1,2}/\d{4}', ggchild.text):
                                    current_entry = 'Date'
                                values.append(ggchild.text)

                    else:
                        for gggchild in ggchild:
                            if current_entry is not None:
                                values.append(gggchild.text)

            if current_entry == 'Date':
                length = len(values)
            while len(values) < length:
                values.insert(0, np.NaN)

            pandas_current = pd.Series(data=values, name=current_entry)
            if pandas_current is not None:
                if pandas_current.name in pandas_table.columns:
                    if not all(pd.isnull(pandas_table[pandas_current.name])):
                        if not all(pd.isnull(pandas_current)):
                            warnings.warn('Warning: we have two entries with the same name but with different values!')
                            pandas_table = pd.concat([pandas_table, pandas_current], join='outer', axis=1)
                    pandas_table[pandas_current.name] = pandas_current
                else:
                    pandas_table = pd.concat([pandas_table, pandas_current], join='outer', axis=1)

        if pandas_table.empty:
            warnings.warn('There is no data from Yahoo! for {0}'.format(self.quote))
            return pandas_table

        date_in_date_units = pandas_table.Date.apply(str2date)
        pandas_table.Date = date_in_date_units
        pandas_table.set_index('Date', drop=True, inplace=True)

        return pandas_table

    def financials_to_pkl(self, statement, frequency, folder):
        self.assertion_name(statement, frequency)

        table = self.get_financials_pandas_table(statement, frequency)
        if not table.empty:
            last_date = str(max(table.index.to_datetime()).date())
            table.to_pickle(r'{4}/{0}_{1}_{2}_{3}.pkl'.format(
                self.quote, statement.replace(" ", "_"), frequency, last_date, folder))
        else:
            if os.path.isfile('{0}/check/empty.pkl'.format(folder)):
                empties = pickle.load(open('{0}/check/empty.pkl'.format(folder), 'rb'))
                if not self.quote in empties:
                    empties.append(self.quote)
                    pickle.dump(empties, open('{0}/check/empty.pkl'.format(folder), 'wb'))
            else:
                pickle.dump([self.quote], open('{0}/check/empty.pkl'.format(folder), 'wb'))

    def statistics_to_pkl(self, folder):
        series = self.get_statistics_pandas_series()
        date_today = datetime.datetime.now().date()
        series.name = date_today
        series.to_pickle(r'{2}/{0}_Statistics_{1}.pkl'.format(self.quote, str(date_today), folder))

    @staticmethod
    def assertion_name(statement, frequency):
        assert isinstance(statement, str)
        statement = statement.title()
        assert statement in [GlobalNames.income_statement, GlobalNames.cash_flow, GlobalNames.balance_sheet]
        assert isinstance(frequency, str)
        frequency = frequency.title()
        assert frequency in [GlobalNames.annual, GlobalNames.quarterly]


class RetrieveStock:
    def __init__(self, quote, frequency, folder, verbose=False):
        if verbose:
            print 'Retrieving data for', quote
        self.verbose = verbose
        self.folder = folder
        self.quote = quote
        self.frequency = frequency
        self.income_statement = self._get_income_statement()
        self.cash_flow = self._get_cash_flow()
        self.balance_sheet = self._get_balance_sheet()
        self.statistics = self._get_statistics()
        self.fundamentals = self._get_fundamentals()  # dict of the other self variables

    def _get_income_statement(self):
        statement = GlobalNames.income_statement.replace(" ", "_")
        return self._get_data_from_file(statement)

    def _get_cash_flow(self):
        statement = GlobalNames.cash_flow.replace(" ", "_")
        return self._get_data_from_file(statement)

    def _get_balance_sheet(self):
        statement = GlobalNames.balance_sheet.replace(" ", "_")
        return self._get_data_from_file(statement)

    def _get_statistics(self):
        statement = 'Statistics'
        return self._get_data_from_file(statement)

    def _get_fundamentals(self):
        return {GlobalNames.income_statement: self.income_statement, GlobalNames.cash_flow: self.cash_flow,
                GlobalNames.balance_sheet: self.balance_sheet, 'statistics': self.statistics}

    def _get_data_from_file(self, statement):
        if statement == 'Statistics':
            string = '{0}_{1}_*.pkl'.format(self.quote, statement)
        else:
            string = '{0}_{1}_{2}_*.pkl'.format(self.quote, statement, self.frequency)
        file_paths = find(string, self.folder)
        assert file_paths, 'No file of that kind: ' + string
        date, i = self._most_recent_file_index(file_paths)
        file_path = file_paths[i]  # most recent date
        data = pd.read_pickle(file_path)
        if self.verbose:
            print 'Date for {0}: {1}'.format(statement, date)
        return data

    @staticmethod
    def _most_recent_file_index(file_paths):
        date_ref = None
        j = 0
        for i, file_path in enumerate(file_paths):
            date = re.search('\d{4}-\d{2}-\d{2}', file_path).group()
            date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            if date_ref is None or date > date_ref:
                date_ref = date
                j = i
        return date_ref, j


class ComputeRatios:
    def __init__(self, quote, folder, frequency=GlobalNames.quarterly):
        self.quote = quote
        self.frequency = frequency
        try:
            self.fundamentals = RetrieveStock(quote, self.frequency, folder).fundamentals
            self.fundamentals_trailing = RetrieveStock(quote, GlobalNames.annual, folder).fundamentals
        except AssertionError as e:
            if 'No file of that kind' in e.message and frequency == GlobalNames.quarterly:
                self.frequency = GlobalNames.annual
                print 'trying with annual frequency for ' + quote
                self.fundamentals = RetrieveStock(quote, self.frequency, folder).fundamentals
                self.fundamentals_trailing = self.fundamentals.copy()
                print 'we use annual data instead'
            else:
                raise e
        if self.frequency == GlobalNames.quarterly:
            self.income_statement = self._sum_quarters(GlobalNames.income_statement)
            self.cash_flow = self._sum_quarters(GlobalNames.cash_flow)
        else:
            self.income_statement = self._get_last(GlobalNames.income_statement).apply(ele2nb)
            self.cash_flow = self._get_last(GlobalNames.cash_flow)
        self.balance_sheet = self._get_last(GlobalNames.balance_sheet).apply(ele2nb)
        self.statistics = self.fundamentals['statistics']

        # extracted data
        self.market_cap = str2thousands(self.statistics['Market Cap (intraday)'])
        self.debt_asset_ratio = self.balance_sheet['Total Liabilities'] / self.balance_sheet['Total Assets']

        self.ebit = self.income_statement['Earnings Before Interest and Taxes']
        self.operating_income = self.income_statement['Operating Income or Loss']
        self.operating_income_adjusted = self.compute_adjusted_operating_income()
        self.free_cash_flow = self.compute_free_cash_flow()

        self.tangible_capital_employed = self.compute_tangible_capital_employed()
        if self.tangible_capital_employed < 0:
            print 'ERR capital in ', self.quote
            self.tangible_capital_employed = 1
        self.enterprise_value = self.compute_enterprise_value()
        if self.enterprise_value < 0:
            print 'ERR enterprise value in ', self.quote
            self.enterprise_value = 1e20
        self.opinc_on_tangible_capital = self.operating_income_adjusted / self.tangible_capital_employed
        self.fcf_on_tangible_capital = self.free_cash_flow / self.tangible_capital_employed
        self.opinc_yield = self.operating_income_adjusted / self.enterprise_value
        self.fcf_yield = self.free_cash_flow / self.enterprise_value
        self.return_on_tangible_capital = self.operating_income_adjusted / self.tangible_capital_employed
        self.earning_yield = self.operating_income_adjusted / self.enterprise_value
        self.quality_of_business = self.return_on_tangible_capital * 100
        self.cheapness = self.earning_yield * 100

    # computed ratios
    def compute_return_on_tangible_assets(self):
        pass

    def compute_tax_rate(self):
        i = self.income_statement
        return np.abs(i['Income Tax Expense'] / i['Income Before Tax'])

    def compute_free_cash_flow(self):
        c = self._trailing(GlobalNames.cash_flow)
        i = self._trailing(GlobalNames.income_statement)
        tau = self.compute_tax_rate()
        fcf = c['Total Cash Flow From Operating Activities'] + c['Capital Expenditures'] + \
              i['Interest Expense'] * (1 - tau)
        return fcf

    def compute_adjusted_operating_income(self):
        operating_income = self.income_statement['Operating Income or Loss']
        non_recurring_expenses = self.income_statement['Non Recurring']
        return operating_income + non_recurring_expenses

    def compute_adjusted_net_working_capital(self):
        receivables = self.balance_sheet['Net Receivables']
        inventory = self.balance_sheet['Inventory']
        payable = self.balance_sheet['Accounts Payable']
        short_term_investment = self.balance_sheet['Short Term Investments']
        other_current_assets = self.balance_sheet['Other Current Assets']
        other_current_liabilities = self.balance_sheet['Other Current Liabilities']
        adjusted_net_working_capital = receivables + inventory + short_term_investment + other_current_assets - \
                                       other_current_liabilities - payable
        return adjusted_net_working_capital

    def compute_net_fixed_assets(self):
        long_term_investments = self.balance_sheet['Long Term Investments']
        plant_equipment = self.balance_sheet['Property Plant and Equipment']
        accumulated_amortization =self.balance_sheet['Accumulated Amortization']
        other_assets = self.balance_sheet['Other Assets']
        deferred_lt_asset_changes = self.balance_sheet['Deferred Long Term Asset Charges']
        net_fixed_assets = long_term_investments + plant_equipment + accumulated_amortization + other_assets + \
                           deferred_lt_asset_changes
        return net_fixed_assets

    def compute_tangible_capital_employed(self):
        adjusted_net_working_capital = self.compute_adjusted_net_working_capital()
        net_fixed_assets = self.compute_net_fixed_assets()
        return net_fixed_assets + adjusted_net_working_capital

    def compute_enterprise_value(self):
        market_cap = self.market_cap
        total_debt =self.balance_sheet['Long Term Debt'] + self.balance_sheet['Short/Current Long Term Debt']
        equity_options = self.balance_sheet['Misc. Stocks Options Warrants']
        r_preferred_stock = self.balance_sheet['Redeemable Preferred Stock']
        preferred_stock = self.balance_sheet['Preferred Stock']
        minority_interest = self.balance_sheet['Minority Interest']
        cash = self.balance_sheet['Cash And Cash Equivalents']
        enterprise_value = market_cap + total_debt + preferred_stock + r_preferred_stock + equity_options + \
                           minority_interest - cash
        return enterprise_value

    def compute_price_to_book_ratio(self):
        b = self.balance_sheet
        book_value = b['Total Assets'] - b['Intangible Assets'] - b['Goodwill'] - b['Total Liabilities']
        return self.market_cap / book_value

    def compute_trailing_pe_ratio(self):
        net_income = self._trailing(GlobalNames.cash_flow)['Net Income']
        return self.market_cap / net_income

    def compute_trailing_dividend_yield(self):
        i = self._trailing(GlobalNames.cash_flow)
        return - i['Dividends Paid'] / self.market_cap

    def compute_net_net(self):
        """ Graham's net-net. If ratio larger than one, we get 'a dollar for 40 cents.'
        """
        b = self.balance_sheet
        cash = b['Cash And Cash Equivalents']
        receivables = b['Net Receivables']
        inventory = b['Inventory']
        fixed_assets = b['Total Assets'] - b['Total Current Assets']
        liabilities = b['Total Liabilities']
        liquidation_weights = np.array([1, 0.8, 2/3, 0.1, -1])
        liquidation_value = liquidation_weights * [cash, receivables, inventory, fixed_assets, liabilities]
        liquidation_value = liquidation_value.sum()
        return liquidation_value / self.market_cap

    def _get_last(self, statement):
        statement_ = self.fundamentals[statement]
        used_statement = statement_.loc[max(statement_.index)].copy()
        used_statement.fillna('0', inplace=True)  # be a little careful here, in case important values are missing
        return used_statement

    def _trailing(self, statement):
        statement_ = self.fundamentals_trailing[statement]
        statement_.fillna('0', inplace=True)  # be a little careful here, in case important values are missing
        statement_ = statement_.applymap(ele2nb)
        used_statement = statement_.mean()
        if statement == GlobalNames.income_statement:
            adj_non_recurring = self._adjust_repeated_data(statement_['Non Recurring'])
            used_statement['Non Recurring'] = adj_non_recurring / statement_.shape[0]
        return used_statement

    def _sum_quarters(self, statement):
        statement_ = self.fundamentals[statement]
        assert statement_.shape[0] == 4, 'It seems that we do not have data for four quarters for {0}, {1}. ' \
                                         'Please check the data.\nNumber of quarters: {2} instead of 4.'. \
            format(self.quote, statement, statement_.shape[0])
        statement_.fillna('0', inplace=True)  # be a little careful here, in case important values are missing
        statement_ = statement_.applymap(ele2nb)
        used_statement = statement_.sum()
        if statement == GlobalNames.income_statement:
            adj_non_recurring = self._adjust_repeated_data(statement_['Non Recurring'])
            used_statement['Non Recurring'] = adj_non_recurring

        return used_statement

    @staticmethod
    def _adjust_repeated_data(data):
        data = data.values
        ls = []
        for d in data:
            if d not in ls:
                ls.append(d)
        return np.sum(ls)


class Ranking:
    # plot distribution of the two scores and position of each stock?
    def __init__(self, list_quote, folder):
        self.ls = list_quote
        self.folder = folder
        self.quality_cheapness = self.rank_ratios
        self.score = self.quantile_score()

    @property
    def rank_ratios(self):
        quality = pd.Series(index=self.ls)
        cheapness = pd.Series(index=self.ls)
        capital = pd.Series(index=self.ls)
        enterprise_value = pd.Series(index=self.ls)
        operating_income = pd.Series(index=self.ls)
        operating_income_adjusted = pd.Series(index=self.ls)
        debt_asset_ratio = pd.Series(index=self.ls)
        market_cap = pd.Series(index=self.ls)
        net_net = pd.Series(index=self.ls)
        dividend_yield = pd.Series(index=self.ls)
        price_to_book_ratio = pd.Series(index=self.ls)
        pe_ratio = pd.Series(index=self.ls)
        fcf_capital = pd.Series(index=self.ls)
        opinc_capital = pd.Series(index=self.ls)
        fcf_yield = pd.Series(index=self.ls)
        opinc_yield = pd.Series(index=self.ls)
        free_cash_flow = pd.Series(index=self.ls)
        for q in self.ls:
            try:
                ratio = ComputeRatios(q, self.folder)
                quality[q] = ratio.quality_of_business
                cheapness[q] = ratio.cheapness
                capital[q] = ratio.tangible_capital_employed
                enterprise_value[q] = ratio.enterprise_value
                operating_income_adjusted[q] = ratio.operating_income_adjusted
                free_cash_flow[q] = ratio.free_cash_flow
                debt_asset_ratio[q] = ratio.debt_asset_ratio
                market_cap[q] = ratio.market_cap
                net_net[q] = ratio.compute_net_net()
                dividend_yield[q] = ratio.compute_trailing_dividend_yield()
                price_to_book_ratio[q] = ratio.compute_price_to_book_ratio()
                pe_ratio[q] = ratio.compute_trailing_pe_ratio()
                fcf_capital[q] = ratio.fcf_on_tangible_capital
                opinc_capital[q] = ratio.opinc_on_tangible_capital
                fcf_yield[q] = ratio.fcf_yield
                opinc_yield[q] = ratio.opinc_yield
                if ratio.operating_income < 0 < ratio.operating_income_adjusted:
                    operating_income[q] = 'x'
                print q, 'OK!'
            except Exception as e:
                print e.message
                print 'No data for ' + q
        quality_rank = quality.rank(ascending=False)
        cheapness_rank = cheapness.rank(ascending=False)
        return pd.concat(
            [quality_rank, cheapness_rank, quality, cheapness, capital, enterprise_value,
             operating_income_adjusted, operating_income, free_cash_flow, debt_asset_ratio, dividend_yield,
             price_to_book_ratio, pe_ratio, market_cap, fcf_capital, opinc_capital, fcf_yield,
             opinc_yield, net_net],
            axis=1, join='outer',
            keys=['quality_rank', 'cheapness_rank', 'quality', 'cheapness', 'capital', 'enterprise_value',
                  'operating_income_adjusted', 'opinc_negative', 'free_cash_flow', 'debt_asset_ratio', 'dividend_yield',
                  'price_to_book_ratio', 'pe_ratio', 'market_cap', 'fcf_capital', 'opinc_capital', 'fcf_yield',
                  'opinc_yield', 'net_net'])

    def quantile_score(self):
        q_c = self.quality_cheapness
        q_c['rank'] = q_c.quality_rank + q_c.cheapness_rank

        return q_c.sort('rank')


class Price:
    def __init__(self, quotes, folder):
        self.folder = folder
        if isinstance(quotes, str):
            self.quotes = [quotes]
        else:
            self.quotes = quotes

    def get(self):
        prices = pd.Series(index=self.quotes)
        for q in self.quotes:
            prices[q] = RetrieveStock(q, GlobalNames.quarterly, self.folder).statistics['price']
        return prices


class Allocation:
    def __init__(self, quotes, folder, capital):
        self.quotes = quotes
        self.folder = folder
        self.capital = capital
        self.share, self._remaining_capital, self._temp_capital_allocation = self._rounding_algorithm()
        self.capital = self._capital_allocation()

    def _rounding_algorithm(self):
        price = Price(self.quotes, self.folder).get()
        capital_per_stock = self.capital / float(len(price))
        share_allocation_first_round = np.floor(capital_per_stock / price)
        get_more = pd.Series(data=True, index=self.quotes)
        remaining_capital = self.capital - share_allocation_first_round.dot(price)
        share_allocation_second_round = share_allocation_first_round
        still_to_go = capital_per_stock - share_allocation_first_round * price

        while any(get_more):
            still_to_go *= get_more
            farthest = still_to_go.argmax()
            if 2 * still_to_go[farthest] >= price[farthest] and price[farthest] <= remaining_capital:
                share_allocation_second_round[farthest] += 1
                still_to_go = capital_per_stock - share_allocation_first_round * price
                remaining_capital -= price[farthest]
            get_more[farthest] = False

        return share_allocation_second_round, remaining_capital, share_allocation_second_round * price

    def _capital_allocation(self):
        capital_allocation = self._temp_capital_allocation
        return capital_allocation.set_value('cash', self._remaining_capital)


def fetch_fundamentals(quote, folder, timeout=50, web_kit_session=None):
    try:
        s = FetchStock(quote, timeout=timeout, web_kit_session=web_kit_session)
        s.fundamentals_to_pkl(folder)
        Success.success[quote] = True
        Success.save_success()
        print Success.success
    except Exception as e:
        print 'Error in {}'.format(quote)
        print e


class Success:
    def __init__(self):
        pass

    success = None
    file_success = None

    @classmethod
    def get_success(cls):
        cls.load_success()
        return cls.success

    @classmethod
    def load_success(cls):
        cls.success = pd.read_pickle(cls.file_success)

    @classmethod
    def save_success(cls):
        cls.success.to_pickle(cls.file_success)
        print 'saved'


def download_data(quote_list, folder, first_index, last_index, tries=3, timeout=50):
    quote_list = quote_list[first_index:last_index]
    first_index = str(first_index)
    last_index = str(last_index)

    str1 = (4 - len(first_index)) * '0' + first_index
    str2 = (4 - len(last_index)) * '0' + last_index
    print 'Now looking at quotes {}_{}'.format(str1, str2)
    file_success = '{2}/check/success_{0}_{1}.pkl'.format(str1, str2, folder)
    file_failed = '{2}/check/failed_{0}_{1}.pkl'.format(str1, str2, folder)

    if not os.path.isfile(file_success):
        success = pd.Series(False, index=quote_list)
        success.to_pickle(file_success)
    Success.file_success = file_success
    success = Success.get_success()
    quotes = success.ix[~success.values].index.values

    while tries > 0 and len(quotes) > 0:
        for q in quotes:
            print q

            fetch_fundamentals(q, folder=folder, timeout=timeout)

        tries -= 1
        success = Success.get_success()
        print success
        quotes = success.ix[~success.values].index.values

    if len(quotes) == 0:
        print 'All succeeded!'
    else:
        print 'They failed:', quotes
        pickle.dump(list(quotes), open(file_failed, 'wb'))
        all_failures = '{0}/check/fail.pkl'.format(folder)
        if os.path.isfile(all_failures):
            fail = pickle.load(open(all_failures, 'rb'))
            fail = set(fail)
            for q in quotes:
                fail.add(q)
            pickle.dump(list(fail), open(all_failures, 'wb'))
        else:
            pickle.dump(list(quotes), open(all_failures, 'wb'))


def download_data_wrapper(arg):
    return download_data(*arg)


def grouper(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def splitter(seq, size):
    return ((pos[0], pos[-1] + 1) for pos in grouper(seq, size))


def download_data_multiprocess(quotes, folder,
                               first_index=None, last_index=None, size_before_nap=100, size=5, time_nap=60 * 4):
    folder = folder
    if not os.path.isdir(folder):
        os.makedirs(folder)
        os.makedirs(folder + '/check')
    start = time.time()
    if first_index is None:
        first_index = 0
    if last_index is None:
        last_index = len(quotes) - -1
    split_sessions = splitter(range(first_index, last_index), size_before_nap)
    nap = False
    for f_l_index in split_sessions:
        if nap:
            countdown(time_nap)
        else:
            nap = True
        p = Pool()
        iterator = ((quotes, folder, i, j) for i, j in splitter(range(f_l_index[0], f_l_index[1]), size))
        p.map(download_data_wrapper, iterator)
        p.close()
        subprocess.call(['pkill', 'webkit_server'])
        print_lapse_of_time(start, time.time())


def retry_failed(folder, tries=1, first_index=None, last_index=None, size_before_nap=100, size=5, time_nap=60 * 4):
    retry_failures = '{0}/check/fail.pkl'.format(folder)
    if os.path.isfile(retry_failures):
        quotes = pickle.load(open(retry_failures, 'rb'))
        all_failures = '{0}/check/fail_all.pkl'.format(folder)
        if not os.path.isfile(all_failures):
            pickle.dump(quotes, open(all_failures, 'wb'))
        while tries > 0:
            size += 1
            tries -= 1
            quotes = pickle.load(open(retry_failures, 'rb'))
            if not quotes:
                break
            pickle.dump([], open(retry_failures, 'wb'))

            download_data_multiprocess(quotes, folder, first_index, last_index, size_before_nap, size, time_nap)


def clean_check(folder):
    string = 'failed_*.pkl'
    file_paths = find(string, '{}/check'.format(folder))
    string = 'success_*.pkl'
    file_paths += find(string, '{}/check'.format(folder))
    for f in file_paths:
        os.remove(f)


def save_data(quotes, folder):
    download_data_multiprocess(quotes, folder)
    retry_failed(folder)
    clean_check(folder)


if __name__ == '__main__':

    # y = FetchStock('ALAGR.PA')
    # print ''
    # folder = '../data4'
    # apple = ComputeRatios('AAPL', folder, GlobalNames.annual)
    # empty = []
    # fail = []
    # file_empty = r'../data3/empty.pkl'
    # if os.path.isfile(file_empty):
    #     empty = pd.read_pickle(file_empty)
    # file_fail = r'../data3/fail.pkl'
    # if os.path.isfile(file_fail):
    #     fail = pd.read_pickle(file_fail)

    # quotes = pd.read_pickle('Russell3000skimmed2.pkl').index.values
    # folder = '../data4'
    # r = Ranking(quotes, folder)
    # # r.score.to_pickle('../results/score20170425adjopinc.pkl')
    # r.score.to_csv('../results/score20170426.csv')

    # apple = Ranking(['TSRA'], '../data3')
    # print apple.quality_of_business, apple.cheapness
    # apple = ComputeRatios('AAPL', folder, GlobalNames.annual)
    # print apple.quality_of_business, apple.cheapness
    # raise Exception

    quotes_US = pd.read_pickle('Russell3000skimmed2.pkl').index.values
    quotes_europe = pickle.load(open('euronext_tickers.pkl', 'rb'))
    folder = '../data5US'
    save_data(quotes_US, folder)
    folder = '../data5europe'
    save_data(quotes_europe, folder)


    # ToDo
    # retry failed, get it and erase file  ... done
    # Negative tangible capital employed
    # sum is strange
    # organise code and files
    # keep process running  Preferences Systeme/Economiseur d'energie
    # kill webkit automatically after a timeout  ... done
    # only fetch statistics !!!  ... ok
    # TRI TK says no data, but data exist ... ok
    # Operating Income or Loss - Non Recurring  ... ok
    # check computations !!!

    # use trailing free cash flow (to avoid accounting scam like underestimation of amortization of goodwill)
    # add warning column if goodwill is large compared to fixed assets
    # ToDo



    # ComputeRatios('YHOO')

    # WebKitSession.open_session()
    # yahoo = FetchStock('SSYS').get_financials_pandas_table('Income Statement', 'Annual')
    #
    # yahoo.fundamentals_to_pkl()
    # WebKitSession.kill()

    # WebKitSession.open_session()
    # yahoo = FetchStock('YHOO')
    # yahoo.fundamentals_to_pkl()
    # WebKitSession.kill()
    # yahoo.kill()
    #
    # WebKitSession.open_session()
    # apple = FetchStock('AAPL')
    # apple.fundamentals_to_pkl()
    # google = FetchStock('GOOGL')
    # google.fundamentals_to_pkl()
    # WebKitSession.kill()

    # RetrieveStock('YHOO')

    # apple_ratios = ComputeRatios('AAPL')
    # print apple_ratios.quality_of_business, apple_ratios.cheapness
    #
    # google_ratios = ComputeRatios('GOOGL')
    # print google_ratios.quality_of_business, google_ratios.cheapness
    #
    # google_ratios = ComputeRatios('IBM')
    # print google_ratios.quality_of_business, google_ratios.cheapness
    #
    # print Ranking(['AAPL', 'GOOGL', 'YHOO']).score
    #
    # allocation = Allocation(['AAPL', 'GOOGL'], 2000)
    # print allocation.share
    # print allocation.capital


    # np.diag([0 if l in out else 1 for l in range(3)]) * m
