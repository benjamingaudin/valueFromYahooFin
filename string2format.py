import datetime


def str2date(date):
    """ Converts dates from Yahoo (string) to datetime.
    """
    return datetime.datetime.strptime(date, "%m/%d/%Y").date()


def ele2nb(element):
    """ Converts numbers from Yahoo financial statements (strings with commas) to int.
        Returned number is in thousands.
    """
    if isinstance(element, str):
        return float(element.replace(',', ''))
    else:
        return element


def str2thousands(string):
    """ Converts numbers from Yahoo statistics (strings with letters like B or M) to int.
        Returned number is in thousands.
    """
    try:
        float(string)
        return float(string)
    except ValueError as e:
        if string[-1] == 'B':
            return float(string[:-1]) * 1e6
        elif string[-1] == 'M':
            return float(string[:-1]) * 1e3
        elif string[-1] == 'k':
            return float(string[:-1])
        else:
            raise e
