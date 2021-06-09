import time

def countdown(t):
    mins, secs = divmod(t, 60)
    timeformat = '{:02d}m{:02d}s'.format(mins, secs)
    print '\n\n\nI take a short nap of {}.'.format(timeformat)
    while t:
        print '\rtime remaining ' + timeformat,
        time.sleep(1)
        t -= 1
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}m{:02d}s'.format(mins, secs)
    print '\rtime remaining ' + timeformat + '\nOk, back to work.\n\n\n'


def print_lapse_of_time(start_time, end_time):
    print (end_time - start_time), 'seconds'
    secs = end_time - start_time
    secs = int(round(secs))
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    timeformat = '{:02d}h{:02d}m{:02d}s'.format(hours, mins, secs)
    print timeformat


if __name__ == '__main__':

    countdown(4)
    # for i in xrange(10, 0, -1):
    #     time.sleep(1)
    #     sys.stdout.write(str(i) + ' ')
    #     sys.stdout.flush()