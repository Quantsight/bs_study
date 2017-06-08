from __future__ import print_function

import contextlib
import datetime
import sys

@contextlib.contextmanager
def timed_execution(descr, end=None, n=None, strm=sys.stderr):
    start = datetime.datetime.now()
    if end is None:
        print('%s ' % (descr, ))
    else:
        print('%s ' % (descr, ), end=end)
    strm.flush()
    yield
    td = datetime.datetime.now() - start
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hms = '%02d:%02d:%02d' % (hours, minutes, seconds)
    if n and td.seconds > 0:
        rate = n / float(td.seconds)
        hms += '%10d %8.2f/s' % (int(n), rate,)
    if end is None:
        print(hms)
    else:
        print(hms,end=end)
    strm.flush()


def ftp_creds(config_fname):
    with open(config_fname, 'r') as f:
        server   = f.readline()
        username = f.readline()
        password = f.readline()
    return server, username, password

import ftplib
def put_ftp(df, put_path, config_fname):
    server, username, password = ftp_creds(config_fname)
    session = ftplib.FTP(server, username, password)
    file = open('kitten.jpg', 'rb')  # file to send
    session.storbinary('STOR kitten.jpg', file)  # send the file
    file.close()  # close file and FTP
    session.quit()