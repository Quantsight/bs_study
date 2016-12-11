from __future__ import print_function

import contextlib
import datetime
import sys

@contextlib.contextmanager
def timed_execution(descr, default_end=True, strm=sys.stderr):
    start = datetime.datetime.now()
    print('%s ' % (descr, ), end="", file=strm)
    strm.flush()
    yield
    td = datetime.datetime.now() - start
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hms = '%02d:%02d:%02d' % (hours, minutes, seconds)
    if default_end:
        print(hms, file=strm)
    else:
        print(hms, end="", file=strm)
    strm.flush()
