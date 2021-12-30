
from datetime import datetime, timedelta

def get_timestamp(time_hit):
    return datetime.timestamp(datetime.strptime(time_hit, '%H:%M:%S'))


def diff_time(t1, t2):
    return get_timestamp(t1)-get_timestamp(t2)
