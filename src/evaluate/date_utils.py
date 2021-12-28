
from datetime import datetime, timedelta

def get_timestamp(time_hit):
    return datetime.timestamp(datetime.strptime(time_hit, '%H:%M:%S'))


def get_strptime(time_hit):
    return datetime.fromtimestamp(time_hit).strftime('%H:%M:%S')[1:]


def get_delta_time(delta):
    return str(timedelta(0, delta))


def get_diff_time(t1, t2):
    return get_timestamp(t1)-get_timestamp(t2)
