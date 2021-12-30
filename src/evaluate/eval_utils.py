def location_func(k):
    return k['location']


def sorted_metrics(file, user):
    return sorted(file['tracks'][user], key=location_func)


def get_header():
    return['user', 'set', 'location',
           'accuracy detection', 'accuracy recognition',
           'f1', 'precision', 'recall']
