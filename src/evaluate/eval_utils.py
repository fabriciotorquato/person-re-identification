def same_row(row, user, location):
    return row[0] == user and row[2] == location


def search_row_metrics(user, location, list_results):
    for idx, row in enumerate(list_results):
        if same_row(row, user, location):
            return idx
    return None


def location_func(k):
    return k['location']


def start_func(k):
    return k['start']


def get_sorted_metrics(file, user):
    return sorted(file['tracks'][user], key=location_func)

def get_header():
    return['user', 'set', 'location',
            'accuracy detector', 'accuracy recognition',
            'f1', 'fp', 'fn', 'tp', 'tn',
            'precision', 'recall']