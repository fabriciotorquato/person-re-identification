import argparse
import datetime
import json
import os
from database import Tracker, getConnection, select_all
from tqdm import tqdm
from datetime import timedelta


def get_strptime(time_hit):
    return datetime.datetime.fromtimestamp(time_hit).strftime('%H:%M:%S')[1:]


def get_timestamp(time_hit):
    return datetime.datetime.timestamp(datetime.datetime.strptime(time_hit, '%H:%M:%S'))


def get_delta_time(delta):
    return str(timedelta(1000, delta))


def search_end(list_values, current_location, current_begin):
    for idx, row in enumerate(list_values):
        delta_time = get_strptime(get_timestamp(row['end'])+1)
        if row['location'] == current_location and current_begin <= delta_time:
            return idx
    return None


def remove_same_time(row):
    return row['start'] != row['end']


def resume(set, results, output):
    json_results = {'set': set, 'tracks': {}}
    for row in tqdm(results):
        row_tracker = Tracker(*row)
        if row_tracker.predict_name not in json_results['tracks'].keys():
            json_results['tracks'][row_tracker.predict_name] = []

        start = str(datetime.timedelta(
            milliseconds=row_tracker.start)).split('.')[0]
        end = str(datetime.timedelta(
            milliseconds=row_tracker.end)).split('.')[0]

        idx = search_end(json_results['tracks'][row_tracker.predict_name],
                         row_tracker.location,
                         start)

        if idx:
            json_results['tracks'][row_tracker.predict_name][idx]['end'] = end
        else:
            json_results['tracks'][row_tracker.predict_name].append(
                {'location': row_tracker.location, 'start': start, 'end': end})

    for user in json_results['tracks'].keys():
        json_results['tracks'][user] = list(filter(
            remove_same_time, json_results['tracks'][user]))

    with open(output, 'w') as f:
        json.dump(json_results, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hash', '-x', default=0,
                    help='Number for sync many videos (default 0).')
    ap.add_argument('--video', '-v', default='/home',
                    help='Path to video file (default /home).')
    ap.add_argument('--set', '-s', default=0,
                    help='Set to video file (default 0).')
    ap.add_argument('--detector', '-d', default='facenet',
                    help='Detector mode: facenet or mobilenet')
    args = vars(ap.parse_args())

    output = '{}/../../output/set_{}'.format(args['video'], args['set'])
    os.makedirs(output, exist_ok=True)
    script_dir = os.path.dirname(__file__)
    output = os.path.join(
        script_dir, '{}/tracking_predict_db_{}.json'.format(output, args['detector']))

    print('Generating resume file...')

    con = getConnection()
    results = select_all(con, args['hash'])
    resume(args['set'], results, output)


if __name__ == '__main__':
    main()
