import argparse
import os
import json
import csv
from itertools import groupby
from datetime import datetime
import sys


def location_func(k):
    return k['location']


def start_func(k):
    return k['start']


def get_timestamp(time_hit):
    return datetime.timestamp(datetime.strptime(time_hit, '%H:%M:%S'))


def get_diference_time(t1, t2):
    return get_timestamp(t1)-get_timestamp(t2)


def calcule_confusion_matrix(hits_tp, hits_gt, fp, fn, tp, total_current):
    if hits_gt['start'] < hits_tp['end'] <= hits_gt['end'] and hits_tp['start'] < hits_gt['start']:
        total_current += get_diference_time(hits_tp['end'], hits_tp['start'])
        fp += get_diference_time(hits_gt['start'], hits_tp['start'])
        fn += get_diference_time(hits_gt['end'], hits_tp['end'])
        tp += get_diference_time(hits_tp['end'], hits_gt['start'])
    elif hits_gt['start'] > hits_tp['start'] < hits_gt['end'] and hits_tp['end'] > hits_gt['end']:
        total_current += get_diference_time(hits_tp['end'], hits_tp['start'])
        tp += get_diference_time(hits_gt['end'], hits_gt['start'])
        fp += get_diference_time(hits_tp['end'], hits_gt['end'])
        fp += get_diference_time(hits_gt['start'], hits_tp['start'])
    elif hits_gt['start'] <= hits_tp['start'] < hits_gt['end']:
        total_current += get_diference_time(hits_tp['end'], hits_tp['start'])
        if hits_gt['end'] <= hits_tp['end']:
            tp += get_diference_time(hits_gt['end'], hits_tp['start'])
        if hits_gt['end'] > hits_tp['end']:
            tp += get_diference_time(hits_tp['end'], hits_tp['start'])
            fn += get_diference_time(hits_gt['end'], hits_tp['end'])
        if hits_tp['start'] > hits_gt['start']:
            fn += get_diference_time(hits_tp['start'], hits_gt['start'])
        if hits_gt['end'] < hits_tp['end']:
            fp += get_diference_time(hits_tp['end'], hits_gt['end'])
    return fp, fn, tp, total_current


def get_confusion_matrix(list_hits, location_gt, locations_tp):
    tp = 0.0
    fn = 0.0
    fp = 0.0
    total_expect = 0.0
    total_current = 0.0

    if location_gt == 'space 2':
        video_time = 60
    else:
        video_time = 58

    for hits_gt in sorted(list_hits, key=start_func):
        total_expect += get_diference_time(hits_gt['end'], hits_gt['start'])
        for location_tp, predict_value in groupby(locations_tp, location_func):
            print(location_tp,location_gt)
            if location_tp == location_gt:
                for hits_tp in sorted(list(predict_value), key=start_func):
                    print(hits_tp)
                    fp, fn, tp, total_current = calcule_confusion_matrix(
                        hits_tp, hits_gt, fp, fn, tp, total_current)
    tn = video_time - max(total_current, total_expect)
    return fp, fn, tp, tn


def get_sorted_metrics(file, user):
    return sorted(file['tracks'][user], key=location_func)


def row_with_match(user, set_tp, list_hits, location_gt, locations_tp):
    fp, fn, tp, tn = get_confusion_matrix(list_hits, location_gt, locations_tp)
    if tp == 0.0:
        return None
    return [user, set_tp, location_gt, -1, -1, -1, fp, fn, tp, tn, -1, -1]


def get_row_without_match_user(user, set_tp, list_hits, location):
    tp = 0.0
    fn = 0.0
    fp = 0.0
    if location == 'space 2':
        video_time = 60
    else:
        video_time = 58
    for hits_tp in sorted(list_hits, key=start_func):
        fp += get_diference_time(hits_tp['end'], hits_tp['start'])
    tn = video_time-fp
    return [user, set_tp, location, -1, -1, -1, fp, fn, tp, tn, -1, -1]


def metrics(ground_truth, tracking_predict, output, predict_ranges=[1, 3, 5]):
    with open(ground_truth, 'r') as file:
        gt_json = json.loads(file.read())

    with open(tracking_predict, 'r') as file:
        tp_json = json.loads(file.read())

    set_gt = int(gt_json['set'])
    set_tp = int(tp_json['set'])

    if set_gt == set_tp:
        results = [['user', 'set', 'location',
                    'accuracy detector', 'accuracy recognition',
                    'f1', 'fp', 'fn', 'tp', 'tn',
                    'precision', 'recall']]

        for user in tp_json['tracks'].keys():
            if user in gt_json['tracks'].keys():
                locations_tp = get_sorted_metrics(tp_json, user)
                locations_gt = get_sorted_metrics(gt_json, user)
                for location_gt, value in groupby(locations_gt, location_func):
                    row = row_with_match(user,
                                         set_tp,
                                         list(value),
                                         location_gt,
                                         locations_tp)
                    if row:
                        results.append(row)
            else:
                locations_tp = get_sorted_metrics(tp_json, user)
                for location_gt, value in groupby(locations_tp, location_func):
                    row = get_row_without_match_user(user,
                                                     set_tp,
                                                     list(value),
                                                     location_gt)
                    results.append(row)

        with open(output, mode='w') as file:
            results_writer = csv.writer(file)
            for rows in results:
                results_writer.writerow(rows)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground_truth', '-g', type=str,
                    help='Path to ground truth file.')
    ap.add_argument('--tracking_predict', '-p', type=str,
                    help='Path to tracking predict file.')
    ap.add_argument('--ouput', '-o', type=str,
                    help='Path to saved results file metrics.')
    argv = vars(ap.parse_args(argv))

    output = '{}/metrics.csv'.format(argv['ouput'])

    os.makedirs(argv['ouput'], exist_ok=True)

    print('Generating metrics file...')

    metrics(argv['ground_truth'], argv['tracking_predict'], output)


if __name__ == '__main__':
    main(sys.argv[1:])
