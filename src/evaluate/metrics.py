import argparse
import os
import json
import csv
from itertools import groupby
from datetime import datetime, timedelta
import sys


class ConfusionMatrix:
    def __init__(self, ground_truth, tracking_predict, predict_ranges=0):

        self.predict_ranges = predict_ranges

        with open(ground_truth, 'r') as file:
            self.json_gt = json.loads(file.read())

        with open(tracking_predict, 'r') as file:
            self.json_tp = json.loads(file.read())

    def calcule_confusion_matrix(self, hits_tp, hits_gt, fp, fn, tp):

        if hits_gt['start'] < hits_tp['end'] <= hits_gt['end'] and hits_tp['start'] < hits_gt['start']:
            self.json_tp['tracks'][self.user].remove(hits_tp)
            self.json_gt['tracks'][self.user].remove(hits_gt)
            fp += get_diference_time(hits_gt['start'], hits_tp['start'])
            fn += get_diference_time(hits_gt['end'], hits_tp['end'])
            tp += get_diference_time(hits_tp['end'], hits_gt['start'])
        elif hits_gt['start'] > hits_tp['start'] < hits_gt['end'] and hits_tp['end'] > hits_gt['end']:
            self.json_tp['tracks'][self.user].remove(hits_tp)
            self.json_gt['tracks'][self.user].remove(hits_gt)
            tp += get_diference_time(hits_gt['end'], hits_gt['start'])
            fp += get_diference_time(hits_tp['end'], hits_gt['end'])
            fp += get_diference_time(hits_gt['start'], hits_tp['start'])
        elif hits_gt['start'] <= hits_tp['start'] < hits_gt['end']:
            self.json_tp['tracks'][self.user].remove(hits_tp)
            self.json_gt['tracks'][self.user].remove(hits_gt)
            if hits_gt['end'] <= hits_tp['end']:
                tp += get_diference_time(hits_gt['end'], hits_tp['start'])
            if hits_gt['end'] > hits_tp['end']:
                tp += get_diference_time(hits_tp['end'], hits_tp['start'])
                fn += get_diference_time(hits_gt['end'], hits_tp['end'])
            if hits_tp['start'] > hits_gt['start']:
                fn += get_diference_time(hits_tp['start'], hits_gt['start'])
            if hits_gt['end'] < hits_tp['end']:
                fp += get_diference_time(hits_tp['end'], hits_gt['end'])
        return fp, fn, tp

    def match_time(self, hits_tp, hits_gt):
        print( hits_tp, hits_gt)
        print(hits_gt['start'] <= hits_tp['start'] < hits_gt['end'])
        if hits_gt['start'] < hits_tp['end'] <= hits_gt['end'] and hits_tp['start'] < hits_gt['start']:
            print("1")
            return get_diference_time(hits_tp['end'], hits_gt['start'])
        elif hits_gt['start'] > hits_tp['start'] < hits_gt['end'] and hits_tp['end'] > hits_gt['end']:
            print("2")
            return get_diference_time(hits_gt['end'], hits_gt['start'])
        elif hits_gt['start'] <= hits_tp['start'] < hits_gt['end']:
            print("3")
            delta_time_match = 0
            if hits_gt['end'] <= hits_tp['end']:
                print("4")
                delta_time_match+= get_diference_time(hits_gt['end'], hits_tp['start'])
            if hits_gt['end'] > hits_tp['end']:
                print("5")
                delta_time_match+= get_diference_time(hits_tp['end'], hits_tp['start'])
            return delta_time_match
        return 0

    def get_confusion_matrix(self, list_hits, locations_tp):
        tp = 0.0
        fn = 0.0
        fp = 0.0
        time_expect = 0.0

        for hits_gt in sorted(list_hits, key=start_func):
            time_expect += get_diference_time(hits_gt['end'], hits_gt['start'])
            for location_tp, predict_value in groupby(locations_tp, location_func):
                if location_tp == self.location_gt:
                    for hits_tp in sorted(list(predict_value), key=start_func):
                        fp, fn, tp = self.calcule_confusion_matrix(
                            hits_tp, hits_gt, fp, fn, tp)
                        if hits_gt not in  self.json_gt['tracks'][self.user]:
                            break
                if hits_gt not in  self.json_gt['tracks'][self.user]:
                    break
        return fp, fn, tp, time_expect

    def row_with_match(self, list_hits, locations_tp):
        fp, fn, tp, time_expect = self.get_confusion_matrix(
            list_hits, locations_tp)
        if tp == 0.0:
            return None
        return [self.user, self.set_tp, self.location_gt, -1, -1, -1, fp, fn, tp, time_expect, -1, -1]

    def get_row_without_match_user(self, list_hits):
        tp = 0.0
        fn = 0.0
        fp = 0.0
        time_expect = 0.0
        for hits_tp in sorted(list_hits, key=start_func):
            self.json_tp['tracks'][self.user].remove(hits_tp)
            fp += get_diference_time(hits_tp['end'], hits_tp['start'])
        return [self.user, self.set_tp, self.location_gt, -1, -1, -1, fp, fn, tp, time_expect, -1, -1]

    def calcule_acc_detector_model(self):
        seconds_tp = 0
        seconds_gt = 0
        for user in self.json_gt['tracks'].keys():

            locations_gt = get_sorted_metrics(self.json_gt, user)
            for location_gt, value in groupby(locations_gt, location_func):
                if location_gt:
                    for hits_gt in list(value):
                        seconds_gt += get_diference_time(
                            hits_gt['end'], hits_gt['start'])
                        for user in self.json_tp['tracks'].keys():
                            print(user)
                            locations_tp = get_sorted_metrics(
                                self.json_tp, user)
                            for location_tp, predict_value in groupby(locations_tp, location_func):
                                if location_tp == location_gt:
                                    for hits_tp in sorted(list(predict_value), key=start_func):
                                        print(hits_tp)
                                        seconds_tp += self.match_time(
                                            hits_tp, hits_gt)
                                        print(seconds_tp)
        return seconds_tp, seconds_gt

    def get_time_video(self, location):
        return 60 if location == 'space 2' else 58

    def calcule_tn(self, results):
        for row in results[1:]:
            video_time = self.get_time_video(row[2])
            row[9] = video_time - max(row[9], row[6]+row[8])
        return results

    def class_report(self, results, seconds_tp, seconds_gt):
        for row in results[1:]:
            row[3] = round(seconds_tp / seconds_gt, 2)
            row[4] = get_accuracy_recognition(
                row[6], row[7], row[8], row[9])
            row[10] = get_precission(row[8], row[7])
            row[11] = get_recall(row[8], row[6])
            row[5] = get_f_score(row[10], row[11])
        return results

    def remove_unknown(self):
        if 'UNK' in self.json_tp['tracks']:
            del self.json_tp['tracks']['UNK']

    def get_header(self):
        return['user', 'set', 'location',
               'accuracy detector', 'accuracy recognition',
               'f1', 'fp', 'fn', 'tp', 'tn',
               'precision', 'recall']

    def apply_windows_range(self, windows_range):
        zero_time = get_timestamp('00:00:00')
        print("##")
        for location_gt in self.json_gt['tracks'].values():
            if location_gt:
                for hits_gt in list(location_gt):
                    start_time = get_timestamp(hits_gt['start'])-windows_range
                    if start_time >= zero_time:
                        hits_gt['start'] = get_strptime(start_time)
                    video_time = self.get_time_video(hits_gt['location'])
                    final_Video_time = get_timestamp(
                        get_delta_time(video_time))
                    end_time = get_timestamp(hits_gt['end'])+windows_range
                    if end_time <= final_Video_time:
                        hits_gt['end'] = get_strptime(end_time)
                    print(hits_gt)
        print("##")

    def metrics(self, output, windows_range=0):

        set_gt = int(self.json_gt['set'])
        self.set_tp = int(self.json_tp['set'])

        if set_gt == self.set_tp:
            results = [self.get_header()]

            self.apply_windows_range(windows_range)

            seconds_tp, seconds_gt = self.calcule_acc_detector_model()
            print(seconds_tp, seconds_gt)
            self.remove_unknown()

            for user in self.json_tp['tracks'].keys():
                locations_tp = get_sorted_metrics(self.json_tp, user)
                self.user = user
                if user in self.json_gt['tracks'].keys():
                    locations_gt = get_sorted_metrics(self.json_gt, user)
                    for location_gt, value in groupby(locations_gt, location_func):
                        self.location_gt = location_gt
                        row = self.row_with_match(list(value), locations_tp)
                        if row:
                            results.append(row)
                else:
                    for location_gt, value in groupby(locations_tp, location_func):
                        self.location_gt = location_gt
                        row = self.get_row_without_match_user(list(value))
                        results.append(row)

            for user, location_tp in self.json_tp['tracks'].items():
                if location_tp:
                    for hits_tp in list(location_tp):
                        idx_list = search_row_metrics(
                            user, hits_tp['location'], results)
                        if idx_list:
                            self.json_tp['tracks'][user].remove(hits_tp)
                            results[idx_list[0]][6] += get_diference_time(
                                hits_tp['end'], hits_tp['start'])
                        else:
                            fp = 0.0
                            fn = 0.0
                            tp = 0.0
                            time_expect = 0.0
                            self.json_tp['tracks'][user].remove(hits_tp)
                            fp = get_diference_time(
                                hits_tp['end'], hits_tp['start'])
                            row = [user, self.set_tp, hits_tp['location'],
                                   -1, -1, -1, fp, fn, tp, time_expect, -1, -1]
                            results.append(row)

            for user, location_gt in self.json_gt['tracks'].items():
                if location_gt:
                    for hits_gt in list(location_gt):
                        idx_list = search_row_metrics(
                            user, hits_gt['location'], results)
                        if idx_list:
                            self.json_gt['tracks'][user].remove(hits_gt)
                            results[idx_list[0]][7] += get_diference_time(
                                hits_gt['end'], hits_gt['start'])
                            results[idx_list[0]][9] += get_diference_time(
                                hits_gt['end'], hits_gt['start'])
                        else:
                            fp = 0.0
                            fn = 0.0
                            tp = 0.0
                            self.json_gt['tracks'][user].remove(hits_gt)
                            fn = get_diference_time(
                                hits_gt['end'], hits_gt['start'])
                            time_expect = fn
                            row = [user, self.set_tp, hits_gt['location'],
                                   -1, -1, -1, fp, fn, tp, time_expect, -1, -1]
                            results.append(row)

            results = self.calcule_tn(results)
            results = self.class_report(results, seconds_tp, seconds_gt)

            with open(output, mode='w') as file:
                results_writer = csv.writer(file)
                for rows in results:
                    results_writer.writerow(rows)


def get_recall(tp, fn):
    try:
        return round(tp / (tp + fn), 2)
    except ZeroDivisionError:
        return 0.0


def get_precission(tp, fp):
    try:
        return round(tp / (tp + fp), 2)
    except ZeroDivisionError:
        return 0.0


def get_f_score(precission, recall):
    try:
        return round((2*precission*recall) / (precission+recall), 2)
    except ZeroDivisionError:
        return 0.0


def get_accuracy_recognition(fp, fn, tp, tn):
    try:
        return round((tp + tn) / (tp + fp + tn + fn), 2)
    except ZeroDivisionError:
        return 0.0


def same_row(row, user, location):
    return row[0] == user and row[2] == location


def search_row_metrics(user, location, list_results):
    return [idx for idx, row in enumerate(list_results) if same_row(row, user, location)]


def location_func(k):
    return k['location']


def start_func(k):
    return k['start']


def get_timestamp(time_hit):
    return datetime.timestamp(datetime.strptime(time_hit, '%H:%M:%S'))


def get_strptime(time_hit):
    return datetime.fromtimestamp(time_hit).strftime('%H:%M:%S')


def get_delta_time(delta):
    return str(timedelta(0, delta))


def get_diference_time(t1, t2):
    return get_timestamp(t1)-get_timestamp(t2)


def get_sorted_metrics(file, user):
    return sorted(file['tracks'][user], key=location_func)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground_truth', '-g', type=str,
                    help='Path to ground truth file.')
    ap.add_argument('--tracking_predict', '-p', type=str,
                    help='Path to tracking predict file.')
    ap.add_argument('--ouput', '-o', type=str,
                    help='Path to saved results file metrics.')
    argv = vars(ap.parse_args(argv))

    

    os.makedirs(argv['ouput'], exist_ok=True)

    print('Generating metrics file...')

    for windows_range in [0, 3, 5]:
        output = '{}/metrics_{}_windows.csv'.format(argv['ouput'],windows_range)
        ConfusionMatrix(argv['ground_truth'], argv['tracking_predict']).metrics(
            output, windows_range)


if __name__ == '__main__':
    main(sys.argv[1:])
