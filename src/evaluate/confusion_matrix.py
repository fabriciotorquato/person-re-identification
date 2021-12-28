
import json
import csv
from itertools import groupby
from eval_utils import *
from date_utils import *
from math_utils import *


class ConfusionMatrix:
    def __init__(self, ground_truth, tracking_predict, video_time, predict_ranges=0):

        self.predict_ranges = predict_ranges

        with open(ground_truth, 'r') as file:
            self.json_gt = json.loads(file.read())

        with open(tracking_predict, 'r') as file:
            self.json_tp = json.loads(file.read())

        with open(video_time, 'r') as file:
            self.video_time_json = json.loads(file.read())

    def calcule_confusion_matrix(self, hits_tp, hits_gt, fp, fn, tp):
        flag = False
        if hits_gt['start'] < hits_tp['end'] <= hits_gt['end'] and hits_tp['start'] < hits_gt['start']:
            self.json_tp['tracks'][self.user].remove(hits_tp)
            fp += get_diff_time(hits_gt['start'], hits_tp['start'])
            fn += get_diff_time(hits_gt['end'], hits_tp['end'])
            tp += get_diff_time(hits_tp['end'], hits_gt['start'])
            flag=True
        elif hits_gt['start'] > hits_tp['start'] < hits_gt['end'] and hits_tp['end'] > hits_gt['end']:
            self.json_tp['tracks'][self.user].remove(hits_tp)
            tp += get_diff_time(hits_gt['end'], hits_gt['start'])
            fp += get_diff_time(hits_tp['end'], hits_gt['end'])
            fp += get_diff_time(hits_gt['start'], hits_tp['start'])
            flag=True
        elif hits_gt['start'] <= hits_tp['start'] < hits_gt['end']:
            self.json_tp['tracks'][self.user].remove(hits_tp)
            if hits_gt['end'] <= hits_tp['end']:
                tp += get_diff_time(hits_gt['end'], hits_tp['start'])
                flag=True
            if hits_gt['end'] > hits_tp['end']:
                tp += get_diff_time(hits_tp['end'], hits_tp['start'])
                fn += get_diff_time(hits_gt['end'], hits_tp['end'])
                flag=True
            if hits_tp['start'] > hits_gt['start']:
                fn += get_diff_time(hits_tp['start'], hits_gt['start'])
                flag=True
            if hits_gt['end'] < hits_tp['end']:
                fp += get_diff_time(hits_tp['end'], hits_gt['end'])
                flag=True
        return fp, fn, tp,flag

    def match_time(self, hits_tp, hits_gt):
        if hits_gt['start'] < hits_tp['end'] <= hits_gt['end'] and hits_tp['start'] < hits_gt['start']:
            return get_diff_time(hits_tp['end'], hits_gt['start'])
        elif hits_gt['start'] > hits_tp['start'] < hits_gt['end'] and hits_tp['end'] > hits_gt['end']:
            return get_diff_time(hits_gt['end'], hits_gt['start'])
        elif hits_gt['start'] <= hits_tp['start'] < hits_gt['end']:
            delta_time_match = 0
            if hits_gt['end'] <= hits_tp['end']:
                delta_time_match += get_diff_time(
                    hits_gt['end'], hits_tp['start'])
            if hits_gt['end'] > hits_tp['end']:
                delta_time_match += get_diff_time(
                    hits_tp['end'], hits_tp['start'])
            return delta_time_match
        return 0

    def get_confusion_matrix(self, list_hits, locations_tp):
        tp = 0.0
        fn = 0.0
        fp = 0.0
        time_expect = 0.0
        for hits_gt in sorted(list_hits, key=start_func):
            time_expect += get_diff_time(hits_gt['end'], hits_gt['start'])
            flag = False
            for location_tp, predict_value in groupby(locations_tp, location_func):
                if location_tp == self.location_gt:
                    for hits_tp in sorted(list(predict_value), key=start_func):
                        if hits_tp in self.json_tp['tracks'][self.user]:
                            fp, fn, tp,flag_cl = self.calcule_confusion_matrix(hits_tp, hits_gt, fp, fn, tp)
                            if flag and flag_cl:
                                fn -= get_diff_time(hits_gt['end'], hits_gt['start'])
                            if flag_cl:
                                flag=flag_cl
                if flag:
                    break
            if not flag:                
                fn += get_diff_time(hits_gt['end'], hits_gt['start'])
        return fp, fn, tp, time_expect

    def row_with_match(self, list_hits, locations_tp):
        fp, fn, tp, time_expect = self.get_confusion_matrix(list_hits, locations_tp)
        return [self.user, self.set_tp, self.location_gt, -1, -1, -1, fp, fn, tp, time_expect, -1, -1]

    def get_row_without_match_user(self, list_hits):
        tp = 0.0
        fn = 0.0
        fp = 0.0
        time_expect = 0.0
        for hits_tp in sorted(list_hits, key=start_func):
            self.json_tp['tracks'][self.user].remove(hits_tp)
            fp += get_diff_time(hits_tp['end'], hits_tp['start'])
        return [self.user, self.set_tp, self.location_gt, -1, -1, -1, fp, fn, tp, time_expect, -1, -1]

    def get_time_video(self, location):
        return self.video_time_json[str(self.set_tp)][location]

    def calcule_tn(self, results):
        for row in results[1:]:
            video_time = self.get_time_video(row[2])
            row[9] = video_time - (row[8]+row[7]+row[6])
        return results

    def class_report(self, results):
        for row in results[1:]:
            if row[7]+row[8] == 0:
                row[3] = round(0, 2)
                row[4] = round(0, 2)
            else:
                row[3] = round(row[8] / (row[7]+row[8]), 2)
                row[4] = get_accuracy_recognition(row[6], row[7], row[8], row[9])
            row[10] = get_precission(row[8], row[6])
            row[11] = get_recall(row[8], row[7])
            row[5] = get_f_score(row[10], row[11])
        return results

    def remove_unknown(self):
        if 'UNK' in self.json_tp['tracks']:
            del self.json_tp['tracks']['UNK']


    def apply_windows_range(self, windows_range):
        zero_time = get_timestamp('00:00:00')
        for location_gt in self.json_gt['tracks'].values():
            if location_gt:
                for hits_gt in list(location_gt):
                    start_time = get_timestamp(hits_gt['start'])-windows_range
                    if start_time >= zero_time:
                        hits_gt['start'] = get_strptime(start_time)
                    video_time = self.get_time_video(hits_gt['location'])
                    final_video_time = get_timestamp(get_delta_time(video_time))
                    end_time = get_timestamp(hits_gt['end'])+windows_range
                    if end_time <= final_video_time:
                        hits_gt['end'] = get_strptime(end_time)

    def execute_tp_in_gt(self,results):
        for user in self.json_tp['tracks'].keys():
            locations_tp = get_sorted_metrics(self.json_tp, user)
            self.user = user
            if self.user in self.json_gt['tracks'].keys():
                locations_gt = get_sorted_metrics(self.json_gt, self.user)
                for location_gt, value in groupby(locations_gt, location_func):
                    self.location_gt = location_gt
                    row = self.row_with_match(list(value), locations_tp)
                    results.append(row)
            else:
                for location_gt, value in groupby(locations_tp, location_func):
                    self.location_gt = location_gt
                    row = self.get_row_without_match_user(list(value))
                    results.append(row)
        return results

    def tp_left_list(self,results):
        for user, location_tp in self.json_tp['tracks'].items():
            if location_tp:
                for hits_tp in list(location_tp):
                    idx_list = search_row_metrics(user, hits_tp['location'], results)
                    if idx_list:
                        self.json_tp['tracks'][user].remove(hits_tp)
                        results[idx_list][6] += get_diff_time(hits_tp['end'], hits_tp['start'])
                    else:
                        fp = 0.0
                        fn = 0.0
                        tp = 0.0
                        time_expect = 0.0
                        self.json_tp['tracks'][user].remove(hits_tp)
                        fp = get_diff_time(hits_tp['end'], hits_tp['start'])
                        row = [user, self.set_tp, hits_tp['location'],
                                -1, -1, -1, fp, fn, tp, time_expect, -1, -1]
                        results.append(row)
        return results

    def gt_left_list(self,results):
        for user, location_gt in self.json_gt['tracks'].items():
            if location_gt:
                for hits_gt in list(location_gt):
                    idx_list = search_row_metrics(user, hits_gt['location'], results)
                    if idx_list:
                        self.json_gt['tracks'][user].remove(hits_gt)
                        if user not in list(self.json_tp['tracks'].keys()):
                            results[idx_list][7] += get_diff_time(hits_gt['end'], hits_gt['start'])
                            results[idx_list][9] += get_diff_time(hits_gt['end'], hits_gt['start'])
                        else:                                
                            results[idx_list][9] -= results[idx_list][8]
                            break
                    else:
                        fp = 0.0
                        fn = 0.0
                        tp = 0.0
                        self.json_gt['tracks'][user].remove(hits_gt)
                        fn = get_diff_time(hits_gt['end'], hits_gt['start'])
                        time_expect = get_diff_time(hits_gt['end'], hits_gt['start'])
                        row = [user, self.set_tp, hits_gt['location'],
                                -1, -1, -1, fp, fn, tp, time_expect, -1, -1]
                        results.append(row)
        return results

    def save_csv(self,output,results):
        with open(output, mode='w') as file:
            results_writer = csv.writer(file)
            for rows in results:
                results_writer.writerow(rows)


    def match_gt_tp(self,output,windows_range):
        results = [get_header()]

        self.apply_windows_range(windows_range)
        self.remove_unknown()

        results = self.execute_tp_in_gt(results)

        results = self.tp_left_list(results)
        results = self.gt_left_list(results)

        results = self.calcule_tn(results)
        results = self.class_report(results)

        self.save_csv(output,results)



    def metrics(self, output, windows_range=0):
        set_gt = int(self.json_gt['set'])
        self.set_tp = int(self.json_tp['set'])
        if set_gt == self.set_tp:
            self.match_gt_tp(output,windows_range)