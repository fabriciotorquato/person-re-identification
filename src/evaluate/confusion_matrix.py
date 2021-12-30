
import json
import csv
import random
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from itertools import groupby
from eval_utils import *
from date_utils import *
from math_utils import *


class ConfusionMatrix:
    def __init__(self, ground_truth, tracking_predict, video_time, predict_ranges=0):

        with open(ground_truth, 'r') as file:
            self.json_gt = json.loads(file.read())

        with open(tracking_predict, 'r') as file:
            self.json_tp = json.loads(file.read())

        with open(video_time, 'r') as file:
            self.video_time_json = json.loads(file.read())

        self.predict_ranges = predict_ranges
        self.fps_video = self.get_fps_video()

    def time_video(self, location):
        return self.video_time_json[str(self.set_tp)][location]

    def get_fps_video(self):
        return self.video_time_json['fps']

    def report_score(self, cm, users_tp, acc_detection):
        results = [get_header()]
        for location, dict_cm in cm.items():
            location_results = []
            total_acc, match_acc = 0, 0
            for user, row in dict_cm.items():
                precission_total = sum([el for el in row.values()])

                if dict_cm[user][user] != 0:
                    precission = dict_cm[user][user] / precission_total
                else:
                    precission = 0

                recall_total = sum([el[user] for el in dict_cm.values()])
                if dict_cm[user][user] != 0:
                    recall = dict_cm[user][user] / recall_total
                else:
                    recall = 0

                f_score = get_f_score(precission, recall)

                total_acc += precission_total
                match_acc += row[user]
                metrics_row = [user,
                               self.set_tp,
                               location,
                               acc_detection,
                               0.0,
                               f_score,
                               round(precission, 2),
                               round(recall, 2)]
                if location in users_tp and user in users_tp[location]:
                    location_results.append(metrics_row)
            acc = match_acc / total_acc
            location_results = np.array(location_results)
            for row in location_results:
                if float(row[5]) > 0:
                    row[4] = round(acc, 2)
            results.extend(location_results)

        return results

    def remove_unknown(self):
        if 'UNK' in self.json_tp['tracks']:
            del self.json_tp['tracks']['UNK']

    def save_csv(self, output, results):
        with open(output, mode='w') as file:
            results_writer = csv.writer(file)
            for rows in results:
                results_writer.writerow(rows)

    def gt_detection(self, list_hits, gt_recog_key):
        for hits in list(list_hits):
            frame_start = int(
                diff_time(hits['start'], '0:00:00')) * self.fps_video
            frame_end = int(diff_time(hits['end'], '0:00:00')) * self.fps_video
            gt_recog_key[frame_start:frame_end] += 1
        return gt_recog_key

    def tp_detection(self, list_hits, tp_recog_key):
        for hits in list(list_hits):
            tp_recog_key[hits['frame_start']:hits['frame_end']] += 1
        return tp_recog_key

    def new_row_frame(self, location):
        return np.array([0] * self.time_video(location) * self.fps_video)

    def acc_detection(self):
        gt_dict = {}
        tp_dict = {}
        acc = []
        acc_total = []

        for user in self.json_gt['tracks'].keys():
            locs_gt = sorted_metrics(self.json_gt, user)
            for loc_gt, value in groupby(locs_gt, location_func):
                key = loc_gt
                if key not in gt_dict:
                    gt_dict[key] = self.new_row_frame(loc_gt)
                gt_dict[key] = self.gt_detection(value, gt_dict[key])

        for user in self.json_tp['tracks'].keys():
            locs_tp = sorted_metrics(self.json_tp, user)
            for loc_tp, value in groupby(locs_tp, location_func):
                key = loc_tp
                if key in gt_dict:
                    if key not in tp_dict:
                        tp_dict[key] = self.new_row_frame(key)
                    tp_dict[key] = self.tp_detection(value, tp_dict[key])

        for key, value in gt_dict.items():
            if key in tp_dict:
                mask_idx = (gt_dict[key]!=0)
                a = np.array(tp_dict[key][mask_idx], dtype=float)
                b = np.array(gt_dict[key][mask_idx], dtype=float)
                mask_idx = (np.where(a > b))
                a[mask_idx] = b[mask_idx]
                acc.extend(a)
                acc_total.extend(b)
        return round(np.average(np.array(np.array(acc)/np.array(acc_total))), 2)

    def get_class_name(self):
        a = list(self.json_gt['tracks'].keys())
        b = list(self.json_tp['tracks'].keys())
        return sorted(list(set(np.concatenate((a, b), axis=0))))

    def cm_multiclass(self):
        cm = {}
        users_gt = {}
        users_tp = {}
        class_name = self.get_class_name()
        for user in self.json_gt['tracks'].keys():
            locations_gt = sorted_metrics(self.json_gt, user)
            for loc_gt, value_gt in groupby(locations_gt, location_func):
                if loc_gt not in users_gt:
                    users_gt[loc_gt] = {}
                if user not in users_gt[loc_gt]:
                    users_gt[loc_gt][user] = self.new_row_frame(loc_gt)
                users_gt[loc_gt][user] = self.gt_detection(
                    value_gt, users_gt[loc_gt][user])

        for user in self.json_tp['tracks'].keys():
            locations_tp = sorted_metrics(self.json_tp, user)
            for loc_tp, value_tp in groupby(locations_tp, location_func):
                if loc_tp not in users_tp:
                    users_tp[loc_tp] = {}
                if user not in users_tp[loc_tp]:
                    users_tp[loc_tp][user] = self.new_row_frame(loc_tp)
                users_tp[loc_tp][user] = self.gt_detection(
                    value_tp, users_tp[loc_tp][user])

        for loc_gt, frames_id_gt in users_gt.items():
            if loc_gt in users_tp:
                if loc_gt not in cm:
                    cm[loc_gt] = {
                        il: {el: 0 for el in class_name} for il in class_name}
                for user_gt, frames_gt in frames_id_gt.items():
                    for idx, frame in enumerate(frames_gt):
                        if frame == 1:
                            if user_gt in users_tp[loc_gt] and users_tp[loc_gt][user_gt][idx] == 1:
                                cm[loc_gt][user_gt][user_gt] += 1
                                users_tp[loc_gt][user_gt][idx] = -1
                                users_gt[loc_gt][user_gt][idx] = -1

        for loc_gt, frames_id_gt in users_gt.items():
            if loc_gt in users_tp:
                if loc_gt not in cm:
                    cm[loc_gt] = {
                        il: {el: 0 for el in class_name} for il in class_name}
                for user_gt, frames_gt in frames_id_gt.items():
                    for idx, frame in enumerate(frames_gt):
                        if frame == 1:
                            list_users = []
                            for user_tp, frames_id_tp in users_tp[loc_gt].items():
                                if frames_id_tp[idx] == 1:
                                    list_users.append(user_tp)
                            if len(list_users):
                                user_tp = random.choice(list_users)
                                cm[loc_gt][user_tp][user_gt] += 1
                                users_tp[loc_gt][user_tp][idx] = -1
                                users_gt[loc_gt][user_gt][idx] = -1

        self.plot_heatmap(cm)
        return cm, users_tp

    def plot_heatmap(self, cm):
        for location,data in cm.items():
            prob_matrix = pd.DataFrame(data).T
            ax = plt.axes()
            sb.heatmap(prob_matrix, annot=True, fmt="d")
            ax.set_title(location)
            plt.show()

    def metrics(self, output):
        set_gt = int(self.json_gt['set'])
        self.set_tp = int(self.json_tp['set'])
        if set_gt == self.set_tp:
            self.remove_unknown()
            acc_detection = self.acc_detection()
            cm, users_tp = self.cm_multiclass()
            results = self.report_score(cm, users_tp, acc_detection)
            self.save_csv(output, results)
