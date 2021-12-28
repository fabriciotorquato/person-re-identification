

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