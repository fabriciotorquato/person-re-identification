def get_f_score(precission, recall):
    try:
        return round((2*precission*recall) / (precission+recall), 2)
    except ZeroDivisionError:
        return 0.0

