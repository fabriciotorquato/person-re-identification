import argparse
import datetime
import json
import os
from database import Tracker, getConnection, select_all


def resume(set, results,output):
    json_results = {"set": set, "tracks": {}}
    for row in results:
        row_tracker = Tracker(*row)
        if row_tracker.predict_name not in json_results["tracks"].keys():
            json_results["tracks"][row_tracker.predict_name] = []

        start = str(datetime.timedelta(
            milliseconds=row_tracker.start)).split('.')[0]
        end = str(datetime.timedelta(
            milliseconds=row_tracker.end)).split('.')[0]
        json_results["tracks"][row_tracker.predict_name].append(
            {"location": row_tracker.location, "start": start, "end": end})

    with open(output, 'w') as f:
        json.dump(json_results, f)
            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", "-x", default=0,
                    help="Number for sync many videos (default 0).")
    ap.add_argument("--video", "-v", default="/home",
                    help="Path to video file (default /home).")
    ap.add_argument("--set", "-s", default=0,
                    help="Set to video file (default 0).")
    args = vars(ap.parse_args())

    output = '{}/../output/set_{}'.format(args["video"], args["set"])
    os.makedirs(output, exist_ok=True)
    script_dir = os.path.dirname(__file__)
    output = os.path.join(
        script_dir, "{}/tracking_predict_db.json".format(output))

    con = getConnection()
    results = select_all(con, args["hash"])
    resume(args["set"],results,output)


if __name__ == '__main__':
    main()
