import argparse
import os
import sys
from confusion_matrix import ConfusionMatrix

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground_truth', '-g', type=str,
                    help='Path to ground truth file.')
    ap.add_argument('--tracking_predict', '-p', type=str,
                    help='Path to tracking predict file.')
    ap.add_argument('--video_time', '-v', type=str,
                    help='Path to video time file.')
    ap.add_argument('--ouput', '-o', type=str,
                    help='Path to saved results file metrics.')
    argv = vars(ap.parse_args(argv))

    os.makedirs(argv['ouput'], exist_ok=True)

    print('Generating metrics file...')


    output = '{}/metrics.csv'.format(argv['ouput'])
    cm = ConfusionMatrix(argv['ground_truth'], argv['tracking_predict'], argv['video_time'])
    cm.metrics(output)


if __name__ == '__main__':
    main(sys.argv[1:])
