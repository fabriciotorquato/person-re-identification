import argparse
import sys
sys.path.append('../../libs/facenet/src')
sys.path.append('../../src')

from glob import glob
import os
import subprocess

from align.align_dataset_mtcnn import main
from align.align_dataset_mtcnn import parse_arguments


def process(dataset,video_path_base):

    for idx in range(1,12):
        video_path = '{}/set_{}'.format(video_path_base,idx)
        subprocess.call(['sh','convert_videos.sh',video_path])
        os.makedirs(dataset, exist_ok=True)
        args = [video_path,
                dataset,
                '--image_size', '160',
                '--margin', '32',
                '--gpu_memory_fraction', '.25',
                '--random_order']
        main(parse_arguments(args))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', '-d', default='/home',
                    help='Path to saved dataset (default /home).')
    ap.add_argument('--video_path', '-v', default='/home',
                    help='Path to directory of video (default /home).')
    args = vars(ap.parse_args())
    dataset=args.dataset
    video_path_base=args.video_path
    process(dataset,video_path_base)