import sys
sys.path.append("../facenetLib")
sys.path.append("..")

from glob import glob
import os
import subprocess

from facenetLib.align.align_dataset_mtcnn import main
from facenetLib.align.align_dataset_mtcnn import parse_arguments


def process():
    dataset = '../data/wisenet_dataset/videos_frames'

    for idx in range(2,11):
        video_path = '../data/wisenet_dataset/video/set_{}'.format(idx)
        subprocess.call(['sh','convert-videos.sh',video_path])
        os.makedirs(dataset, exist_ok=True)
        args = [video_path,
                dataset,
                '--image_size', '160',
                '--margin', '32',
                '--gpu_memory_fraction', '.25',
                '--random_order']
        main(parse_arguments(args))


if __name__ == '__main__':
    process()