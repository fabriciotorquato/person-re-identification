hash=$(date +%s)
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/videos -s 1 -i 5 -t show &
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/videos -s 1 -i 1 -t show 

python resume_evaluate.py -x $hash -v ../data/wisenet_dataset/videos -s 1