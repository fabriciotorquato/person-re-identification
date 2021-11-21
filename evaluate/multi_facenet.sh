#conda activate tcc-17
hash=$(date +%s)
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/video -s 5 -i 1 -t show -d facenet &
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/video -s 5 -i 3 -t show -d facenet 

python resume_evaluate.py -x $hash -v ../data/wisenet_dataset/video -s 11