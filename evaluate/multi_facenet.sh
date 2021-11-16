#conda activate tcc-17
hash=$(date +%s)
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/video -s 11 -i 5 -t show -d facenet &
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/video -s 11 -i 4 -t show -d facenet 

python resume_evaluate.py -x $hash -v ../data/wisenet_dataset/video -s 11