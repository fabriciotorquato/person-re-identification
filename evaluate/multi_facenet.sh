#conda activate tcc-17
hash=$(date +%s)
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/videos -s 1 -i 5 -t show -d facenet &
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/videos -s 1 -i 1 -t show -d facenet 

python resume_evaluate.py -x $hash -v ../data/wisenet_dataset/videos -s 1