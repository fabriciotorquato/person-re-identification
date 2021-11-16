# conda activate tcc-25
hash=$(date +%s)
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/video -s 11 -i 5 -t show -d cnn

python resume_evaluate.py -x $hash -v ../data/wisenet_dataset/video -s 11