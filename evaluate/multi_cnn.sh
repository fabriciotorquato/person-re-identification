# conda activate tcc-25
hash=$(date +%s)
python evaluate_realtime.py -x $hash -v ../data/wisenet_dataset/videos -s 1 -i 5 -t show -d cnn

python resume_evaluate.py -x $hash -v ../data/wisenet_dataset/videos -s 1