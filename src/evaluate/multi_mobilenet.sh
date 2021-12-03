# conda activate tcc-25
hash=$(date +%s)
SECONDS=0 
python evaluate_realtime.py -x $hash -v ../../data/wisenet_dataset/video -s 10 -i 4 -t show -d mobilenet &
python evaluate_realtime.py -x $hash -v ../../data/wisenet_dataset/video -s 10 -i 5 -t show -d mobilenet

echo "Time to run model for evaluate videos camera: $SECONDS s"

python resume_evaluate.py -x $hash -v ../../data/wisenet_dataset/video -s 10 -d mobilenet