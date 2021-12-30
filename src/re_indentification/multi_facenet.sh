#conda activate tcc-17
hash=$(date +%s)
set_video=2
SECONDS=0 
python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 1 -d facenet &
python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 2 -d facenet &
python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 3 -d facenet &
python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 4 -d facenet &
python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 5 -d facenet 

echo "Time to run model for evaluate videos camera: $SECONDS s"

python report.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -d facenet