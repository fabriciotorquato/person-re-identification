# conda activate tcc-25
hash=$(date +%s)
set_video=$1
SECONDS=0 

cd ../src/re_indentification && python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 1 -d mobilenet &
cd ../src/re_indentification && python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 2 -d mobilenet &
cd ../src/re_indentification && python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 3 -d mobilenet &
cd ../src/re_indentification && python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 4 -d mobilenet &
cd ../src/re_indentification && python indetification.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -i 5 -d mobilenet 

echo "Time to run model for evaluate videos camera: $SECONDS s"

cd ../src/re_indentification && python report.py -x $hash -v ../../data/wisenet_dataset/video -s $set_video -d mobilenet