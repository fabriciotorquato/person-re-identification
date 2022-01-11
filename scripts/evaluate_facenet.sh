# conda activate tcc-25

value_set=$1

cd ../src/evaluate &&
python metrics.py \
-p ../../data/output/set_$value_set/tracking_predict_db_facenet.json \
-g ../../data/ground_truth/gt_set_$value_set.json \
-v ../../data/ground_truth/video_time.json \
-o ../../output/set_$value_set/facenet

