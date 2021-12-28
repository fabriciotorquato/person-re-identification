# conda activate tcc-25

for VALUE_SET in 1 2 10
do
  python metrics.py \
  -p ../../data/output/set_$VALUE_SET/tracking_predict_db_facenet.json \
  -g ../../data/ground_truth/gt_set_$VALUE_SET.json \
  -v ../../data/ground_truth/video_time.json \
  -o ../../output/set_$VALUE_SET/facenet
done
