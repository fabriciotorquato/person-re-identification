videos_path=$1

for f in `find . -name "$videos_path/*.avi"`
do
	dir=${f%.*}
    mkdir -p "$videos_path/$dir"
    ffmpeg -i $f -vf fps=5 $videos_path/$dir/%d.jpg
done
