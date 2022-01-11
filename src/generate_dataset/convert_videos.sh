videos_path=$1

for f in `find $videos_path -name *.avi`
do
	dir=${f%.*}
    rm -rf "$dir"
    mkdir -p "$dir"
    hash=$(date +%s)
    ffmpeg -i $f -vf fps=5 -vb 20M $dir/%d_$hash.jpg 
done
