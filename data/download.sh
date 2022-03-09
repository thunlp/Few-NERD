data_dir=`dirname $0`
if [ $# == 0 ]; then
bash $data_dir/download.sh inter
bash $data_dir/download.sh intra
bash $data_dir/download.sh supervised
bash $data_dir/download.sh episode-data
elif [ $1 == 'supervised' ]; then
wget -O $data_dir/supervised.zip https://cloud.tsinghua.edu.cn/f/09265750ae6340429827/?dl=1
unzip -o -d $data_dir/ $data_dir/supervised.zip
rm -rf $data_dir/supervised.zip
elif [ $1 == 'inter' ]; then
wget -O $data_dir/inter.zip https://cloud.tsinghua.edu.cn/f/165693d5e68b43558f9b/?dl=1
unzip -o -d $data_dir/ $data_dir/inter.zip
rm -rf $data_dir/inter.zip
elif [ $1 == 'intra' ]; then
wget -O $data_dir/intra.zip https://cloud.tsinghua.edu.cn/f/a0d3efdebddd4412b07c/?dl=1
unzip -o -d $data_dir/ $data_dir/intra.zip
rm -rf $data_dir/intra.zip
elif [ $1 == 'episode-data' ]; then
wget -O $data_dir/episode-data.zip https://cloud.tsinghua.edu.cn/f/0e38bd108d7b49808cc4/?dl=1
unzip -o -d $data_dir/ $data_dir/episode-data.zip
rm -rf $data_dir/episode-data.zip
fi

