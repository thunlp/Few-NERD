#!/bin/sh
echo $1
if [ $1 == 'supervised' ]; then
rm -rf supervised
mkdir supervised
wget -O supervised/test.txt https://cloud.tsinghua.edu.cn/f/4666d28af98a4e63afb5/?dl=1
wget -O supervised/train.txt https://cloud.tsinghua.edu.cn/f/6293b3d54f954ef8a0b1/?dl=1
wget -O supervised/dev.txt https://cloud.tsinghua.edu.cn/f/ae245e131e5a44609617/?dl=1
elif [ $1 == 'inter' ]; then
rm -rf inter
mkdir inter
wget -O inter/test.txt https://cloud.tsinghua.edu.cn/f/eeec65751e3148af945e/?dl=1
wget -O inter/train.txt https://cloud.tsinghua.edu.cn/f/45d55face2a14c098a13/?dl=1
wget -O inter/dev.txt https://cloud.tsinghua.edu.cn/f/9b529ee30f4544299bc2/?dl=1
elif [ $1 == 'intra' ]; then
rm -rf intra
mkdir intra
wget -O intra/test.txt https://cloud.tsinghua.edu.cn/f/9a1dc235abc746a6b444/?dl=1
wget -O intra/train.txt https://cloud.tsinghua.edu.cn/f/b169cfbeb90a48c1bf23/?dl=1
wget -O intra/dev.txt https://cloud.tsinghua.edu.cn/f/997dc82d29064e5ca8de/?dl=1
fi