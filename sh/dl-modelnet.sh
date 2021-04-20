#!/bin/bash

#Exception for if scripts are called from within the /sh/ directory
#This will break if there's a space in the path
if [$(basename $PWD) = "sh" ]
then
	cd ..
fi

mkdir -p data
cd data
mkdir -p modelnet40
cd modelnet40

#Download modelnet 40 if not already downloaded
if [ ! -d "modelnet40_ply_hdf5_2048" ]
then
	wget -O modelnet.zip --no-check-certificate "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
	unzip modelnet.zip
	rm modelnet.zip
fi

MODEL_DIR=$PWD/modelnet40_ply_hdf5_2048

cd ..
cd ..

echo $PWD

#Clone PAConv
if [ -d "/PAConv" ]
then
	cd PAConv
else
	sh sh/dl-paconv.sh
	cd PAConv
fi

#Create symlink
cd obj_cls
mkdir -p data
ln -s $MODEL_DIR data
