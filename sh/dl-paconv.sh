#!/bin/bash

#Exception for if scripts are called from within the /sh/ directory
#This will break if there's a space in the path
if [$(basename $PWD) = "sh" ]
then
	cd ..
fi

#Clone PAConv repo
git clone https://github.com/CVMI-Lab/PAConv

cd PAConv

#Remove git stuff + non-classification bits
rm -r figure
rm -r part_seg
rm -r scene_seg
rm -r -f .git
rm LICENSE
rm README.md
rm .gitignore
