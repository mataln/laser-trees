# Tree Species Identification From Terrestrial Laser Scanning Data
Repository for my TLS MRes project
(Readmes to follow during w/b 12/7/21)

## notebooks
Notebooks to run various experiments, detailed in folder readme. Includes notebook to parse data from .txt lists of points per tree to PyTorch dataset

## sh
Shell files to pull relevant parts of other repos used in this project. Run as 
`sh sh/dl-simpleview.sh`
from either the base directory or from within /sh to download the code for the network architecture

## utils
.py files containing various useful functions, including training, testing & camera projections, as well as a custom class to store the data as a pytorch dataset


laser-trees.yml - YAML file containing environment specification to rerun this code


## Data used

Data gathered by H.J.F Owen (2021) - not openly available for download at the moment

https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2745.13670
