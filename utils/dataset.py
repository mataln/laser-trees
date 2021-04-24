import utils

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

import pandas as pd
import os


class TreeSpeciesDataset(Dataset):
    """Dataset for tree species classification from
    point cloud depth projection images"""

    def __init__(self, data_dir, metadata_file, image_dim=128, camera_fov_deg=90, f=1, camera_dist=1.4):
        """
        Args:
            metadata_file (string): Path to the metadata file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_frame = pd.read_csv(metadata_file, keep_default_na=False)
        self.species = list(self.meta_frame.groupby('sp')['id'].nunique().keys())
        
        filenames = list(filter(lambda t:t.endswith('.txt'), os.listdir(data_dir)))
        no_files = len(filenames)
        
        #6 views, 1 channel
        self.depth_images = torch.zeros(size=(no_files, 6, 1, image_dim, image_dim))
        self.labels = torch.zeros(no_files)
        
        #Build the projections here rather than at access time
        for i, file in tqdm(enumerate(filenames), total=no_files):
            cloud = utils.pc_from_txt(data_dir + file)
            cloud = utils.center_and_scale(cloud)
            
            self.depth_images[i] =  torch.unsqueeze(
                                    torch.from_numpy(
                                    utils.get_depth_images_from_cloud(cloud, 
                                                                   image_dim=image_dim, 
                                                                   camera_fov_deg=camera_fov_deg, 
                                                                   f=f, 
                                                                   camera_dist=camera_dist
                                                                   )
                                    ), 1)
            
            meta_entry = self.meta_frame[self.meta_frame.id==file[:-4]]

            self.labels[i] = self.species.index(meta_entry.sp.values[0])#Get index of labels from species
            

    def __len__(self):
        return len(self.meta_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_images = self.depth_images[idx]
        labels = self.labels[idx]
        
        sample = {'depth_images': depth_images, 'labels': labels}

        return sample