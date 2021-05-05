import utils

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

import pandas as pd
import os


class TreeSpeciesDataset(Dataset):
    """Dataset for tree species classification from
    point cloud depth projection images"""

    def __init__(self, data_dir, metadata_file):
        """
        Args:
            metadata_file (string): Path to the metadata file.
            root_dir (string): Directory with all the images.
        """
        self.meta_frame = pd.read_csv(metadata_file, keep_default_na=False)
        self.species = list(self.meta_frame.groupby('sp')['id'].nunique().keys())
        
        self.counts = self.meta_frame['sp'].value_counts() 
        
        self.data_dir = data_dir
        
        self.depth_images = None
        self.labels = None
        
        self.image_dim = None
        self.camera_fov_deg = None
        self.f = None
        self.camera_dist = None
        self.soft_min_k = None
        
        return
            
    def build_depth_images(self, image_dim=128, camera_fov_deg=90, f=1, camera_dist=1.4, soft_min_k=50):  
        self.image_dim = image_dim
        self.camera_fov_deg = camera_fov_deg
        self.f = f
        self.soft_min_k = soft_min_k
        self.camera_dist = camera_dist
        
        filenames = list(filter(lambda t:t.endswith('.txt'), os.listdir(self.data_dir)))
        no_files = len(filenames)
        
        #6 views, 1 channel
        self.depth_images = torch.zeros(size=(no_files, 6, 1, image_dim, image_dim))
        self.labels = torch.zeros(no_files)
        
        #Build the projections here rather than at access time
        for i, file in tqdm(enumerate(filenames), total=no_files):
            cloud = utils.pc_from_txt(self.data_dir + file)
            cloud = utils.center_and_scale(cloud)
            
            self.depth_images[i] =  torch.unsqueeze(
                                    utils.get_depth_images_from_cloud(cloud, 
                                                                   image_dim=image_dim, 
                                                                   camera_fov_deg=camera_fov_deg, 
                                                                   f=f, 
                                                                   camera_dist=camera_dist
                                                                   )
                                    , 1)
            
            meta_entry = self.meta_frame[self.meta_frame.id==file[:-4]]

            self.labels[i] = self.species.index(meta_entry.sp.values[0])#Get index of labels from species
            
        self.labels = self.labels.long()
        
        return
    
    def remove_species(self, species):
        if self.depth_images == None:
            print("Only run remove_species after building depth images")
            return
        
        idx = [] #Indices to keep
        
        for i in range(len(self.labels)): #Remove entries in images and labels for that species
            if not(self.species[int(self.labels[i])] == species):
                #self.depth_images = torch.cat([self.depth_images[:i], self.depth_images[i+1:]]) #Remove examples from depth images
                #self.labels = torch.cat([self.labels[:i], self.labels[i+1:]]) #And labels
                idx.append(i)
                
        self.depth_images = self.depth_images[idx]
        self.labels = self.labels[idx]
                
        old_species = self.species.copy() 
        self.species.pop(self.species.index(species)) #Pop from species list
            
        species_map = [self.species.index(species) if species in self.species else None for species in old_species]     
            
        #for j, species in enumerate(old_species): #Species map for old->new labels
        #    #species_map[old_label] = new_label
        #    if species in self.species: #If not the deleted species
        #        species_map[j] = self.species.index(species)
        

        for k in range(len(self.labels)): #Apply species map to relabel
            self.labels[k] = torch.tensor(species_map[int(self.labels[k])])
            
        self.counts = self.counts.drop(species)
        
        return

            
    def __len__(self):
        if self.labels == None:
            return len(self.meta_frame)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_images = self.depth_images[idx]
        labels = self.labels[idx]
        
        sample = {'depth_images': depth_images, 'labels': labels}

        return sample