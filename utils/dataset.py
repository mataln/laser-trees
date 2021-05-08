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
    
    
class TreeSpeciesPointDataset(Dataset):
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
        
        self.point_clouds = []
        self.labels = None
        
        self.image_dim = None
        self.camera_fov_deg = None
        self.f = None
        self.camera_dist = None
        self.soft_min_k = None
        
        filenames = list(filter(lambda t:t.endswith('.txt'), os.listdir(self.data_dir)))
        no_files = len(filenames)
        
        self.labels = torch.zeros(no_files)
        
        for i, file in tqdm(enumerate(filenames), total=no_files):
            cloud = utils.pc_from_txt(self.data_dir + file)
            cloud = utils.center_and_scale(cloud)
            
            self.point_clouds.append(torch.from_numpy(cloud))
            
            meta_entry = self.meta_frame[self.meta_frame.id==file[:-4]]
            self.labels[i] = self.species.index(meta_entry.sp.values[0])
            
        self.labels = self.labels.long()
        
        return
    
    
    def get_depth_image(self, i):
        return torch.unsqueeze(
               utils.get_depth_images_from_cloud(points=self.point_clouds[i], 
                                                 image_dim=self.image_dim, 
                                                 camera_fov_deg=self.camera_fov_deg, 
                                                 f=self.f, 
                                                 camera_dist=self.camera_dist
                                                 )
                                    , 1)
    
    
    def remove_species(self, species):
        
        idx = [] #Indices to keep
        
        for i in range(len(self.labels)): #Remove entries in images and labels for that species
            if not(self.species[int(self.labels[i])] == species):
                idx.append(i)
                
        self.point_clouds = [self.point_clouds[i] for i in idx]
        self.labels = self.labels[idx]
                
        old_species = self.species.copy() 
        self.species.pop(self.species.index(species)) #Pop from species list
            
        species_map = [self.species.index(species) if species in self.species else None for species in old_species]     
            

        for k in range(len(self.labels)): #Apply species map to relabel
            self.labels[k] = torch.tensor(species_map[int(self.labels[k])])
            
        self.counts = self.counts.drop(species)
        
        return
    
    def set_params(self, 
                   image_dim = None,
                   camera_fov_deg = None,
                   f = None,
                   camera_dist = None,
                   soft_min_k = None):
     
        if image_dim:    
            self.image_dim = image_dim        
        if camera_fov_deg:
            self.camera_fov_deg = camera_fov_deg 
        if f:
            self.f = f      
        if camera_dist:
            self.camera_dist = camera_dist      
        if soft_min_k:
            self.soft_min_k = soft_min_k
            
        return
            
    def __len__(self):
        if self.labels == None:
            return len(self.meta_frame)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            num_trees = len(idx)
        elif type(idx) == int:
            num_trees = 1
        

        depth_images = torch.zeros(size=(num_trees, 6, 1, self.image_dim, self.image_dim))
        
        if type(idx) == list:
            for i in range(len(idx)):
                depth_images[i] = self.get_depth_image(int(idx[i]))
        elif type(idx) == int:
            depth_images = self.get_depth_image(idx)
        
        labels = self.labels[idx]
        
        sample = {'depth_images': depth_images, 'labels': labels}

        return sample    