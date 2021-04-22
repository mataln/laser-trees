import utils
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class TreeSpeciesDataset(Dataset):
    """Dataset for tree species classification from
    point cloud depth projection images"""

    def __init__(self, metadata_file, data_dir, image_dim=128, camera_fov_deg=90, f=1, camera_dist=1.4):
        """
        Args:
            metadata_file (string): Path to the metadata file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_frame = pd.read_csv(metadata_file)
        self.species = list(df.groupby('sp')['id'].nunique().keys())
    
        filenames = filter(lambda t:t.endswith('.txt'), os.listdir('data_dir'))
        no_files = len(list(filenames))
        
        #6 views, 1 channel
        self.depth_images = torch.zeros(size=(no_files, 6, 1, image_dim, image_dim))
        self.labels = torch.zeros(size=no_files)
        
        #Build the projections here rather than at access time
        for i, file in enumerate(tqdm(filenames, total=no_files)):
            cloud = utils.pc_from_txt(file)
            cloud = utils.center_and_scale(cloud)
            
            self.depth_images[i] = utils.get_depth_images_from_cloud(cloud, 
                                                                   image_dim=image_dim, 
                                                                   camera_fov_deg=camera_fov_deg, 
                                                                   f=f, 
                                                                   camera_dist=camera_dist
                                                                   )

            self.labels[i] = #Get index of labels from species
            

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample