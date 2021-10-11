import os
import sys
import copy
pdir = os.path.dirname(os.getcwd())
sys.path.append(pdir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import random

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR


import utils
from simpleview_pytorch import SimpleView

from torch.utils.data.dataset import Dataset

def predict(dataset, val_indices, model, params):
    '''
    Given a dataset, test indices (n<N indices)
    and a trained model, returns either:
    
    logits : nxC for C classes
    label predictions: n
    '''
    #val_data = torch.load
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for specie in list(set(dataset.species) - set(params["species"])):
        print("Removing: {}".format(specie))
        dataset.remove_species(specie)

            
    dataset.set_params(transforms=['none'])
    
    val_sampler = SubsetRandomSampler(val_indices)
    
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                    sampler=val_sampler)
    

    model = model.to(device=device)
    model.eval()
    
    with torch.no_grad():
        
        all_logits = torch.tensor([]).to(device)
        all_labels = torch.tensor([]).to(device)
        all_predictions = torch.tensor([]).to(device)
        
        for data in validation_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)
                
                all_logits = torch.cat((all_logits, scores))
                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                
        return all_logits, all_labels, all_predictions
    
def predict_from_dirs(dataset_dir, model_dir):#Load data
    val_data = torch.load(dataset_dir)

    params = {
        "dataset_type":type(val_data),
        "batch_size":128,
        "validation_split":.2,
        "shuffle_dataset":True,
        "random_seed":0,
        "learning_rate":[0.0005, 100, 0.5],  #[init, step_size, gamma] for scheduler
        "momentum":0.9, #Only used for sgd
        "epochs":300,
        "loss_fn":"smooth-loss",
        "optimizer":"adam",
        "voting":"None",
        "train_sampler":"balanced",

        "model":"SimpleView",

        "image_dim":256,
        "camera_fov_deg":90,
        "f":1,
        "camera_dist":1.4,
        "depth_averaging":"min",
        "soft_min_k":50,
        "num_views":6,

        "transforms":['none'],
        "min_rotation":0,
        "max_rotation":2*np.pi,
        "min_translation":0,
        "max_translation":0.5,
        "jitter_std":3e-5, 

        "species":["QUEFAG", "PINNIG", "QUEILE", "PINSYL", "PINPIN"],
        "data_resolution":"2.5cm"
    }

    val_data.set_params(image_dim = params["image_dim"],
                     camera_fov_deg = params["camera_fov_deg"],
                     f = params["f"],
                     camera_dist = params["camera_dist"],
                     soft_min_k = params["soft_min_k"],
                     transforms = params["transforms"],
                     min_rotation = params["min_rotation"],
                     max_rotation = params["max_rotation"],
                     min_translation = params["min_translation"],
                     max_translation = params["max_translation"],
                     jitter_std = params["jitter_std"]
                     )

    #Load model
    model = SimpleView(
            num_views=params["num_views"],
            num_classes=len(params["species"])
            )

    model.load_state_dict(torch.load(model_dir))

    #Load validation indices
    val_indices = list(np.load(f'indices/new_test_indices_{params["random_seed"]}.npy'))
    val_indices = [int(vi) for vi in val_indices]

    logits, labels, predictions = utils.predict(dataset=val_data, val_indices=val_indices, model=model, params=params)
    
    return logits, labels, predictions, val_data.species
    
    
def get_angles(start, end, n_angles):
    '''
    Returns n_angles evenly space angles in range
    start-end (RADIANS)
    '''
    gap = (end-start)/n_angles
    return np.linspace(start, end-gap, n_angles)

def rotation_vote(dataset, model, num_rotations, val_indices, params, max_rotation = 2*np.pi):
    '''
    Given a dataset, model, and number of rotations, performs rotation voting.
    Returns soft_predictions, hard_predictions, labels
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    angles = get_angles(0, max_rotation, num_rotations)
    init_random_state = torch.random.get_rng_state()
    
    all_logits = []
    all_predictions = []
    
    for i, theta in enumerate(angles):
        if i == 0:
            delta_theta = angles[0]
        else:
            delta_theta = angles[i] - angles[i-1]
        
        delta_theta = torch.tensor(delta_theta)
        Rz = torch.tensor([
        [torch.cos(delta_theta), -torch.sin(delta_theta), 0],
        [torch.sin(delta_theta),  torch.cos(delta_theta), 0],
        [0               ,                 0, 1],
        ]).double()
        
        for idx in range(len(dataset.point_clouds)):
            dataset.point_clouds[idx] = torch.matmul(dataset.point_clouds[idx], Rz.t())
            
        torch.random.set_rng_state(init_random_state) #Keeps dataset sampler in the same order
        logits, labels, predictions = predict(dataset=dataset, val_indices=val_indices, model=model, params=params)
        
        all_logits.append(logits)
        all_predictions.append(predictions)
        
    all_logits = torch.stack(all_logits, 0)
    all_predictions = torch.stack(all_predictions, 0)
    
    total_logits = torch.sum(all_logits, 0)
    _, soft_predictions = torch.max(total_logits, 1)
            
    hard_predictions, _ = torch.mode(all_predictions, 0)
    
    
    return soft_predictions, hard_predictions, labels

def rot_vot_from_dirs(dataset_dir, model_dir, num_rotations=1, max_rotation=2*np.pi):
    val_data = torch.load(dataset_dir)

    params = {
        "dataset_type":type(val_data),
        "batch_size":128,
        "validation_split":.2,
        "shuffle_dataset":True,
        "random_seed":0,
        "learning_rate":[0.0005, 100, 0.5],  #[init, step_size, gamma] for scheduler
        "momentum":0.9, #Only used for sgd
        "epochs":300,
        "loss_fn":"smooth-loss",
        "optimizer":"adam",
        "voting":"None",
        "train_sampler":"balanced",

        "model":"SimpleView",

        "image_dim":256,
        "camera_fov_deg":90,
        "f":1,
        "camera_dist":1.4,
        "depth_averaging":"min",
        "soft_min_k":50,
        "num_views":6,

        "transforms":['none'],
        "min_rotation":0,
        "max_rotation":2*np.pi,
        "min_translation":0,
        "max_translation":0.5,
        "jitter_std":3e-5, 

        "species":["QUEFAG", "PINNIG", "QUEILE", "PINSYL", "PINPIN"],
        "data_resolution":"2.5cm"
    }

    val_data.set_params(image_dim = params["image_dim"],
                     camera_fov_deg = params["camera_fov_deg"],
                     f = params["f"],
                     camera_dist = params["camera_dist"],
                     soft_min_k = params["soft_min_k"],
                     transforms = params["transforms"],
                     min_rotation = params["min_rotation"],
                     max_rotation = params["max_rotation"],
                     min_translation = params["min_translation"],
                     max_translation = params["max_translation"],
                     jitter_std = params["jitter_std"]
                     )

    #Load model
    model = SimpleView(
            num_views=params["num_views"],
            num_classes=len(params["species"])
            )

    model.load_state_dict(torch.load(model_dir))

    #Load validation indices
    val_indices = list(np.load(f'indices/val_indices_{params["random_seed"]}.npy'))
    val_indices = [int(vi) for vi in val_indices]

    soft_predictions, hard_predictions, labels = utils.rotation_vote(dataset=val_data, 
                                                                     model=model,  
                                                                     val_indices=val_indices, 
                                                                     params=params,
                                                                     num_rotations=num_rotations,
                                                                     max_rotation=max_rotation)
    
    return soft_predictions, hard_predictions, labels, val_data.species