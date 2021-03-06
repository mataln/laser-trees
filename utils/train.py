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
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import confusion_matrix

import utils
from simpleview_pytorch import SimpleView

from torch.utils.data.dataset import Dataset


def train(data_dir, model_dir, params, wandb_project="laser-trees-bayes", init_wandb=True):
    #wandb.login()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name = data_dir

    trees_data = torch.load(dataset_name)
    val_data = torch.load(dataset_name)
    
    print(trees_data.counts)
    print('Species: ', trees_data.species)
    print('Labels: ', trees_data.labels)
    print('Total count: ', len(trees_data))



    if params["dataset_type"] == utils.dataset.TreeSpeciesPointDataset: #Change these by hand using point dataset
        trees_data.set_params(image_dim = params["image_dim"],
                             camera_fov_deg = params["camera_fov_deg"],
                             f = params["f"],
                             camera_dist = params["camera_dist"],
                             soft_min_k = params["soft_min_k"],
                             transforms = params["transforms"],
                             min_rotation = params["min_rotation"],
                             max_rotation = params["max_rotation"],
                             min_translation = params["min_translation"],
                             max_translation = params["max_translation"],
                             min_scale = params["min_scale"],
                             max_scale = params["max_scale"],
                             jitter_std = params["jitter_std"]
                             )

        val_data.set_params(image_dim = params["image_dim"],
                             camera_fov_deg = params["camera_fov_deg"],
                             f = params["f"],
                             camera_dist = params["camera_dist"],
                             soft_min_k = params["soft_min_k"],
                             transforms = params["transforms"], #Note to self - this is changed to none further down. Not going insane.
                             min_rotation = params["min_rotation"],
                             max_rotation = params["max_rotation"],
                             min_translation = params["min_translation"],
                             max_translation = params["max_translation"],
                             min_scale = params["min_scale"],
                             max_scale = params["max_scale"],
                             jitter_std = params["jitter_std"]
                             )

    elif params["dataset_type"] == utils.dataset.TreeSpeciesDataset:
        params["image_dim"] = trees_data.image_dim
        params["camera_fov_deg"] = trees_data.camera_fov_deg
        params["f"] = trees_data.f
        params["camera_dist"] = trees_data.camera_dist
        params["num_views"] = trees_data.depth_images.shape[1]
        params["depth_averaging"] = "min"


    if trees_data.soft_min_k:
        params["soft_min_k"] = trees_data.soft_min_k

    experiment_name = wandb.util.generate_id()

    if init_wandb:
        run = wandb.init(
            project=wandb_project,
            group=experiment_name,
            config=params,
            reinit=False
            )
    
    wandb.config.update(params)
    config = wandb.config
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    torch.random.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    for specie in list(set(trees_data.species) - set(config.species)):
        print("Removing: {}".format(specie))
        trees_data.remove_species(specie)
        val_data.remove_species(specie)

    print('Train Dataset:')
    print(trees_data.counts)
    print('Species: ', trees_data.species)
    print('Labels: ', trees_data.labels)
    print('Total count: ', len(trees_data))
    print()

    print('Validation Dataset (should match):')
    print(val_data.counts)
    print('Species: ', val_data.species)
    print('Labels: ', val_data.labels)
    print('Total count: ', len(val_data))
    print()

    assert len(val_data) == len(trees_data)

    dataset_size = len(trees_data)
    indices = list(range(dataset_size))
    split2 = int(np.floor( (config.validation_split+config.test_split) * dataset_size ))
    split1 = int(np.floor( config.test_split * dataset_size ))

    if config.shuffle_dataset:
        print("Shuffling dataset...")
        np.random.shuffle(indices)
                 
    train_indices, val_indices, test_indices = indices[split2:], indices[split1:split2], indices[:split1]

    #Train sampler==========================================
    if config.train_sampler == "random": 
        print("Using random/uniform sampling...")
        train_sampler = SubsetRandomSampler(train_indices) #Unifrom random sampling without replacement
    elif config.train_sampler == "balanced":
        print("Using balanced sampling...")
        labels = trees_data.labels[train_indices] #Counts over 
        counts = torch.bincount(labels) #Training set only
        label_weights = 1 / counts 

        sample_weights = torch.stack([label_weights[label] for label in trees_data.labels]) #Corresponding weight for each sample
        sample_weights[val_indices] = 0 #Never sample the validation dataset - set weights to zero
        sample_weights[test_indices] = 0 #Same for test dataset

        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) #Replacement is true by default for the weighted sampler
    #=======================================================    

    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(trees_data, batch_size=config.batch_size, 
                                               sampler=train_sampler)

    val_data.set_params(transforms=['none']) #Turn off transforms for the validation dataset - DON'T GIVE IT AN EMPTY LIST
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, #Val_data has all the data in it - not just val and test. Make sure to use the right samplers
                                                    sampler=val_sampler)
    
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size,
                                              sampler=test_sampler)
                                                

    assert set(config.species) == set(trees_data.species)

    if config.model=="SimpleView":
        model = SimpleView(
            num_views=config.num_views,
            num_classes=len(config.species)
        )

    model = model.to(device=device)

    if config.loss_fn=="cross-entropy":
        loss_fn = nn.CrossEntropyLoss()
        print("Using cross-entropy loss...")
    if config.loss_fn=="smooth-loss":
        loss_fn = utils.smooth_loss
        print("Using smooth-loss")

    if type(config.learning_rate) == list:
        lr = config.learning_rate[0]
        step_size = config.learning_rate[1]
        gamma = config.learning_rate[2]
    else:
        lr = config.learning_rate

    if config.optimizer=="sgd":
        print("Optimizing with SGD...")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.momentum)
    elif config.optimizer=="adam":
        print("Optimizing with AdaM...")
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if type(config.learning_rate) == list:
        print("Using step LR scheduler...")
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    #wandb.watch(model)
    best_acc = 0
    best_min_acc = 0
    
    best_test_acc = 0
    best_min_test_acc = 0
    
    for epoch in range(config.epochs):  # loop over the dataset multiple times

        #Training loop============================================
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            depth_images = data['depth_images']
            labels = data['labels']

            depth_images = depth_images.to(device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(depth_images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

        #Test loop================================================
        num_train_correct = 0
        num_train_samples = 0

        num_val_correct = 0
        num_val_samples = 0
        
        num_test_correct = 0
        num_test_samples = 0

        running_train_loss = 0
        running_val_loss = 0
        running_test_loss = 0

        model.eval()  
        with torch.no_grad():
            #Train set eval==============
            for data in train_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)
                num_train_correct += (predictions == labels).sum()
                num_train_samples += predictions.size(0)

                running_train_loss += loss_fn(scores, labels)

            train_acc = float(num_train_correct)/float(num_train_samples)
            train_loss = running_train_loss/len(validation_loader)

            
            
            
            

            #Val set eval===============
            all_labels = torch.tensor([]).to(device)
            all_predictions = torch.tensor([]).to(device)

            for data in validation_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)

                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                num_val_correct += (predictions == labels).sum()
                num_val_samples += predictions.size(0)

                running_val_loss += loss_fn(scores, labels)

            val_acc = float(num_val_correct)/float(num_val_samples)
            val_loss = running_val_loss/len(validation_loader)

            print(f'OVERALL (Val): Got {num_val_correct} / {num_val_samples} with accuracy {val_acc*100:.2f}')

            cm = confusion_matrix(all_labels.cpu(), all_predictions.cpu())
            totals = cm.sum(axis=1)
            
            accs = np.zeros(len(totals))
            for i in range(len(totals)):
                accs[i] = cm[i,i]/totals[i]
                print(f"{trees_data.species[i]}: Got {cm[i,i]}/{totals[i]} with accuracy {(cm[i,i]/totals[i])*100:.2f}")
                wandb.log({f"{trees_data.species[i]} Accuracy":(cm[i,i]/totals[i])}, commit = False)


            if val_acc >= best_acc:
                best_model_state = copy.deepcopy(model.state_dict())
                best_acc = val_acc
                
            wandb.log({"Best_acc":best_acc}, commit = False)
                
            if min(accs) >= best_min_acc:
                best_min_model_state = copy.deepcopy(model.state_dict())
                best_min_acc = min(accs)
                
            wandb.log({"Best_min_acc":best_min_acc}, commit = False)
            #==================================
            
            
            
            
            
            
            
            
            
            #test set eval===============
            all_labels = torch.tensor([]).to(device)
            all_predictions = torch.tensor([]).to(device)

            for data in test_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)

                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                num_test_correct += (predictions == labels).sum()
                num_test_samples += predictions.size(0)

                running_test_loss += loss_fn(scores, labels)

            test_acc = float(num_test_correct)/float(num_test_samples)
            test_loss = running_test_loss/len(test_loader)

            print(f'OVERALL (test): Got {num_test_correct} / {num_test_samples} with accuracy {test_acc*100:.2f}')

            cm = confusion_matrix(all_labels.cpu(), all_predictions.cpu())
            totals = cm.sum(axis=1)
            
            accs = np.zeros(len(totals))
            for i in range(len(totals)):
                accs[i] = cm[i,i]/totals[i]
                print(f"{trees_data.species[i]}: Got {cm[i,i]}/{totals[i]} with accuracy {(cm[i,i]/totals[i])*100:.2f}")
                wandb.log({f"{trees_data.species[i]} Accuracy":(cm[i,i]/totals[i])}, commit = False)


            if test_acc >= best_test_acc:
                best_test_model_state = copy.deepcopy(model.state_dict())
                best_test_acc = test_acc
                
            wandb.log({"Best_test_acc":best_test_acc}, commit = False)
                
            if min(accs) >= best_min_test_acc:
                best_min_test_model_state = copy.deepcopy(model.state_dict())
                best_min_test_acc = min(accs)
                
            wandb.log({"Best_min_test_acc":best_min_test_acc}, commit = False)
            #==================================
            
            
            
            
                
            wandb.log({
                "Train Loss":train_loss,
                "Validation Loss":val_loss,
                "Test Loss":test_loss,
                "Train Accuracy":train_acc,
                "Validation Accuracy":val_acc,
                "Test Accuracy":test_acc,
                "Learning Rate":optimizer.param_groups[0]['lr'],
                "Epoch":epoch
                })

            scheduler.step()


    print('Finished Training')

    print('Saving best model...')
    print('Best overall accuracy: {}'.format(best_acc))
    torch.save(best_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=wandb.run.name+'_best')
              )
    print('Saved!')
    
    print('Saving converged model...')
    print('Converged accuracy: {}'.format(val_acc))
    converged_model_state = copy.deepcopy(model.state_dict())
    torch.save(converged_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=wandb.run.name+'_converged')
              )
    print('Saved!')
    
    print('Saving best producer accuracy model...')
    print('Best min producer accuracy: {}'.format(best_min_acc))
    torch.save(best_min_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=wandb.run.name+'_best_prod')
              )
    print('Saved!')
    
    print('Saving best (test) model...')
    print('Best overall accuracy: {}'.format(best_test_acc))
    torch.save(best_test_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=wandb.run.name+'_best_test'))
               
    print('Saving best (test) producer accuracy model...')
    print('Best (test) min producer accuracy: {}'.format(best_min_test_acc))
    torch.save(best_min_test_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=wandb.run.name+'_best_test_prod'))
               


    if "run" in locals():
        run.finish()