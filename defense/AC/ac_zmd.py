




# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements methods performing poisoning detection based on activations clustering.
| Paper link: https://arxiv.org/abs/1811.03728
| Please keep in mind the limitations of defences. For more information on the limitations of this
    defence, see https://arxiv.org/abs/1905.13409 . For details on how to evaluate classifier security
    in general, see https://arxiv.org/abs/1902.06705
"""
import logging
import time

from calendar import c

import torch
import logging
import argparse
import sys
import os


sys.path.append('../')
sys.path.append(os.getcwd())
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils_ac.clustering_analyzer import ClusteringAnalyzer
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import *
from utils.save_load_attack import load_attack_result
sys.path.append(os.getcwd())
import yaml
from pprint import pprint, pformat

def segment_by_class(data , classes: np.ndarray, num_classes: int) -> List[np.ndarray]:
    by_class: List[List[int]] = [[] for _ in range(num_classes)]

    for indx, feature in enumerate(classes):
        if len(classes.shape) == 2 and classes.shape[1] > 1:

            assigned = np.argmax(feature)

        else:

            assigned = int(feature)
        if torch.is_tensor(data[indx]):
            by_class[assigned].append(data[indx].cpu().numpy())
        else:
            by_class[assigned].append(data[indx])
    return [np.asarray(i) for i in by_class]

def measure_misclassification(
    classifier, x_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Computes 1-accuracy given x_test and y_test
    :param classifier: Classifier to be used for predictions.
    :param x_test: Test set.
    :param y_test: Labels for test set.
    :return: 1-accuracy.
    """
    predictions = np.argmax(classifier.predict(x_test), axis=1)
    return 1.0 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]


def train_remove_backdoor(
    classifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    tolerable_backdoor: float,
    max_epochs: int,
    batch_epochs: int,
) -> tuple:
    """
    Trains the provider classifier until the tolerance or number of maximum epochs are reached.
    :param classifier: Classifier to be used for predictions.
    :param x_train: Training set.
    :param y_train: Labels used for training.
    :param x_test: Samples in test set.
    :param y_test: Labels in test set.
    :param tolerable_backdoor: Parameter that determines how many misclassifications are acceptable.
    :param max_epochs: maximum number of epochs to be run.
    :param batch_epochs: groups of epochs that will be run together before checking for termination.
    :return: (improve_factor, classifier).
    """
    # Measure poison success in current model:
    initial_missed = measure_misclassification(classifier, x_test, y_test)

    curr_epochs = 0
    curr_missed = 1.0
    while curr_epochs < max_epochs and curr_missed > tolerable_backdoor:
        classifier.fit(x_train, y_train, nb_epochs=batch_epochs)
        curr_epochs += batch_epochs
        curr_missed = measure_misclassification(classifier, x_test, y_test)

    improve_factor = initial_missed - curr_missed
    return improve_factor, classifier


def cluster_activations(
    separated_activations: List[np.ndarray],
    nb_clusters: int = 2,
    nb_dims: int = 10,
    reduce: str = "FastICA",
    clustering_method: str = "KMeans",
    generator = None,
    clusterer_new = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Clusters activations and returns two arrays.
    1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each data point
    in the class has been assigned.
    2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method.
    :param separated_activations: List where separated_activations[i] is a np matrix for the ith class where
           each row corresponds to activations for a given data point.
    :param nb_clusters: number of clusters (defaults to 2 for poison/clean).
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :param clustering_method: Clustering method to use, default is KMeans.
    :param generator: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations).
    :param clusterer_new: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations)
    """
    separated_clusters = []
    separated_reduced_activations = []

    if clustering_method == "KMeans":
        clusterer = KMeans(n_clusters=nb_clusters)
    else:
        raise ValueError(clustering_method + " clustering method not supported.")

    for activation in separated_activations:
        # Apply dimensionality reduction
        nb_activations = np.shape(activation)[1]
        if nb_activations > nb_dims & np.shape(activation)[0] > nb_dims:
            # TODO: address issue where if fewer samples than nb_dims this fails
            reduced_activations = reduce_dimensionality(activation, nb_dims=nb_dims, reduce=reduce)
        elif nb_activations <= nb_dims:
            reduced_activations = activation
        else:
            reduced_activations = activation[:,0:(nb_dims)]
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        if generator is not None and clusterer_new is not None:
            clusterer_new = clusterer_new.partial_fit(reduced_activations)
            # NOTE: this may cause earlier predictions to be less accurate
            clusters = clusterer_new.predict(reduced_activations)
        elif reduced_activations.shape[0] != 1:
            clusters = clusterer.fit_predict(reduced_activations)
        else:
            clusters = 1
        separated_clusters.append(clusters)

    return separated_clusters, separated_reduced_activations


def reduce_dimensionality(activations: np.ndarray, nb_dims: int = 10, reduce: str = "FastICA") -> np.ndarray:
    """
    Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.
    :param activations: Activations to be reduced.
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :return: Array with the reduced activations.
    """
    # pylint: disable=E0001
    from sklearn.decomposition import FastICA, PCA

    if reduce == "FastICA":
        projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    elif reduce == "PCA":
        projector = PCA(n_components=nb_dims)
    else:
        raise ValueError(reduce + " dimensionality reduction method not supported.")

    reduced_activations = projector.fit_transform(activations)
    return reduced_activations


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    ####添加额外
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')

    #####AC
    parser.add_argument('--nb_dims', type=int, help='train epoch')
    parser.add_argument('--nb_clusters', type=int, help='the number of mini_batch train model')
    parser.add_argument('--cluster_analysis', type=str, help='the method of cluster analysis')
    
    arg = parser.parse_args()

    print(arg)
    return arg

def ac(args,result,config):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    if args.log is not None and args.log != '':
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    nb_dims = args.nb_dims
    nb_clusters = args.nb_clusters
    cluster_analysis = args.cluster_analysis

    model = generate_cls_model(args.model,args.num_classes)
    model.load_state_dict(result['model'])
    model.to(args.device)
    #data_set = get_dataset_train(args)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = torch.tensor(nCHW_to_nHWC(result['bd_train']['x'].numpy()))
    y = result['bd_train']['y']
    data_set = torch.utils.data.TensorDataset(x,y)
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    batch_size = args.batch_size
    num_samples = len(data_loader.dataset)
    num_classes = args.num_classes
    for i, (x_batch,y_batch) in enumerate(data_loader):  # type: ignore
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)
        batch_activations = get_activations(result['model_name'],model,x_batch.to(args.device))
        activation_dim = batch_activations.shape[-1]

        # initialize values list of lists on first run
        if i == 0:
            activations_by_class = [np.empty((0, activation_dim)) for _ in range(num_classes)]
            clusters_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]
            red_activations_by_class = [np.empty((0, nb_dims)) for _ in range(num_classes)]

        activations_by_class_i = segment_by_class(batch_activations, y_batch,args.num_classes)
        clusters_by_class_i, red_activations_by_class_i = cluster_activations(
            activations_by_class_i,
            nb_clusters=nb_clusters,
            nb_dims=nb_dims,
            reduce='PCA',
            clustering_method='KMeans'
        )

        for class_idx in range(num_classes):
            activations_by_class[class_idx] = np.vstack(
                [activations_by_class[class_idx], activations_by_class_i[class_idx]]
            )
            clusters_by_class[class_idx] = np.append(
                clusters_by_class[class_idx], [clusters_by_class_i[class_idx]]
            )
            red_activations_by_class[class_idx] = np.vstack(
                [red_activations_by_class[class_idx], red_activations_by_class_i[class_idx]]
            )

    ###analyze
    analyzer = ClusteringAnalyzer()
    if cluster_analysis == "smaller":
        (
            assigned_clean_by_class,
            poisonous_clusters,
            report,
        ) = analyzer.analyze_by_size(clusters_by_class)
    elif cluster_analysis == "relative-size":
        (
            assigned_clean_by_class,
            poisonous_clusters,
            report,
        ) = analyzer.analyze_by_relative_size(clusters_by_class)
    elif cluster_analysis == "distance":
        (assigned_clean_by_class, poisonous_clusters, report,) = analyzer.analyze_by_distance(
            clusters_by_class,
            separated_activations=red_activations_by_class,
        )
    elif cluster_analysis == "silhouette-scores":
        (assigned_clean_by_class, poisonous_clusters, report,) = analyzer.analyze_by_silhouette_score(
            clusters_by_class,
            reduced_activations_by_class=red_activations_by_class,
        )
    else:
        raise ValueError("Unsupported cluster analysis technique " + cluster_analysis)

    ###detect

    batch_size = args.batch_size
    is_clean_lst = []
    # loop though the generator to generator a report
    last_loc = torch.zeros(args.num_classes).numpy().astype(int)
    for i, (x_batch,y_batch) in enumerate(data_loader):  # type: ignore
        indices_by_class = segment_by_class(np.arange(batch_size), y_batch,args.num_classes)
        is_clean_lst_i = [0] * batch_size
        clean_class = [0] * batch_size
        for class_idx, idxs in enumerate(indices_by_class):
            for idx_in_class, idx in enumerate(idxs):
                is_clean_lst_i[idx] = assigned_clean_by_class[class_idx][idx_in_class + last_loc[class_idx]]
            last_loc[class_idx] = last_loc[class_idx] + len(idxs)
        is_clean_lst += is_clean_lst_i
    

    ###reliable
    
    data_set_o.subset([i for i,v in enumerate(is_clean_lst) if v==1])
    data_loader_sie = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) 
    criterion = torch.nn.CrossEntropyLoss() 
    
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].numpy()))
    y = result['bd_test']['y']
    data_bd_test = torch.utils.data.TensorDataset(x,y)
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].numpy()))
    y = result['clean_test']['y']
    data_clean_test = torch.utils.data.TensorDataset(x,y)
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    best_acc = 0
    best_asr = 0
    for j in range(args.epochs):
        for i, (inputs,labels) in enumerate(data_loader_sie):  # type: ignore
            model.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            asr_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

            
            clean_acc = 0
            for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)
        
        if not (os.path.exists(os.getcwd() + f'{args.save_path}/ac/ckpt_best/')):
            os.makedirs(os.getcwd() + f'{args.save_path}/ac/ckpt_best/')
        if best_acc < clean_acc:
            best_acc = clean_acc
            best_asr = asr_acc
            torch.save(
            {
                'model_name':args.model,
                'model': model.cpu().state_dict(),
                'asr': asr_acc,
                'acc': clean_acc
            },
            f'./{args.save_path}/ac/ckpt_best/defense_result.pt'
            )
        logging.info(f'Epoch{j}: clean_acc:{clean_acc} asr:{asr_acc} best_acc:{best_acc} best_asr{best_asr}')

    result['model'] = model
    result['dataset'] = data_set_o
    return result       

   

def get_activations(name,model,x_batch):
    TOO_SMALL_ACTIVATIONS = 32
    assert name in ['preactresnet18', 'vgg19', 'resnet18']
    if name == 'preactresnet18':
        inps,outs = [],[]
        def layer_hook(module, inp, out):
            outs.append(out.data)
        hook = model.avgpool.register_forward_hook(layer_hook)
        _ = model(x_batch)
        activations = outs[0].view(outs[0].size(0), -1)
        hook.remove()
    elif name == 'vgg19':
        inps,outs = [],[]
        def layer_hook(module, inp, out):
            outs.append(out.data)
        hook = model.relu5_4.register_forward_hook(layer_hook)
        _ = model(x_batch)
        activations = outs[0].view(outs[0].size(0), -1)
        hook.remove()
    elif name == 'resnet18':
        inps,outs = [],[]
        def layer_hook(module, inp, out):
            outs.append(out.data)
        hook = model.layer4.register_forward_hook(layer_hook)
        _ = model(x_batch)
        activations = outs[0].view(outs[0].size(0), -1)
        hook.remove()


    # if nodes_last_layer <= TOO_SMALL_ACTIVATIONS:
    #     logger.warning(
    #         "Number of activations in last hidden layer is too small. Method may not work properly. " "Size: %s",
    #         str(nodes_last_layer),
    #     )
    return activations




if __name__ == '__main__':
    
    args = get_args()
    with open("./defense/AC/config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    
    

    ######为了测试临时写的代码
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/ac/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/ac/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log) 
    args.save_path = save_path
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    if args.save_path is not None:
        print("Continue training...")
        result_defense = ac(args,result,config)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].numpy()))
        y = result['bd_test']['y']
        data_bd_test = torch.utils.data.TensorDataset(x,y)
        data_bd_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_bd_test,
            poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    
        asr_acc = 0
        for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = result_defense['model'](inputs)
            pre_label = torch.max(outputs,dim=1)[1]
            asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].numpy()))
        y = result['clean_test']['y']
        data_clean_test = torch.utils.data.TensorDataset(x,y)
        data_clean_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_clean_test,
            poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    
        clean_acc = 0
        for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = result_defense['model'](inputs)
            pre_label = torch.max(outputs,dim=1)[1]
            clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)


        if not (os.path.exists(os.getcwd() + f'{save_path}/ac/')):
            os.makedirs(os.getcwd() + f'{save_path}/ac/')
        torch.save(
        {
            'model_name':args.model,
            'model': result_defense['model'].cpu().state_dict(),
            'asr': asr_acc,
            'acc': clean_acc
        },
        f'./{save_path}/ac/defense_result.pt'
        )
    else:
        print("There is no target model")