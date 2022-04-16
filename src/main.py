import torch
import pickle
import torch.nn as nn
import torch.distributions as dists
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import random_split

from src.data.make_dataset import get_data
from src.models.model import VGG, Net, NetSoftmax
from src.models.metric_laplace import MetricLaplace
from src.utils.pytorch_metric_learning import setup_pytorch_metric_learning
from src.utils.plots import plot_umap
from src.models.metric_laplace import ContrastiveHessianCalculator

import argparse
import yaml
import wandb
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

from laplace import Laplace
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram

def train(model, loss_func, mining_func, train_loader, optimizer, epoch, device):
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        model.train()
        optimizer.zero_grad()
        embeddings = model(data)

        if mining_func:
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
        else:
            loss = loss_func(embeddings, labels)

        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss})
        if batch_idx % 20 == 0:
            if TRAINING_HP['miner'] == 'TripletMarginMiner':
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}, Number of mined triplets = {mining_func.num_triplets}")
                wandb.log({"mined triples": mining_func.num_triplets})
            elif TRAINING_HP['miner'] == 'BatchEasyHardMiner':
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}")
        
        
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def _curv_closure(model, miner, loss_fn, calculator, X, y):
    embeddings = model(X)

    a1, p, a2, n = miner(embeddings, y)
    loss = loss_fn(embeddings, y, (a1, p, a2, n))

    x1 = X[torch.cat((a1, a2))]
    x2 = X[torch.cat((p, n))]
    t = torch.cat((torch.ones(p.shape[0]), torch.zeros(n.shape[0])))

    H = calculator.calculate_hessian(x1, x2, t, model=model, num_outputs=embeddings.shape[-1])

    return loss, H


def knn(train_embeddings, train_labels, test_embeddings, test_labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_embeddings, train_labels)
    testing_acc = knn.score(test_embeddings, test_labels)

    return testing_acc
    
def set_global_args(args):
    config_file = args.config
    with open(config_file) as infile:
        config_dict = yaml.load(infile, Loader=yaml.SafeLoader)
    global TRAINING_HP, WANDB_KEY, PROJECT

    TRAINING_HP = config_dict['training']
    WANDB_KEY = config_dict['wandb']
    PROJECT = config_dict['project']


def preproc_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    train_data, test_data = get_data(data_dir="./data/", 
                                    transform=transform)

    lengths = [int(len(train_data)*0.9), int(len(train_data)*0.1)]
    training_data, val_data = random_split(
                train_data, lengths
            )

    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=TRAINING_HP['batch_size'])

    return train_loader, test_loader,val_loader, training_data, val_data, test_data


def reshuffle_train(training_data):
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)
    return train_loader


def run():

    wandb.login(key=WANDB_KEY)
    wandb.init(project=PROJECT['name'], name=PROJECT['experiment'], config=TRAINING_HP)

    train_loader, test_loader, val_loader, train_data, val_data, test_data = preproc_data()

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    model = Net().to(device)
    optimizer_model = optim.Adam(model.parameters(), lr=TRAINING_HP['lr'])
    
    loss_func, mining_func = setup_pytorch_metric_learning(TRAINING_HP)

    print (train_loader)
    for epoch in range(1, TRAINING_HP['epochs'] + 1):

        train(model, loss_func, mining_func, train_loader, optimizer_model, epoch, device)
        break

        ### RESHUFFLE FOR NEXT EPOCH ###
        train_loader = reshuffle_train(train_data)

    #torch.save(model.state_dict(),'temp/tensor.pt')
    calculator = ContrastiveHessianCalculator()

    hs = []
    for x, y in iter(val_loader):
        loss, h = _curv_closure(model, mining_func, loss_func, calculator, x, y)
        hs.append(h)
    hs = torch.stack(hs, dim=0)
    h = torch.sum(hs, dim=0)
    
    mu_q = parameters_to_vector(model.parameters())
    sigma_q = 1 / (h + 1e-6)

    def sample(parameters, posterior_scale, n_samples=100):
        n_params = len(parameters)
        samples = torch.randn(n_samples, n_params, device="cpu")
        samples = samples * posterior_scale.reshape(1, n_params)
        return parameters.reshape(1, n_params) + samples

    samples = sample(mu_q, sigma_q, n_samples=16)

    preds = []
    for net_sample in samples:
        vector_to_parameters(net_sample, model.parameters())
        batch_preds = []
        for x, _ in val_loader:
            pred = model(x)
            batch_preds.append(pred)
        preds.append(torch.cat(batch_preds, dim=0))
    preds = torch.stack(preds)

    with open("pred_in.pkl", "wb") as f:
        pickle.dump({"means": preds.mean(dim=0), "vars": preds.var(dim=0)}, f)

    preds_ood = []
    for net_sample in samples:
        vector_to_parameters(net_sample, model.parameters())
        batch_preds = []
        for x, _ in val_loader:
            x = x + torch.randn(x.shape)  # batch_size, n_channels, width, height
            pred = model(x)
            batch_preds.append(pred)
        preds_ood.append(torch.cat(batch_preds, dim=0))
    preds_ood = torch.stack(preds_ood)

    with open("pred_ood.pkl", "wb") as f:
        pickle.dump({"means": preds_ood.mean(dim=0), "vars": preds_ood.var(dim=0)}, f)

    #print("Fitting Laplace approximation...") # VERY SLOW
    #la = MetricLaplace(model, 'classification')#,
        #subset_of_weights='last_layer')#,
        #hessian_structure='diag')
    #la.fit(train_loader)
    #la.optimize_prior_precision(method='marglik', n_steps=2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Uncertainty in Image Retrieval with Laplace approx.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    set_global_args(args)

    run()