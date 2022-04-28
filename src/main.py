import torch
import pickle
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import random_split

from src.data.make_dataset import get_KMNIST_data, get_MNIST_data
from src.models.model import VGG, Net, NetSoftmax, LinearNet, ConvNet
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

        #data = torch.reshape(data, (-1,784,))

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

def _curv_closure(model, miner, loss_fn, calculator, X, y, batch_idx):
    embeddings = model(X)

    a1, p, a2, n = miner(embeddings, y)
    loss = loss_fn(embeddings, y, (a1, p, a2, n))

    x1 = X[torch.cat((a1, a2))]
    x2 = X[torch.cat((p, n))]
    t = torch.cat((torch.ones(p.shape[0]), torch.zeros(n.shape[0])))

    print(f"Hessian {batch_idx}")
    H = calculator.calculate_hessian(x1, x2, t, model=model, num_outputs=embeddings.shape[-1], agg='mean')

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


def preproc_data(dataset_name='KMNIST'):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    if dataset_name == 'KMNIST':
        train_data, test_data = get_KMNIST_data(data_dir="./data/",
                                    transform=transform)
    else:
        train_data, test_data = get_MNIST_data(data_dir="./data/",
                                    transform=transform)

    lengths = [int(len(train_data)*0.9), int(len(train_data)*0.1)]
    training_data, val_data = random_split(train_data, lengths)
    
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=TRAINING_HP['batch_size'])

    return train_loader, test_loader, val_loader, training_data, val_data, test_data


def reshuffle_train(training_data):
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)
    return train_loader


def run():

    wandb.login(key=WANDB_KEY)
    wandb.init(project=PROJECT['name'], name=PROJECT['experiment'], entity="unc-laplace", config=TRAINING_HP)

    train_loader, test_loader, val_loader, train_data, val_data, test_data = preproc_data('KMNIST')

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    model = LinearNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_HP['lr'])
    
    loss_func, mining_func = setup_pytorch_metric_learning(TRAINING_HP)

    for epoch in range(1, TRAINING_HP['epochs'] + 1):

        train(model, loss_func, mining_func, train_loader, optimizer, epoch, device)
        #break

        ### RESHUFFLE FOR NEXT EPOCH ###
        #train_loader = reshuffle_train(train_data)

    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)

        #data = torch.reshape(data, (-1,784,))

        model.eval()
        optimizer.zero_grad()
        embeddings = model(data)

        if mining_func:
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
        else:
            loss = loss_func(embeddings, labels)

        print(f"Iteration {batch_idx}/{len(test_loader)}: Loss = {loss}")
    
    train_embeddings, train_labels = get_all_embeddings(train_data, model)
    test_embeddings, test_labels = get_all_embeddings(test_data, model)
    knn_score = knn(train_embeddings, train_labels, test_embeddings, test_labels)
    print(f"KNN Score: {knn_score}")
    wandb.log({"knn": knn_score})
    
    #plot_umap("FUCKTHISSHIT", train_embeddings, train_labels)

    print("Saving model weights...")
    torch.save(model.state_dict(),'temp/model.pt')
    
    print("Computing Hessian...")
    calculator = ContrastiveHessianCalculator()

    hs = []
    print(f"Val Loader length: {len(val_loader)}")
    for batch_idx, (x, y) in enumerate(val_loader):
        #x = torch.reshape(x, (-1,784,))
        _, h = _curv_closure(model, mining_func, loss_func, calculator, x, y, batch_idx)
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

    samples = sample(mu_q, sigma_q, n_samples=100)

    print("In sample inference...")
    preds = []
    for net_sample in samples:
        vector_to_parameters(net_sample, model.parameters())
        batch_preds = []
        for x, _ in test_loader:
            #x = torch.reshape(x, (-1,784,))
            pred = model(x)
            batch_preds.append(pred)
        preds.append(torch.cat(batch_preds, dim=0))
    preds = torch.stack(preds)

    with open("pred_in.pkl", "wb") as f:
        pickle.dump(preds, f)


    print("Out of sample inference...")
    train_loader, test_loader, val_loader, train_data, val_data, test_data = preproc_data('MNIST')
    preds_ood = []
    for net_sample in samples:
        vector_to_parameters(net_sample, model.parameters())
        batch_preds = []
        for x, _ in test_loader:
            #x = torch.reshape(x, (-1,784,))
            pred = model(x)
            batch_preds.append(pred)
        preds_ood.append(torch.cat(batch_preds, dim=0))
    preds_ood = torch.stack(preds_ood)

    with open("pred_ood.pkl", "wb") as f:
        pickle.dump(preds_ood, f)

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