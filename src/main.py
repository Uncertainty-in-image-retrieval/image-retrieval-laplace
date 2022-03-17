import torch
import torch.nn as nn
import torch.distributions as dists
import torch.optim as optim
from torchvision import transforms

from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from src.data.make_dataset import get_data
from src.models.model import Net, NetSoftmax
from src.utils.pytorch_metric_learning import setup_pytorch_metric_learning
from src.utils.plots import plot_umap

import argparse
import yaml
import wandb
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

from laplace import Laplace
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as pyplot

import matplotlib.pyplot as plt

from visualization import ReliabilityDiagram

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
        wandb.log({"training loss": loss})
        if batch_idx % 20 == 0:
            if TRAINING_HP['miner'] == 'TripletMarginMiner':
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}, Number of mined triplets = {mining_func.num_triplets}")
                wandb.log({"mined triples": mining_func.num_triplets})
            elif TRAINING_HP['miner'] == 'BatchEasyHardMiner':
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}")

def eval(model, loss_func, mining_func, validation_loader, epoch, device):
    for batch_idx, (data, labels) in enumerate(validation_loader):
        data, labels = data.to(device), labels.to(device)
        
        model.eval()
        embeddings = model(data)

        if mining_func:
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
        else:
            loss = loss_func(embeddings, labels)

        wandb.log({"validation loss": loss})
        if batch_idx % 20 == 0:
            if TRAINING_HP['miner'] == 'TripletMarginMiner':
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(validation_loader)}: Loss = {loss}, Number of mined triplets = {mining_func.num_triplets}")
                wandb.log({"mined triples": mining_func.num_triplets})
            elif TRAINING_HP['miner'] == 'BatchEasyHardMiner':
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(validation_loader)}: Loss = {loss}")     
        
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def knn(train_embeddings, train_labels, test_embeddings, test_labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_embeddings, train_labels)
    testing_acc = knn.score(test_embeddings, test_labels)

    return testing_acc


def test(model, test_loader, device, laplace=False):
    acc_map = []
    overall_output, overall_labels = [], []
    #model.eval()
    targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        if laplace:
            embeddings = model(data)
        else:
            embeddings = torch.softmax(model(data), dim=1)

        overall_output.extend(embeddings.detach().numpy())
        overall_labels.extend(labels.detach().numpy())
        probs_map = torch.cat([embeddings], dim=1).detach().numpy()
        y_hat = probs_map.argmax(-1)
        acc_map.extend([1 if y_hat[idx] == labels[idx] else 0 for idx in range(len(y_hat))])
        n_bins = 10
        ece_map = ECE(bins=n_bins).measure(probs_map, labels.detach().numpy())
        nll_map = dists.Categorical(torch.tensor(probs_map)).log_prob(labels).mean()



    diagram = ReliabilityDiagram(n_bins)
    diagram.plot(np.array(overall_output), np.array(overall_labels))
    plt.savefig(f"visualizations/laplace-{PROJECT['experiment']}-{laplace}.png")

    print(f'[MAP] Acc.: {sum(acc_map)/len(acc_map)}; ECE: {ece_map}; NLL: {nll_map}')
    wandb.log({"Accuracy": sum(acc_map)/len(acc_map), "ECE": ece_map, "NLL": nll_map})

    
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

    training_data, test_data, validation_data = get_data(data_dir="./data/", 
                                    transform=transform)

    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=TRAINING_HP['batch_size'])
    validation_loader = torch.utils.data.DataLoader(
        test_data, batch_size=TRAINING_HP['batch_size'])

    return train_loader, test_loader, validation_loader, training_data, test_data, validation_data


def reshuffle_train(training_data):
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)
    return train_loader


def run():

    wandb.login(key=WANDB_KEY)
    wandb.init(project=PROJECT['name'], name=PROJECT['experiment'], config=TRAINING_HP)

    train_loader, test_loader, validation_loader, train_data, test_data, validation_data = preproc_data()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Net().to(device)
    optimizer_model = optim.Adam(model.parameters(), lr=TRAINING_HP['lr'])
    model_softmax = NetSoftmax().to(device)
    optimizer_model_softmax = optim.Adam(model_softmax.parameters(), lr=TRAINING_HP['lr'])
    
    loss_func, mining_func = setup_pytorch_metric_learning(TRAINING_HP)

    cross_entropy_loss = nn.CrossEntropyLoss()
    for epoch in range(1, TRAINING_HP['epochs'] + 1):
        
        ### TRAINING ###
        #print("Training Siamese without Laplace approximation...")
        #train(model, loss_func, mining_func, train_loader, optimizer, epoch, device, laplace=False)

        print("Training Traditional without Laplace approximation...")
        train(model_softmax, cross_entropy_loss, None, train_loader, optimizer_model_softmax, epoch, device)
        eval(model, cross_entropy_loss, None, validation_loader, epoch, device)

        #print("Training Traditional with Laplace approximation...")
        #train(model_softmax, cross_entropy_loss, None, train_loader, optimizer_model_softmax, epoch, device, laplace=True)

        ### RESHUFFLE FOR NEXT EPOCH ###
        train_loader = reshuffle_train(train_data)



    print("Fitting Laplace approximation...") # VERY SLOW
    la = Laplace(model_softmax, 'classification',
         subset_of_weights='last_layer',
         hessian_structure='diag')
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik')

    ### PLOTTING TRAINING EMBEDDINGS ###
    #train_embeddings, train_labels = get_all_embeddings(train_data, model)
    #plot_umap(f"train_{epoch}", train_embeddings, train_labels)


    ### TESTING AND PLOTTING TEST EMBEDDINGS ###
    #print("Testing Siamese without Laplace approximation...")
    #test_embeddings, test_labels = get_all_embeddings(test_data, model)
    #plot_umap(f"test_{epoch}", test_embeddings, test_labels)

    print("Testing Traditional without Laplace approximation...")
    test(model_softmax, test_loader, device, False)
    
    print("Testing with Laplace approximation...")
    test(la, test_loader, device, True)

    ### ACCURACY WITH KNN ###
    #print("Computing accuracy...")
    #train_labels = train_labels.squeeze(1)
    #test_labels = test_labels.squeeze(1)
    #test_acc = knn(train_embeddings, train_labels, test_embeddings, test_labels)
    
    #print(f"Test set accuracy with KNN: {test_acc}")
    #wandb.log({"Embeddings KNN Accuracy": test_acc})

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Uncertainty in Image Retrieval with Laplace approx.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    set_global_args(args)

    run()
    