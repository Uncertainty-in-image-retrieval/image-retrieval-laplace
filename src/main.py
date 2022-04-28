
import torch
import torch.optim as optim

from torchvision import transforms

from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.neighbors import KNeighborsClassifier

from src.data.make_dataset import get_data
from src.models.model import Net
from src.utils.pytorch_metric_learning import setup_pytorch_metric_learning
from src.utils.plots import plot_umap

import argparse
import yaml
import wandb



### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, mining_func, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss})
        if batch_idx % 20 == 0:
            if TRAINING_HP['miner'] == 'TripletMarginMiner':
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}, Number of mined triplets = {mining_func.num_triplets}")
                wandb.log({"mined triples": mining_func.num_triplets})
            elif TRAINING_HP['miner'] == 'BatchEasyHardMiner':
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}")

    return model
            
        
### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def knn(train_embeddings, train_labels, test_embeddings, test_labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_embeddings, train_labels)
    testing_acc = knn.score(test_embeddings, test_labels)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    plot_umap("test", test_embeddings, test_labels)

    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    
    #print("Computing accuracy")
    #accuracies = accuracy_calculator.get_accuracy(
    #    test_embeddings, train_embeddings, test_labels, train_labels, False, include=["precision_at_1"])
    
    #print(f"Test set accuracy (Precision@1) = {accuracies}")


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

    training_data, test_data = get_data(data_dir="./data/", 
                                    transform=transform)

    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=TRAINING_HP['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=TRAINING_HP['batch_size'])

    return train_loader, test_loader, training_data, test_data


def run():

    wandb.login(key=WANDB_KEY)
    wandb.init(project=PROJECT['name'], name=PROJECT['experiment'], config=TRAINING_HP)

    train_loader, test_loader, training_data, test_data = preproc_data()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_HP['lr'])

    loss_func, mining_func = setup_pytorch_metric_learning(TRAINING_HP)

    for epoch in range(1, TRAINING_HP['epochs'] + 1):
        model = train(model, loss_func, mining_func, train_loader, optimizer, epoch)
        train_embeddings, train_labels = get_all_embeddings(training_data, model)
        plot_umap(f"train_{epoch}", train_embeddings, train_labels)
        
    test(training_data, test_data, model)

    train_embeddings, train_labels = get_all_embeddings(training_data, model)
    test_embeddings, test_labels = get_all_embeddings(test_data, model)
    knn_score = knn(train_embeddings, train_labels, test_embeddings, test_labels)
    print(f"KNN Score: {knn_score}")
    wandb.log({"knn": knn_score})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Uncertainty in Image Retrieval with Laplace approx.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    set_global_args(args)

    run()
    