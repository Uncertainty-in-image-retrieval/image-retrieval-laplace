
import torch
#torch.set_num_threads(1)
import torch.optim as optim

from torchvision import transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


from src.data.make_dataset import get_data
from src.models.model import Net

import argparse
import yaml
import wandb
from sklearn.manifold import TSNE


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
        wandb.log({"loss": loss, "mined triples": mining_func.num_triplets})
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}, Number of mined triplets = {mining_func.num_triplets}")
        

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def plot_tsne(embeddings, labels):
    emb_to_numpy = embeddings.detach().numpy()
    labels_numpy = labels.detach().numpy()
    emb_tsne = TSNE(n_components=2, learning_rate='auto',
                init='random').fit_transform(emb_to_numpy)
    
    print(emb_tsne, labels_numpy)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    #plot_tsne(train_embeddings, train_labels)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print(train_embeddings)
    print(train_embeddings.size())
    print("Computing accuracy")
    #accuracies = accuracy_calculator.get_accuracy(
    #    test_embeddings, train_embeddings, test_labels, train_labels, False, include=["precision_at_1"]
    #)
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

def setup_pytorch_metric_learning():
    if TRAINING_HP['distance'] == 'Cosine':
        distance = distances.CosineSimilarity()
    elif TRAINING_HP['distance'] == 'LpDistance':
        distance = distances.LpDistance(power=2)

    reducer = reducers.ThresholdReducer(low=0)

    if TRAINING_HP['loss'] == 'ContrastiveLoss':
        if TRAINING_HP['distance'] == 'Cosine':
            loss_func = losses.ContrastiveLoss(1, 0)
        elif TRAINING_HP['distance'] == 'LpDistance':
            loss_func = losses.ContrastiveLoss(0, 1)
    elif TRAINING_HP['loss'] == 'Triplet':
        loss_func = losses.TripletMarginLoss(
            margin=TRAINING_HP['margin'], distance=distance, reducer=reducer)

    mining_func = miners.TripletMarginMiner(
        margin=TRAINING_HP['margin'], distance=distance, type_of_triplets="semihard")

    return loss_func, mining_func

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

    loss_func, mining_func = setup_pytorch_metric_learning()

    accuracy_calculator = AccuracyCalculator()


    for epoch in range(1, TRAINING_HP['epochs'] + 1):
        train(model, loss_func, mining_func, train_loader, optimizer, epoch)
        
    test(training_data, test_data, model, accuracy_calculator)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Uncertainty in Image Retrieval with Laplace approx.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    set_global_args(args)

    run()
    