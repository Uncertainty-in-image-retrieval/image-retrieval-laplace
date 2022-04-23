from pytorch_metric_learning.miners import BatchEasyHardMiner
from pytorch_metric_learning import distances, losses, miners, reducers

def setup_pytorch_metric_learning(TRAINING_HP):
    ### DISTANCE MEASURE
    
    if TRAINING_HP['distance'] == 'Cosine':
        distance = distances.CosineSimilarity()
    elif TRAINING_HP['distance'] == 'LpDistance':
        distance = distances.LpDistance()

    ### REDUCER
    if TRAINING_HP['reducer'] == 'AvgNonZero':
        reducer = reducers.AvgNonZeroReducer()
    elif TRAINING_HP['reducer'] == 'ThresholdReducer':
        reducer = reducers.ThresholdReducer(low=0)
    

    ### LOSS
    #loss_func = losses.ContrastiveLoss()
    
    if TRAINING_HP['loss'] == 'ContrastiveLoss':
        if TRAINING_HP['distance'] == 'Cosine':
            loss_func = losses.ContrastiveLoss(1, 0)
        elif TRAINING_HP['distance'] == 'LpDistance':
            loss_func = losses.ContrastiveLoss(0, 1)
    elif TRAINING_HP['loss'] == 'Triplet':
        loss_func = losses.TripletMarginLoss(
            margin=TRAINING_HP['margin'], distance=distance, reducer=reducer)
    

    ### MINER
    #mining_func = miners.BatchEasyHardMiner()
    
    if TRAINING_HP['miner'] == 'TripletMarginMiner':
        mining_func = miners.TripletMarginMiner(
            margin=TRAINING_HP['margin'], 
            distance=distance, 
            type_of_triplets="semihard")
    elif TRAINING_HP['miner'] == 'BatchEasyHardMiner':
        mining_func = miners.BatchEasyHardMiner(
            pos_strategy=BatchEasyHardMiner.EASY,
            neg_strategy=BatchEasyHardMiner.SEMIHARD,
            allowed_pos_range=None,
            allowed_neg_range=None)
    

    return loss_func, mining_func