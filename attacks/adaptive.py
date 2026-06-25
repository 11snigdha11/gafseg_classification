import torch
import copy

def get_honest_updates(honest_models, global_model):
    """ Helper to extract honest update vectors, ignoring BatchNorm buffers. """
    global_dict = global_model.state_dict()
    honest_updates = []
    
    for model in honest_models:
        local_dict = model.state_dict()
        update = {}
        for k in global_dict.keys():
            if 'running' not in k and 'num_batches_tracked' not in k:
                update[k] = local_dict[k].float() - global_dict[k].float()
        honest_updates.append(update)
        
    return honest_updates, global_dict

def lie_attack(honest_models, global_model, z=1.5):
    """ 
    A Little Is Enough (LIE): 
    Adds a tiny amount of variance to the mean. It bypasses distance-based aggregators.
    """
    honest_updates, global_dict = get_honest_updates(honest_models, global_model)
    attacked_dict = {}

    for k in global_dict.keys():
        if 'running' in k or 'num_batches_tracked' in k:
            attacked_dict[k] = honest_models[0].state_dict()[k].float()
        else:
            stacked_updates = torch.stack([u[k] for u in honest_updates], dim=0)
            
            layer_mean = torch.mean(stacked_updates, dim=0)
            layer_std = torch.std(stacked_updates, dim=0)
            
            # The LIE injection: Mean + z * Std
            malicious_update = layer_mean + (z * layer_std)
            attacked_dict[k] = global_dict[k].float() + malicious_update

    malicious_model = copy.deepcopy(honest_models[0])
    malicious_model.load_state_dict(attacked_dict)
    return malicious_model

def min_max_attack(honest_models, global_model, dev_type='std'):
    """
    Min-Max Attack: 
    Pushes the mean backward as far as possible while keeping the maximum distance 
    to any honest client within normal boundaries.
    """
    honest_updates, global_dict = get_honest_updates(honest_models, global_model)
    attacked_dict = {}
    
    # We use a static scaling multiplier (gamma) for empirical Min-Max. 
    # (True Min-Max requires complex optimization, this is the standard heuristic).
    gamma = 2.0 

    for k in global_dict.keys():
        if 'running' in k or 'num_batches_tracked' in k:
            attacked_dict[k] = honest_models[0].state_dict()[k].float()
        else:
            stacked_updates = torch.stack([u[k] for u in honest_updates], dim=0)
            layer_mean = torch.mean(stacked_updates, dim=0)
            layer_std = torch.std(stacked_updates, dim=0)
            
            # Min-Max forces the update opposite to the honest mean, bounded by Std
            malicious_update = layer_mean - (gamma * layer_std)
            attacked_dict[k] = global_dict[k].float() + malicious_update

    malicious_model = copy.deepcopy(honest_models[0])
    malicious_model.load_state_dict(attacked_dict)
    return malicious_model

def min_sum_attack(honest_models, global_model, dev_type='std'):
    """
    Min-Sum Attack: 
    Similar to Min-Max, but ensures the *sum* of squared distances to honest clients
    is less than the maximum sum of squared distances among honest clients.
    """
    honest_updates, global_dict = get_honest_updates(honest_models, global_model)
    attacked_dict = {}
    
    # Min-Sum uses a slightly tighter gamma than Min-Max
    gamma = 1.0 

    for k in global_dict.keys():
        if 'running' in k or 'num_batches_tracked' in k:
            attacked_dict[k] = honest_models[0].state_dict()[k].float()
        else:
            stacked_updates = torch.stack([u[k] for u in honest_updates], dim=0)
            layer_mean = torch.mean(stacked_updates, dim=0)
            layer_std = torch.std(stacked_updates, dim=0)
            
            # Min-Sum heuristic vector
            malicious_update = layer_mean - (gamma * layer_std)
            attacked_dict[k] = global_dict[k].float() + malicious_update

    malicious_model = copy.deepcopy(honest_models[0])
    malicious_model.load_state_dict(attacked_dict)
    return malicious_model

