import copy
import torch


def signflip_attack_model(local_model, global_model):
    local_dict = local_model.state_dict()
    global_dict = global_model.state_dict()
    attacked_dict = {}

    for k in global_dict.keys():
        # Prevent structural NaN-bomb crashes
        if 'running' in k or 'num_batches_tracked' in k:
            attacked_dict[k] = local_dict[k].float()
        else:
            # Apply Sign-Flip to the actual learning weights/biases
            update = local_dict[k].float() - global_dict[k].float()
            attacked_dict[k] = global_dict[k].float() - update

    local_model.load_state_dict(attacked_dict)
    return local_model  

def scaling_attack_model(local_model, global_model, factor=20):

    local_dict = local_model.state_dict()
    global_dict = global_model.state_dict()

    attacked_dict = {}

    for k in global_dict.keys():

        update = local_dict[k].float() - global_dict[k].float()

        attacked_dict[k] = (
            global_dict[k].float() + factor * update
        )

    local_model.load_state_dict(attacked_dict)

    return local_model    


def random_attack_model(local_model, global_model):

    local_dict = local_model.state_dict()
    global_dict = global_model.state_dict()

    attacked_dict = {}

    for k in global_dict.keys():
        if 'running' in k or 'num_batches_tracked' in k:
            attacked_dict[k] = local_dict[k].float()
        else:
            update = local_dict[k].float() - global_dict[k].float()
            update_norm = torch.norm(update)
            random_update = torch.randn_like(update)
            # Scale random noise to match the magnitude of the original update
            random_update = (random_update / (torch.norm(random_update) + 1e-12)) * update_norm
            attacked_dict[k] = global_dict[k].float() + random_update

    local_model.load_state_dict(attacked_dict)
    return local_model





def additive_gaussian_attack_model(local_model, global_model, sigma=0.1):
    local_dict = local_model.state_dict()
    global_dict = global_model.state_dict()
    attacked_dict = {}

    for k in global_dict.keys():
        if 'running' in k or 'num_batches_tracked' in k:
            attacked_dict[k] = local_dict[k].float()
        else:
            # 1. Calculate the legitimate update
            update = local_dict[k].float() - global_dict[k].float()
            
            # 2. Generate small additive noise
            noise = sigma * torch.randn_like(update)
            
            # 3. Attacker sends: Base Global Model + Real Update + Small Noise
            attacked_dict[k] = global_dict[k].float() + update + noise

    local_model.load_state_dict(attacked_dict)
    return local_model   


def gaussian_attack_model(local_model, global_model, sigma=10.0):
    local_dict = local_model.state_dict()
    global_dict = global_model.state_dict()
    attacked_dict = {}

    for k in global_dict.keys():
        # Do not corrupt BatchNorm statistics (prevents the NaN-Bomb)
        if 'running' in k or 'num_batches_tracked' in k:
            attacked_dict[k] = global_dict[k].float()
        else:
            # PURE NOISE ATTACK: Completely replace the legitimate update
            # We use a large sigma to ensure the vector is fully orthogonal
            noise = sigma * torch.randn_like(global_dict[k].float())
            
            # Attacker sends: Base Global Model + Pure Noise
            attacked_dict[k] = global_dict[k].float() + noise

    local_model.load_state_dict(attacked_dict)
    return local_model