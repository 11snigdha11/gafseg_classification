import copy
import torch



def signflip_attack_model(local_model, global_model):

    local_dict = local_model.state_dict()
    global_dict = global_model.state_dict()

    attacked_dict = {}

    for k in global_dict.keys():

        update = local_dict[k].float() - global_dict[k].float()

        attacked_dict[k] = (
            global_dict[k].float() - update
        )

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

        update = local_dict[k].float() - global_dict[k].float()

        update_norm = torch.norm(update)

        random_update = torch.randn_like(update)

        random_update = (
            random_update
            / (torch.norm(random_update) + 1e-12)
            * update_norm
        )

        attacked_dict[k] = (
            global_dict[k].float() + random_update
        )

    local_model.load_state_dict(attacked_dict)

    return local_model    



def gaussian_attack_model(local_model, global_model, sigma=0.01):

    local_dict = local_model.state_dict()
    global_dict = global_model.state_dict()

    attacked_dict = {}

    for k in global_dict.keys():

        update = local_dict[k].float() - global_dict[k].float()

        noise = sigma * torch.randn_like(update)

        attacked_dict[k] = (
            global_dict[k].float() + update + noise
        )

    local_model.load_state_dict(attacked_dict)

    return local_model    