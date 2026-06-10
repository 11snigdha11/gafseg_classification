import copy

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