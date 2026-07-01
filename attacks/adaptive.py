import torch
import copy
from scipy.stats import norm

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

def lie_attack(honest_models, global_model, num_clients, num_byzantine):
    """
    A Little Is Enough (LIE)
    Baruch et al., NeurIPS 2019

    Constructs a malicious update:
        delta_m = mean - z_max * std

    where z_max is computed from the number of honest and Byzantine clients.
    """

    honest_updates, global_dict = get_honest_updates(honest_models, global_model)

    attacked_dict = {}

    # ---------- Compute z_max ----------
    # n = total clients
    # f = Byzantine clients

    s = num_clients // 2 + 1 - num_byzantine

    # Prevent invalid probability
    p = (num_clients - s) / num_clients
    p = min(max(p, 1e-6), 1 - 1e-6)

    z_max = norm.ppf(p)

    print(f"LIE attack: z = {z_max:.4f}")

    # ---------- Build malicious model ----------
    for k in global_dict.keys():

        # Keep BatchNorm statistics unchanged
        if "running" in k or "num_batches_tracked" in k:
            attacked_dict[k] = honest_models[0].state_dict()[k].clone()

        else:

            stacked = torch.stack(
                [update[k] for update in honest_updates],
                dim=0
            )

            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0, unbiased=False)

            # LIE malicious update
            malicious_update = mean - z_max * std

            attacked_dict[k] = (
                global_dict[k].float()
                + malicious_update
            )

    malicious_model = copy.deepcopy(global_model)
    malicious_model.load_state_dict(attacked_dict)

    return malicious_model

def min_max_attack(honest_models, global_model):
    """
    True Min-Max attack (Fang et al., USENIX Security 2020)

    Finds the largest perturbation that remains inside the
    maximum pairwise distance of the honest updates.
    """

    honest_updates, global_dict = get_honest_updates(
        honest_models,
        global_model
    )

    attacked_dict = {}

    for k in global_dict.keys():

        # Preserve BatchNorm buffers
        if "running" in k or "num_batches_tracked" in k:
            attacked_dict[k] = honest_models[0].state_dict()[k].clone()
            continue

        updates = torch.stack(
            [u[k] for u in honest_updates],
            dim=0
        )

        mean = updates.mean(dim=0)

        # Direction used by Fang et al.
        direction = torch.sign(mean)

        # --------------------------------------------------
        # Compute maximum pairwise honest distance
        # --------------------------------------------------

        honest_max_dist = 0.0

        num_honest = updates.shape[0]

        for i in range(num_honest):
            for j in range(i + 1, num_honest):

                dist = torch.sum(
                    (updates[i] - updates[j]) ** 2
                )

                honest_max_dist = max(
                    honest_max_dist,
                    dist.item()
                )

        # --------------------------------------------------
        # Binary search lambda
        # --------------------------------------------------

        lam_low = 0.0
        lam_high = 10.0

        for _ in range(20):

            lam = (lam_low + lam_high) / 2

            malicious = mean - lam * direction

            max_dist = 0.0

            for i in range(num_honest):

                dist = torch.sum(
                    (malicious - updates[i]) ** 2
                )

                max_dist = max(
                    max_dist,
                    dist.item()
                )

            if max_dist <= honest_max_dist:

                lam_low = lam

            else:

                lam_high = lam

        malicious_update = mean - lam_low * direction

        attacked_dict[k] = (
            global_dict[k].float()
            + malicious_update
        )

    malicious_model = copy.deepcopy(global_model)
    malicious_model.load_state_dict(attacked_dict)

    return malicious_model

def min_sum_attack(honest_models, global_model):
    """
    True Min-Sum attack (Fang et al., USENIX Security 2020)

    Finds the largest perturbation whose SUM of squared
    distances to honest updates remains within the honest
    distribution.
    """

    honest_updates, global_dict = get_honest_updates(
        honest_models,
        global_model
    )

    attacked_dict = {}

    for k in global_dict.keys():

        # Preserve BatchNorm statistics
        if "running" in k or "num_batches_tracked" in k:
            attacked_dict[k] = honest_models[0].state_dict()[k].clone()
            continue

        updates = torch.stack(
            [u[k] for u in honest_updates],
            dim=0
        )

        mean = updates.mean(dim=0)

        direction = torch.sign(mean)

        # ------------------------------------------
        # Honest reference (maximum honest sum distance)
        # ------------------------------------------

        honest_sum_max = 0.0

        num_honest = updates.shape[0]

        for i in range(num_honest):

            total = 0.0

            for j in range(num_honest):

                total += torch.sum(
                    (updates[i] - updates[j]) ** 2
                ).item()

            honest_sum_max = max(
                honest_sum_max,
                total
            )

        # ------------------------------------------
        # Binary search λ
        # ------------------------------------------

        lam_low = 0.0
        lam_high = 10.0

        for _ in range(20):

            lam = (lam_low + lam_high) / 2

            malicious = mean - lam * direction

            malicious_sum = 0.0

            for i in range(num_honest):

                malicious_sum += torch.sum(
                    (malicious - updates[i]) ** 2
                ).item()

            if malicious_sum <= honest_sum_max:

                lam_low = lam

            else:

                lam_high = lam

        malicious_update = mean - lam_low * direction

        attacked_dict[k] = (
            global_dict[k].float()
            + malicious_update
        )

    malicious_model = copy.deepcopy(global_model)
    malicious_model.load_state_dict(attacked_dict)

    return malicious_model

