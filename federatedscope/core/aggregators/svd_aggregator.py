import torch
from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.auxiliaries.utils import param2tensor
from copy import deepcopy
from scipy.optimize import linear_sum_assignment  # pip install scipy


class SVDAggregator(ClientsAvgAggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(ClientsAvgAggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def _para_weighted_avg(self, models, recover_fun=None):
        """
        Calculates the weighted average of models.
        """
        svd_mode = getattr(self.cfg.aggregator, 'svd_mode', 'svd').lower()
        if svd_mode == 'fedma':
            return self._para_aligned_avg(models, recover_fun)
        if svd_mode == 'full_rank':
            return self._full_rank_avg(models, recover_fun)

        # return self._new_avg2(models, recover_fun=recover_fun)
        # return self._new_avg(models, recover_fun=recover_fun)
        # return self._para_weighted_avgb(models, recover_fun=recover_fun)


        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = deepcopy(models[0])

        use_gram = getattr(self.cfg, 'use_gram', False)

        for key in avg_model:
            if 'lora_A' in key:
                local_a = param2tensor(models[0][1][key])
                rank = local_a.shape[0]
                hidden_dim = local_a.shape[-1]

                if use_gram:
                    bkey = key.replace('lora_A', 'lora_B')
                    whitened = []
                    for _, model in models:
                        a_i = param2tensor(model[key]).float()
                        b_i = param2tensor(model[bkey]).float()
                        G = b_i.T @ b_i
                        G = 0.5 * (G + G.T)
                        e, V = torch.linalg.eigh(G.double())
                        e_pos = torch.clamp(e, min=0.0)
                        G_sqrt = (V * torch.sqrt(e_pos + 1e-12)).matmul(
                            V.T).to(dtype=a_i.dtype)
                        whitened.append(G_sqrt @ a_i)
                    A = torch.stack(whitened,
                                    dim=0).reshape(-1, hidden_dim)
                else:
                    A = torch.stack([param2tensor(model[key])
                                     for _, model in models],
                                    dim=0).reshape(-1, hidden_dim).float()
                U, S, V = torch.svd_lowrank(A, q=rank, niter=3)
                Vt = V.t()
                Vt = torch.sqrt(torch.diag(S)) @ Vt
                avg_model[key] = Vt.to(dtype=local_a.dtype)
                # avg_model[key] =  torch.diag(S/len(models)) @ Vt
                # avg_model[key] =  torch.mean(U.reshape((1,8,8)), dim=0) @ torch.diag(S) @ Vt
                # avg_model[key] =  torch.mean(torch.stack([model[key] for _, model in models], dim=0), dim=0)

        for key in avg_model:
            if 'lora_B' in key:
                a_key = key.replace('lora_B', 'lora_A')
                # avg_model[key] = torch.mean(model[key] @ model[a_key] @ torch.linalg.pinv(avg_model[a_key].float()).half() for _, model in models)
                avg_model[key] = torch.mean(
                    torch.stack([
                        param2tensor(model[key]).float() @
                        param2tensor(model[a_key]).float() @
                        avg_model[a_key].float().t()
                        for _, model in models
                    ],
                                dim=0),
                    dim=0).to(dtype=param2tensor(models[0][1][key]).dtype)

        return avg_model

    def _full_rank_avg(self, models, recover_fun=None):
        total_samples = sum(sz for sz, _ in models)
        _, avg_model = deepcopy(models[0])

        def get_weight(sample_size):
            if self.cfg.federate.ignore_weight:
                return 1.0 / len(models)
            if self.cfg.federate.use_ss:
                return 1.0
            return sample_size / total_samples

        def weighted_average(key):
            value = None
            for sample_size, local_model in models:
                local_value = param2tensor(local_model[key])
                weight = get_weight(sample_size)
                contribution = local_value.float() * weight
                value = contribution if value is None else value + contribution
            if self.cfg.federate.use_ss and recover_fun:
                value = recover_fun(value)
                value /= total_samples
                value = torch.FloatTensor(value)
            return value.to(dtype=param2tensor(models[0][1][key]).dtype)

        processed_b = set()
        for key in list(avg_model.keys()):
            if 'lora_A' not in key:
                if 'lora_B' not in key:
                    avg_model[key] = weighted_average(key)
                elif key not in processed_b:
                    avg_model[key] = weighted_average(key)
                continue

            bkey = key.replace('lora_A', 'lora_B')
            if bkey not in avg_model:
                avg_model[key] = weighted_average(key)
                continue

            local_a = param2tensor(models[0][1][key])
            local_b = param2tensor(models[0][1][bkey])
            rank = local_a.shape[0]

            w_avg = None
            for sample_size, local_model in models:
                local_a_i = param2tensor(local_model[key]).float()
                local_b_i = param2tensor(local_model[bkey]).float()
                weight = get_weight(sample_size)
                contribution = (local_b_i @ local_a_i) * weight
                w_avg = contribution if w_avg is None else w_avg + contribution

            U, S, V = torch.svd_lowrank(w_avg, q=rank, niter=3)
            avg_model[key] = V.t().to(dtype=local_a.dtype)
            avg_model[bkey] = (U @ torch.diag(S)).to(dtype=local_b.dtype)
            processed_b.add(bkey)

        # for key in list(avg_model.keys()):
        #     if 'lora_B' in key and key not in processed_b:
        #         avg_model[key] = weighted_average(key)

        return avg_model

    def _para_aligned_avg(self, models, recover_fun=None):
        """
        Align LoRA-A row order client-wise before averaging.
        For LoRA-B, reconstruct the averaged factor against the aligned A.
        """
        total_samples = sum(sz for sz, _ in models)
        _, avg_model = deepcopy(models[0])

        def get_weight(sample_size):
            if self.cfg.federate.ignore_weight:
                return 1.0 / len(models)
            if self.cfg.federate.use_ss:
                return 1.0
            return sample_size / total_samples

        def weighted_average(key):
            value = None
            for sample_size, local_model in models:
                local_value = param2tensor(local_model[key])
                weight = get_weight(sample_size)
                contribution = local_value.float() * weight
                value = contribution if value is None else value + contribution
            if self.cfg.federate.use_ss and recover_fun:
                value = recover_fun(value)
                value /= total_samples
                value = torch.FloatTensor(value)
            return value.to(dtype=param2tensor(models[0][1][key]).dtype)

        aligned_a_by_key = {}

        for key in avg_model:
            if 'lora_A' not in key:
                continue

            reference = param2tensor(models[0][1][key]).float()
            aligned_states = [reference]
            aligned_sum = reference * get_weight(models[0][0])

            for sample_size, client_state in models[1:]:
                mat = param2tensor(client_state[key]).float()
                diff = mat.unsqueeze(1) - reference.unsqueeze(0)
                cost = (diff**2).sum(-1).cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost)
                row_ind = torch.as_tensor(row_ind, device=mat.device)
                col_ind = torch.as_tensor(col_ind, device=mat.device)
                perm = row_ind[torch.argsort(col_ind)]
                aligned = mat[perm]
                aligned_states.append(aligned)
                aligned_sum += aligned * get_weight(sample_size)

            avg_model[key] = aligned_sum.to(dtype=reference.dtype)
            aligned_a_by_key[key] = aligned_states

        for key in list(avg_model.keys()):
            if 'lora_B' not in key:
                if 'lora_A' not in key:
                    avg_model[key] = weighted_average(key)
                continue

            a_key = key.replace('lora_B', 'lora_A')
            if a_key not in aligned_a_by_key:
                avg_model[key] = weighted_average(key)
                continue

            avg_a = avg_model[a_key].float()
            avg_b = None
            for idx, (sample_size, client_state) in enumerate(models):
                if key not in client_state:
                    continue
                local_b = param2tensor(client_state[key]).float()
                local_a = aligned_a_by_key[a_key][idx]
                reconstructed_b = local_b @ local_a @ avg_a.t()
                weight = get_weight(sample_size)
                contribution = reconstructed_b * weight
                avg_b = contribution if avg_b is None else avg_b + contribution

            if avg_b is not None:
                avg_model[key] = avg_b.to(
                    dtype=param2tensor(models[0][1][key]).dtype)

        return avg_model


    def _new_avg(self, models, recover_fun=None):
        return self._full_rank_avg(models, recover_fun)

    def _new_avg2(self, models, recover_fun=None):
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = deepcopy(models[0])
        for key in avg_model:
            if 'lora_A' in key:
                bkey = key.replace('lora_A', 'lora_B')
                AB = torch.stack([torch.concat([model[key], model[bkey].T], dim=0) for _, model in models], dim=0).reshape(-1, 2048).float()
                U, S, V = torch.svd_lowrank(AB, q=8, niter=3)
                Vt = V.t()
                S = torch.diag(S)
                # A = torch.mean(torch.stack([U[i*8:(i+1)*8] @ S @ Vt[:, :1024] for i in range(len(models))], dim=0), dim=0)
                # B = torch.mean(torch.stack([Vt[:, 1024:].T @ S @ U[i*8:(i+1)*8].T for i in range(len(models))], dim=0), dim=0)
                A = Vt[:, :1024]
                B = torch.mean(torch.stack([Vt[:, 1024:].T @ S @ S @ U[i*8:(i+1)*8].T @ U[i*8:(i+1)*8] for i in range(len(models))], dim=0), dim=0)

                avg_model[key] = A.half()
                avg_model[bkey] = B.half()

        return avg_model
