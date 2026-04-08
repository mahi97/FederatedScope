import torch
import logging
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer

from federatedscope.glue.trainer.trainer import GLUETrainer
from copy import deepcopy
from federatedscope.core.auxiliaries.utils import param2tensor, merge_param_dict

logger = logging.getLogger(__name__)


class SVDTrainer(GLUETrainer):
    def _use_client_svd_logic(self):
        svd_mode = getattr(self.cfg.aggregator, 'svd_mode', 'svd').lower()
        personalized_params = getattr(self.cfg.personalization, 'local_param',
                                      [])
        personalized_b = any('lora_B' in param for param in personalized_params)
        # Client-side SVD logic is only needed when the server sends back
        # LoRA-A while LoRA-B stays personalized on the client.
        return self.cfg.federate.method == 'svd' and svd_mode == 'svd' and \
            personalized_b

    @staticmethod
    def _row_full_rank_pinv(mat, eps=1e-6):
        gram = mat @ mat.T
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return torch.linalg.solve(gram + eps * eye, mat).T

    def update(self, model_parameters, strict=False):
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        if not self._use_client_svd_logic():
            return super().update(model_parameters, strict=strict)

        # model_parameters = deepcopy(model_parameters)
        old_A, new_A, old_B = None, None, None
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
            if "lora_A" in key:
                old_A = self.ctx.model.state_dict()[key]
                new_A = model_parameters[key]
                # new_A = old_A @ torch.linalg.pinv(new_A.float()).half() @ new_A
                # model_parameters[key] = new_A
            if "lora_B" in key:
                old_B = self.ctx.model.state_dict()[key]
                # model_parameters[key] = old_B
            #     # G = old_B.T @ old_B
            #     # G2 = self.safe_matrix_sqrt(G, eps=1e-12)
            #     # old_B = old_B @ torch.clip(torch.linalg.pinv(G2.float()).half(), min=0, max=1)
            #     # old_A = G2 @ old_A
                new_A_pinv = self._row_full_rank_pinv(new_A.float())
                new_B = old_B.float() @ old_A.float() @ new_A_pinv
            #     # new_B = old_B @ old_A @ new_A.T
                model_parameters[key] = new_B.to(dtype=old_B.dtype)

        #
        # import wandb
        # wandb.log({'A':torch.norm(old_A).item(), 'B':torch.norm(old_B).item(), 'new_A':torch.norm(new_A).item(), 'new_B':torch.norm(new_B).item()})
        merged_param = merge_param_dict(self.ctx.model.state_dict().copy(), self._param_filter(model_parameters, filter_keywords=['classifier']))
        self.ctx.model.load_state_dict(merged_param, strict=strict)

    def safe_matrix_sqrt(self, G, eps=1e-12):
        out_dtype = G.dtype
        G = 0.5 * (G + G.T)
        if not torch.isfinite(G).all():
            logger.warning("Non-finite entries detected in Gram matrix; sanitizing before sqrt.")
            G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)

        G = G.double()
        eye = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        last_error = None
        for jitter_scale in (1.0, 1e2, 1e4, 1e6):
            try:
                e, V = torch.linalg.eigh(G + eps * jitter_scale * eye)
                e_pos = torch.clamp(e, min=0.0)
                return (V * torch.sqrt(e_pos)).matmul(V.T).to(dtype=out_dtype)
            except torch._C._LinAlgError as error:
                last_error = error

        logger.warning("Falling back to SVD-based matrix sqrt after eigh failed: %s",
                       last_error)
        U, S, _ = torch.linalg.svd(G + eps * eye, full_matrices=False)
        return (U * torch.sqrt(torch.clamp(S, min=0.0))).matmul(U.T).to(
            dtype=out_dtype)

    # def _hook_on_fit_end(self, ctx):
    #     super(SVDTrainer, self)._hook_on_fit_end(ctx)
    #     # return
    #     model = ctx.model.state_dict().copy()
    #     for key in model:
    #         if "lora_A" in key:
    #             bkey = key.replace('lora_A', 'lora_B')
    #             G = model[bkey].T @ model[bkey]
    #             G2 = self.safe_matrix_sqrt(G, eps=1e-12)
    #             # print(torch.norm(G2).cpu().numpy())
    #             model[key] = G2 @ model[key]
    #
    #     self.ctx.model.load_state_dict(model, strict=False)

    def get_model_para(self):
        if not self._use_client_svd_logic():
            return super().get_model_para()

        model = self.ctx.model.state_dict().copy()
        if self.cfg.use_gram:
            for key in model:
                if "lora_A" in key:
                    bkey = key.replace('lora_A', 'lora_B')
                    G = model[bkey].T @ model[bkey]
                    G2 = self.safe_matrix_sqrt(G, eps=1e-12)
                    # print(torch.norm(G2).cpu().numpy())
                    model[key] = G2 @ model[key]
                    # model[bkey] = model[bkey] @ torch.clip(torch.linalg.pinv(G2.float()).half(), min=0, max=1)
                    # model[bkey] = model[bkey] @ torch.linalg.pinv(G2.float()).half()

        if self.cfg.federate.process_num > 1 or \
                self.cfg.federate.share_local_model or \
                self.cfg.llm.deepspeed.use:
            return self._param_filter(model)
        else:
            return self._param_filter(model)

def call_svd_trainer(trainer_type):
    if trainer_type == 'svdtrainer':
        trainer_builder = SVDTrainer
        return trainer_builder


register_trainer('svdtrainer', call_svd_trainer)
