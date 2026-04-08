import torch
import logging
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.glue.model.adapter_builder import AdapterModel
import numpy as np
from federatedscope.glue.trainer.metric_utils import get_glue_metric


logger = logging.getLogger(__name__)

from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator

class MAPATrainer(GeneralTorchTrainer):

    def _hook_on_epoch_start(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.{ctx.cur_split}_loader``      Initialize DataLoader
            ==================================  ===========================
        """
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_split)) is None:
            loader = get_dataloader(
                WrapDataset(ctx.get("{}_data".format(ctx.cur_split))),
                self.cfg, ctx.cur_split)
            setattr(ctx, "{}_loader".format(ctx.cur_split), ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_split)),
                            ReIterator):
            setattr(ctx, "{}_loader".format(ctx.cur_split),
                    ReIterator(ctx.get("{}_loader".format(ctx.cur_split))))
        else:
            ctx.get("{}_loader".format(ctx.cur_split)).reset()


        # added my MAHI
        # 4. Generate a random matrix A of shape (d_pad/k, 1) in float16.
        # ctx.A = torch.randn(ctx.s, 1, device=ctx.device, dtype=torch.float16) #/ np.sqrt(ctx.s)
        ctx.A = [torch.randn(s, 1, device=ctx.device, dtype=torch.float16) for s in ctx.s] #/ np.sqrt(ctx.s)

    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                if ctx.cfg.llm.adapter.args[0].get('adapter_method', '') == "vera":
                    # added by me, for VeRA, introduce separate learning rates for the classification head and the adapted layers
                    vera_params = [param for name, param in ctx.model.named_parameters() if "vera" in name and param.requires_grad]
                    other_params = [param for name, param in ctx.model.named_parameters() if "vera" not in name and param.requires_grad]
                    optimizer_grouped_parameters = [
                        {'params': vera_params, 'lr': ctx.cfg.train.optimizer.lr},
                        {'params': other_params, 'lr': ctx.cfg.train.vera.lr_c}
                    ]
                    from transformers import AdamW, get_linear_schedule_with_warmup
                    ctx.optimizer = AdamW(optimizer_grouped_parameters, no_deprecation_warning=True)
                    ctx.scheduler = get_linear_schedule_with_warmup(
                                    ctx.optimizer, 
                                    num_warmup_steps=0.06 * ctx.cfg.train.local_update_steps * ctx.cfg.federate.total_round_num, 
                                    num_training_steps=ctx.cfg.train.local_update_steps * ctx.cfg.federate.total_round_num
                    )
                else:
                    ctx.optimizer = get_optimizer(
                        ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                    ctx.scheduler = get_scheduler(
                        ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)    # modified by me, for GLUE

        # added by mahi
        grad_list = []
        for name, param in ctx.model.named_parameters():
            if param is not None and ctx.cfg.personalization.local_param[0] not in name and param.numel() >= ctx.cfg.mapa_rank:
                grad_list.append(param.detach().view(-1))
        # if grad_list:
        #     G = torch.cat(grad_list).half()  # convert gradients to float16
        # else:
        #     first_param = next(ctx.model.parameters())
        #     G = torch.tensor([], dtype=torch.float16, device=first_param.device)

        # 2. Pad G with zeros (float16) so that its length is divisible by k.
        d = [G.numel() for G in grad_list]
        k = [dd // ctx.cfg.mapa_rank for dd in d] # Ensure that ctx.k is defined.
        # print('All:  ', d)
        remainder = [dd % kk for dd, kk in zip(d, k)]
        s = [(dd + kk - rr) // kk if rr != 0 else dd // kk for dd, kk, rr in zip(d, k, remainder)]
        ctx.d = CtxVar(d, LIFECYCLE.ROUTINE)
        ctx.k = CtxVar(k, LIFECYCLE.ROUTINE)
        ctx.r = CtxVar(remainder, LIFECYCLE.ROUTINE)
        ctx.s = CtxVar(s, LIFECYCLE.ROUTINE)

        # ctx.A = torch.randn(ctx.s, 1, device=ctx.device, dtype=torch.float16) / np.sqrt(ctx.s)
        print('Total memory overhead: ', sum(s))
        print('Total K: ', sum(k))

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['label'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        
        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        preds = outputs.logits.argmax(dim=-1)  # modified by me, for GLUE
        loss = outputs.loss
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(preds, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        # Zero the gradients and perform the backward pass.
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()

        # Apply gradient clipping if enabled.
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)

        # ===== Custom Gradient Modification Operation (using float16) =====
        # 1. Gather all parameter gradients into a single flat vector G (in float16).
        grad_list = []
        for name, param in ctx.model.named_parameters():
            if param.grad is not None and ctx.cfg.personalization.local_param[0] not in name and param.grad.numel() > ctx.cfg.mapa_rank: # remove personalization
                grad_list.append(param.grad.detach().view(-1))
        # if grad_list:
        #     G = torch.cat(grad_list).half()  # convert gradients to float16
        # else:
        #     first_param = next(ctx.model.parameters())
        #     G = torch.tensor([], dtype=torch.float16, device=first_param.device)

        # 2. Pad G with zeros (float16) so that its length is divisible by k.
        # d = G.numel()
        # k = d//10  #  Ensure that ctx.k is defined.
        # # print('All:  ', d)
        # remainder = d % k


        # if remainder != 0:
        #     pad = k - remainder
        #     G = torch.cat([G, torch.zeros((pad,), device=G.device, dtype=torch.float16)])
        G = [torch.cat([grad, torch.zeros((k-r,), device=grad.device, dtype=torch.float16)]) if r != 0 else grad for grad, k, r in zip(grad_list, ctx.k, ctx.r)]
        # 3. Reshape G into a matrix of shape (d_pad/k, k)
        G_matrix = [GG.view(-1, k) for GG, k in zip(G, ctx.k)]

        # 4. Generate a random matrix A of shape (d_pad/k, 1) in float16.
        # A = torch.randn(G_matrix.shape[0], 1, device=G.device, dtype=torch.float16) / np.sqrt(G_matrix.shape[0])

        # 5. Compute the new gradient matrix: GG_matrix = A @ (A.T @ G_matrix)
        GG_matrix = [A @ (A.T @ GG) for GG, A in zip(G_matrix, ctx.A)]
        # GG_matrix = torch.randn(G_matrix.shape[0], G_matrix.shape[1], device=G.device, dtype=torch.float16)
        # 6. Flatten GG_matrix back to a vector and remove any padding.
        GG = [GG.view(-1)[:d] if r != 0 else GG.view(-1) for GG,d,r in zip(GG_matrix, ctx.d, ctx.r)]
        # if remainder != 0:
        #     GG = GG[:d]

        # 7. Replace the parameter gradients with the corresponding values from GG.
        pointer = 0
        for name, param in ctx.model.named_parameters():
            if param.grad is not None and ctx.cfg.personalization.local_param[0] not in name and param.grad.numel() > ctx.cfg.mapa_rank:
                # numel = param.grad.numel()
                new_grad = GG[pointer].view_as(param.grad)
                param.grad.data.copy_(new_grad)
                pointer += 1
        # ===== End of Custom Operation =====

        # Update model parameters.
        ctx.optimizer.step()

        # Step the scheduler if available.
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return
        
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate, use extend not append
        ctx.ys_true.extend(ctx.y_true.to('cpu').detach().numpy())
        ctx.ys_pred.extend(ctx.y_pred.to('cpu').detach().numpy())
        
        
        # ctx.ys_true.extend(ctx.y_true.detach().cpu().numpy())
        # ctx.ys_pred.extend(ctx.y_pred.detach().cpu().numpy())
        
    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                f'{ctx.cur_split}_total': ctx.num_samples,
                f'{ctx.cur_split}_avg_loss': avg_loss
        }
        glue_metric = get_glue_metric(ctx.cfg.data.type.split('@')[0])
        eval_metric = glue_metric.compute(predictions=ctx.ys_pred, references=ctx.ys_true)
        for k, v in eval_metric.items():
            eval_results[f'{ctx.cur_split}_{k}'] = v
        
        setattr(ctx, 'eval_metrics', eval_results)
        
        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_mapa_trainer(trainer_type):
    if trainer_type == 'mapatrainer':
        trainer_builder = MAPATrainer
        return trainer_builder


register_trainer('mapatrainer', call_mapa_trainer)
