import torch
import logging
from contextlib import nullcontext
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
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken   # added by me, for gsm8k evaluation
from federatedscope.llm.misc.fschat import FSChatBot_My   # added by me, for gsm8k evaluation
from federatedscope.llm.eval.eval_for_gsm8k.eval import *   # added by me, for gsm8k evaluation
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT   # added by me, for gsm8k evaluation
from tqdm import tqdm
logger = logging.getLogger(__name__)


class LLMTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                # Flag AMP autocast for forward pass (no GradScaler needed
                # since HF models load natively in fp16/bf16)
                if not hasattr(ctx, '_use_amp'):
                    ctx._use_amp = True
                    logger.info('Enabled AMP autocast for fp16 '
                                'mixed precision training.')

    def _hook_on_fit_start_init(self, ctx):
        # Optional gradient checkpointing (saves VRAM, trades compute)
        if getattr(ctx.cfg.llm, 'gradient_checkpointing', False):
            if hasattr(ctx.model, 'gradient_checkpointing_enable'):
                if not getattr(ctx, '_grad_ckpt_enabled', False):
                    ctx.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={
                            'use_reentrant': False})
                    ctx._grad_ckpt_enabled = True
                    logger.info('Enabled gradient checkpointing.')

        # Optional torch.compile (speeds up training after warmup)
        if getattr(ctx.cfg.llm, 'torch_compile', False):
            if not getattr(ctx, '_model_compiled', False):
                ctx.model = torch.compile(ctx.model,
                                          mode='reduce-overhead')
                ctx._model_compiled = True
                logger.info('Applied torch.compile with reduce-overhead.')

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
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        use_amp = getattr(ctx, '_use_amp', False) and \
            ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]

        amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) \
            if use_amp else nullcontext()

        with amp_ctx:
            if ctx.cfg.llm.deepspeed.use:
                outputs = ctx.model_engine(input_ids=input_ids,
                                           labels=labels,
                                           attention_mask=attention_mask)
            else:
                outputs = ctx.model(input_ids=input_ids,
                                    labels=labels,
                                    attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            ctx.optimizer.zero_grad()

            ctx.loss_task.backward()
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
            ctx.optimizer.step()
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

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                f'{ctx.cur_split}_total': ctx.num_samples,
                f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        if self._use_gsm8k_generation_eval(ctx):
            eval_results[f'{ctx.cur_split}_acc'] = \
                self._run_gsm8k_generation_eval(ctx)

        setattr(ctx, 'eval_metrics', eval_results)
                
        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _use_gsm8k_generation_eval(self, ctx):
        return ctx.cfg.eval.llm.use_generation and \
            'acc' in ctx.cfg.eval.metrics and \
            ctx.cfg.data.type.lower() == 'gsm8k@llm' and \
            ctx.cur_mode in [MODE.TEST, MODE.VAL]

    def _run_gsm8k_generation_eval(self, ctx):
        eval_loader = getattr(ctx, f'{ctx.cur_split}_loader', None)
        if eval_loader is None:
            logger.warning('Skip GSM8K generation eval because `%s_loader` '
                           'is unavailable.', ctx.cur_split)
            return 0.0

        was_training = ctx.model.training
        model_device = next(ctx.model.parameters()).device
        eval_cfg = ctx.cfg.eval.llm
        gsm8k_cfg = ctx.cfg.llm.gsm8k
        fschatbot = FSChatBot_My(ctx.model,
                                 ctx.cfg,
                                 device=model_device,
                                 copy_model=False,
                                 compile_model=eval_cfg.compile_model)

        if hasattr(eval_loader, 'reset'):
            eval_loader.reset()

        raw_loader = eval_loader.loader if hasattr(eval_loader,
                                                   'loader') else eval_loader
        total_questions = getattr(ctx, f'num_{ctx.cur_split}_data', 0)
        if total_questions <= 0 and hasattr(raw_loader, 'dataset'):
            total_questions = len(raw_loader.dataset)
        if gsm8k_cfg.eval_max_samples > 0:
            total_questions = min(total_questions, gsm8k_cfg.eval_max_samples)
        if total_questions <= 0:
            logger.warning('Skip GSM8K generation eval because `%s` split is '
                           'empty.', ctx.cur_split)
            return 0.0

        dataset = getattr(raw_loader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'list_data_dict'):
            samples = dataset.list_data_dict[:total_questions]
        else:
            samples = []
            for batch in raw_loader:
                for instruction, output in zip(batch['instruction'],
                                               batch['output']):
                    if len(samples) >= total_questions:
                        break
                    samples.append({
                        'instruction': instruction,
                        'output': output
                    })
                if len(samples) >= total_questions:
                    break

        eval_res = evaluate_gsm8k_samples(
            fschatbot,
            samples,
            gsm8k_cfg,
            show_progress=eval_cfg.show_progress,
            batch_size=get_gsm8k_eval_batch_size(ctx.cfg))

        if was_training:
            ctx.model.train()

        return eval_res['accuracy']

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
                labels = ctx.data_batch['labels'].to(ctx.device)
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


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
