import sys
import logging
import torch
import transformers
import copy
from pathlib import Path

transformers.logging.set_verbosity(40)

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger

logger = logging.getLogger(__name__)


def _resolve_checkpoint_path(save_to):
    if not save_to:
        return None

    raw_path = Path(save_to).expanduser()
    if raw_path.is_file():
        return raw_path

    repo_root = Path(__file__).resolve().parents[3]
    search_roots = [Path.cwd(), repo_root, repo_root.parent]
    candidate_names = [raw_path.name]
    if not raw_path.name.startswith('final_'):
        candidate_names.insert(0, f'final_{raw_path.name}')

    seen = set()
    candidates = []

    def add_candidate(path):
        path = path.expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    if raw_path.is_absolute():
        add_candidate(raw_path)
        if not raw_path.name.startswith('final_'):
            add_candidate(raw_path.with_name(f'final_{raw_path.name}'))
    else:
        for root in search_roots:
            add_candidate(root / raw_path)
            if not raw_path.name.startswith('final_'):
                add_candidate(root / raw_path.parent / f'final_{raw_path.name}')

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = []
    for root in search_roots:
        if not root.exists():
            continue
        for candidate_name in candidate_names:
            matches.extend(root.rglob(candidate_name))

    matches = [match for match in matches if match.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


class FSChatBot(object):
    """
    A chatbot class that uses a language model for generating responses.

    This class implements a chatbot that can interact with users using natural
    language. It uses a pretrained language model as the backbone and can
    optionally load a fine-tuned checkpoint from federated learning. It can
    also use history and prompt templates to enhance the conversation quality.
    It provides two methods for generating responses: predict and generate.

    Attributes:
        tokenizer: A transformers.PreTrainedTokenizer object that can
            encode and decode text.
        model: A transformers.PreTrainedModel object that can generate text.
        device: A string representing the device to run the model on.
        add_special_tokens: A boolean indicating whether to add special tokens
            to the input and output texts.
        max_history_len: An integer representing the maximum number of
            previous turns to use as context.
        max_len: An integer representing the maximum number of tokens to
            generate for each response.
        history: A list of lists of integers representing the tokenized input
            and output texts of previous turns.
    """
    def __init__(self, config):
        """
        Initializes the chatbot with the given configuration.

        Args:
            config: A FS configuration object that contains various settings
                for the chatbot.
        """
        model_name, model_hub = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len, model_hub)
        self.model = get_llm(config)

        if config.eval_device >= 0:
            self.device = f'cuda:{config.eval_device}'
        elif config.device >= 0:
            self.device = f'cuda:{config.device}'
        else:
            self.device = 'cpu'
        self.add_special_tokens = True

        if config.llm.offsite_tuning.use:
            from federatedscope.llm.offsite_tuning.utils import \
                wrap_offsite_tuning_for_eval
            self.model = wrap_offsite_tuning_for_eval(self.model, config)
        else:
            try:
                ckpt_path = _resolve_checkpoint_path(config.federate.save_to)
                if ckpt_path is None:
                    raise FileNotFoundError(config.federate.save_to)
                if str(ckpt_path) != str(config.federate.save_to):
                    logger.info("Resolved checkpoint `%s` to `%s`.",
                                config.federate.save_to, ckpt_path)
                ckpt = torch.load(ckpt_path, map_location='cpu')
                if 'model' in ckpt and 'cur_round' in ckpt:
                    self.model.load_state_dict(ckpt['model'])
                else:
                    self.model.load_state_dict(ckpt)
            except Exception as error:
                print(f"{error}, will use raw model.")

        if config.train.is_enable_half:
            self.model.half()

        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        if getattr(config.eval.llm, 'compile_model', False) and \
                torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []

    def _build_prompt(self, input_text):
        """
        Builds a prompt template for the input text.

        Args:
            input_text: A string representing the user's input text.

        Returns:
            A string representing the source text with a prompt template.
        """
        source = {'instruction': input_text}
        return PROMPT_DICT['prompt_no_input'].format_map(source)

    def predict(self, input_text, use_history=True, use_prompt=True):
        """
        Generates a response for the input text using the model.

        Args:
            input_text: A string representing the user's input text.
            use_history: A boolean indicating whether to use previous turns as
                context for generating the response. Default is True.
            use_prompt: A boolean indicating whether to use a prompt
                template for creating the source text. Default is True.

        Returns:
            A string representing the chatbot's response text.
        """
        if use_prompt:
            input_text = self._build_prompt(input_text)
        text_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        self.history.append(text_ids)
        input_ids = []
        if use_history:
            for history_ctx in self.history[-self.max_history_len:]:
                input_ids.extend(history_ctx)
        else:
            input_ids.extend(text_ids)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).to(self.device)
        response = self.model.generate(input_ids=input_ids,
                                       max_new_tokens=self.max_len,
                                       num_beams=4,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True,
                                       temperature=0.0)

        self.history.append(response[0].tolist())
        response_tokens = \
            self.tokenizer.decode(response[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
        return response_tokens

    def _sanitize_generate_kwargs(self, generate_kwargs):
        generation_kwargs = dict(generate_kwargs)
        do_sample = generation_kwargs.get('do_sample')
        if do_sample is None:
            generation_config = getattr(self.model, 'generation_config', None)
            do_sample = getattr(generation_config, 'do_sample', False)

        if not do_sample:
            # Override sampling-only knobs so greedy evaluation stays warning-free
            # even when the base model ships with sampling defaults.
            generation_kwargs['do_sample'] = False
            generation_kwargs['temperature'] = 1.0
            generation_kwargs['top_k'] = 50
            generation_kwargs['top_p'] = 1.0

        return generation_kwargs

    @torch.no_grad()
    def generate(self, input_text, generate_kwargs=None):
        """
        Generates a response for the input text using the model and
        additional arguments.

        Args:
            input_text: A string representing the user's input text.
            generate_kwargs: A dictionary of keyword arguments to pass to the
                model's generate method.

        Returns:
            A string or a list of strings representing the chatbot's response
            text. If the generate_kwargs contains num_return_sequences > 1,
            then a list of strings is returned. Otherwise, a single string is
            returned.
        """
        if generate_kwargs is None:
            generate_kwargs = {}
        generate_kwargs = self._sanitize_generate_kwargs(generate_kwargs)

        is_batch_input = isinstance(input_text, (list, tuple))
        input_texts = list(input_text) if is_batch_input else [input_text]

        original_padding_side = getattr(self.tokenizer, 'padding_side',
                                        'right')
        try:
            if is_batch_input:
                self.tokenizer.padding_side = 'left'
            tokenized_inputs = self.tokenizer(input_texts,
                                              padding=True,
                                              truncation=True,
                                              add_special_tokens=True,
                                              return_tensors="pt")
        finally:
            self.tokenizer.padding_side = original_padding_side
        input_ids = tokenized_inputs.input_ids.to(self.device)
        attention_mask = tokenized_inputs.attention_mask.to(self.device)

        output_ids = self.model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         **generate_kwargs)
        response = []
        prompt_width = input_ids.shape[1]
        for i in range(output_ids.shape[0]):
            response.append(
                self.tokenizer.decode(output_ids[i][prompt_width:],
                                      skip_special_tokens=True,
                                      ignore_tokenization_space=True))

        if len(response) > 1 or is_batch_input:
            return response
        return response[0]

    def clear(self):
        """Clears the history of previous turns.

        This method can be used to reset the chatbot's state and start a new
        conversation.
        """
        self.history = []


class FSChatBot_My(FSChatBot):
    def __init__(self,
                 model,
                 config,
                 device=None,
                 copy_model=False,
                 compile_model=False):
        model_name, model_hub = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len, model_hub)
        if device is None:
            if config.eval_device >= 0:
                device = f'cuda:{config.eval_device}'
            else:
                device = next(model.parameters()).device

        self.device = str(device)
        self.model = copy.deepcopy(model) if copy_model else model
        model_device = str(next(self.model.parameters()).device)
        if model_device != self.device:
            self.model = self.model.to(self.device)

        self.add_special_tokens = True

        if config.llm.offsite_tuning.use:
            from federatedscope.llm.offsite_tuning.utils import \
                wrap_offsite_tuning_for_eval
            self.model = wrap_offsite_tuning_for_eval(self.model, config)

        if config.train.is_enable_half:
            self.model.half()

        self.model = self.model.eval()
        if compile_model and torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    chat_bot = FSChatBot(init_cfg)
    welcome = "Welcome to FSChatBot，" \
              "`clear` to clear history，" \
              "`quit` to end chat."
    print(welcome)
    while True:
        input_text = input("\nUser:")
        if input_text.strip() == "quit":
            break
        if input_text.strip() == "clear":
            chat_bot.clear()
            print(welcome)
            continue
        print(f'\nFSBot: {chat_bot.predict(input_text)}')


if __name__ == "__main__":
    main()
