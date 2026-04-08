# ref: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import os
import torch
import numpy as np
import pandas as pd
from federatedscope.llm.eval.eval_for_mmlu.categories import \
     subcategories, categories
import json
import transformers
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.core.data.utils import download_url
import tarfile

transformers.logging.set_verbosity(40)

choices = ["A", "B", "C", "D"]
DEFAULT_EVAL_BATCH_SIZE = 4
MAX_MMLU_PROMPT_LEN = 1024


def format_subject(subject):
    ll = subject.split("_")
    s = ""
    for entry in ll:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice \
        questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def get_eval_batch_size(cfg):
    return max(1, int(getattr(cfg.eval.llm, 'batch_size',
                              DEFAULT_EVAL_BATCH_SIZE)))


def get_answer_token_ids(tokenizer):
    return torch.tensor([
        tokenizer(choice, add_special_tokens=False).input_ids[-1]
        for choice in choices
    ],
                        dtype=torch.long)


def chunk_list(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_subject_prompts(tokenizer, dev_df, test_df, subject, max_prompt_len):
    prompts = []
    labels = []
    max_k = min(5, dev_df.shape[0])

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        label = test_df.iloc[i, test_df.shape[1] - 1]

        for k in range(max_k, -1, -1):
            prompt = gen_prompt(dev_df, subject, k) + prompt_end
            token_count = len(
                tokenizer(prompt, add_special_tokens=True).input_ids)
            if token_count <= max_prompt_len or k == 0:
                prompts.append(prompt)
                labels.append(label)
                break

    return prompts, labels


def get_last_token_positions(attention_mask):
    return attention_mask.size(1) - 1 - attention_mask.flip(
        dims=[1]).argmax(dim=1)


@torch.no_grad()
def eval(subject,
         model,
         tokenizer,
         dev_df,
         test_df,
         device,
         batch_size,
         answer_token_ids,
         max_prompt_len):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1] - 2]

    prompts, labels = build_subject_prompts(tokenizer, dev_df, test_df,
                                            subject, max_prompt_len)
    answer_token_ids = answer_token_ids[:len(answers)].to(device)

    for prompt_batch, label_batch in zip(chunk_list(prompts, batch_size),
                                         chunk_list(labels, batch_size)):
        tokenized = tokenizer(prompt_batch,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=max_prompt_len)
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)

        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask).logits
        last_token_positions = get_last_token_positions(attention_mask)
        batch_indices = torch.arange(input_ids.size(0), device=device)
        last_token_logits = logits[batch_indices, last_token_positions]

        batch_probs = torch.nn.functional.softmax(last_token_logits.
                                                  index_select(
                                                      dim=1,
                                                      index=answer_token_ids
                                                  ).float(),
                                                  dim=1).detach().cpu().numpy()
        batch_preds = batch_probs.argmax(axis=1)

        for pred_idx, label, probs in zip(batch_preds, label_batch,
                                          batch_probs):
            pred = answers[pred_idx]
            cor = pred == label
            cors.append(cor)
            all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def log_mmlu_subject_to_wandb(subject, acc):
    try:
        import wandb
    except ImportError:
        return
    wandb.log({f'eval/mmlu_subject/{subject}': acc})


def log_mmlu_final_to_wandb(cfg, weighted_acc, results):
    if not cfg.wandb.use:
        return

    try:
        import wandb
    except ImportError:
        return

    log_res = {'eval/mmlu_weighted_accuracy': weighted_acc}
    for cat, acc in results.get('categories', {}).items():
        log_res[f'eval/mmlu_category/{cat}'] = acc
    wandb.log(log_res)
    for key, value in log_res.items():
        wandb.summary[key] = value
    wandb.summary['eval/checkpoint'] = cfg.federate.save_to
    wandb.summary['eval/task'] = 'mmlu'


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)
    tokenizer = fschatbot.tokenizer
    model = fschatbot.model
    device = fschatbot.device
    batch_size = get_eval_batch_size(init_cfg)
    answer_token_ids = get_answer_token_ids(tokenizer)
    max_prompt_len = min(
        int(getattr(tokenizer, 'model_max_length', MAX_MMLU_PROMPT_LEN)),
        MAX_MMLU_PROMPT_LEN)

    if not os.path.exists("data/mmlu"):
        download_url("https://people.eecs.berkeley.edu/~hendrycks/data.tar",
                     init_cfg.data.root)
        t = tarfile.open("data/data.tar", "r:")
        os.makedirs("data/mmlu/")
        t.extractall(path="data/mmlu/")
        t.close()

    data_dir = os.path.join(init_cfg.data.root, "mmlu/data")
    eval_dir = "eval_result"

    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f
    ])

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if not os.path.exists(
            os.path.join(eval_dir, "results_{}".format(
                init_cfg.federate.save_to))):
        os.makedirs(
            os.path.join(eval_dir,
                         "results_{}".format(init_cfg.federate.save_to)))

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects,
                        desc='Evaluating MMLU',
                        disable=not init_cfg.eval.llm.show_progress):
        dev_df = pd.read_csv(os.path.join(data_dir, "dev",
                                          subject + "_dev.csv"),
                             header=None)[:5]
        test_df = pd.read_csv(os.path.join(data_dir, "test",
                                           subject + "_test.csv"),
                              header=None)

        cors, acc, probs = eval(subject, model, tokenizer, dev_df, test_df,
                                device, batch_size, answer_token_ids,
                                max_prompt_len)
        if init_cfg.wandb.use:
            log_mmlu_subject_to_wandb(subject, acc)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(init_cfg.federate.save_to)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(init_cfg.federate.save_to,
                                               choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(eval_dir,
                         "results_{}".format(init_cfg.federate.save_to),
                         "{}.csv".format(subject)),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))
    log_mmlu_final_to_wandb(init_cfg, weighted_acc, results)

    results_file = os.path.join(
        eval_dir, "accuracies_{}.json".format(
            init_cfg.federate.save_to.replace("/", "_")))
    with open(results_file, "w") as f:
        json.dump(results, f)

    if init_cfg.wandb.use:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
