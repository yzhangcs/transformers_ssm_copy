import argparse

import fla  # noqa
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from test_utils import (copy_c4_evaluation, phone_book_evaluation,
                        squad_evaluation)
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model', type=str)

    # evaluation
    parser.add_argument('--eval_task', choices=["c4_copy", "squad", "phone_book"],
                        type=str, required=True, help="evaluation task")

    parser.add_argument('--text_order', choices=["standard", "random"], type=str,
                        help="only applies when eval_task = c4_copy. Order of the text to copy. When text_order = random, we randomly change the order of the text. Otherwise, keeps the same order.", default="standard")

    parser.add_argument('--eval_batch_size', default=32, type=int,
                        help="Size of the batch size for evaluation. Only applies when eval_task == ''c4_copy'' or eval_task == 'phone_book'")
    parser.add_argument('--eval_num_batches', default=3, type=int,
                        help="Number of batches for the evaluation to compute mean + std. Only applies when eval_task == ''c4_copy'' or eval_task == 'phone_book'")

    parser.add_argument('--min_eval_len', default=20, type=int,
                        help="Minimum length of the text to copy (when eval_task == ''c4_copy'') or minimum number of entries in the phone book (when eval_task == ''phone_book''). Only applies when eval_task == ''c4_copy'' or eval_task == 'phone_book")

    parser.add_argument('--max_eval_len', default=20, type=int,
                        help="Maximum length of the text to copy (when eval_task == ''c4_copy'') or maximum number of entries in the phone book (when eval_task == ''phone_book''). Only applies when eval_task == ''c4_copy'' or eval_task == 'phone_book")

    return parser.parse_args()


args = parse_args()

print(args)


# Get tokenizer & model

model = MambaLMHeadModel.from_pretrained(args.model, device="cuda", dtype=torch.float16)

# model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
print("^"*100)
print(model)
print("^"*100)
model.eval()


# Evaluation
if args.eval_task == "c4_copy":
    str_acc_mean_list, str_acc_std_list = copy_c4_evaluation(args, model, tokenizer)
elif args.eval_task == "phone_book":
    str_acc_mean_list, str_acc_std_list = phone_book_evaluation(args, model, tokenizer)
elif args.eval_task == "squad":
    em_list, f1_list, std_em_list, std_f1_list = squad_evaluation(args, model, tokenizer)
else:
    raise ValueError(f"Non-valid evaluation task {args.eval_task}")


print("DONE")
