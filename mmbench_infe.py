import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from attention_new import llama_modify
from constants import INSTRUCTION_TEMPLATE
from eval_data_loader import MMBenchDataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessorList


def setup_seeds():
    seed = 2025

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="MMBench evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="mmbench_dataset/cn", 
    help="data path",
)
parser.add_argument("--batch-size", type=int, default=1)

parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--use_instruction_attn", action="store_true")
parser.add_argument("--use_img_attn", action="store_true")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--start_img_layer", type=int, default=10)
parser.add_argument("--end_img_layer", type=int, default=30)
parser.add_argument("--start_instruction_layer", type=int, default=5)
parser.add_argument("--end_instruction_layer", type=int, default=20)
parser.add_argument("--use_contrast", action="store_true")
parser.add_argument("--gamma", type=float, default=1.1)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--ngram", type=int, default=4)  # no_repeat_ngram
args = parser.parse_known_args()[0]

setup_seeds()

disable_torch_init()

model_loader = ModelLoader(args.model)

mmbench_dataset = MMBenchDataSet(
    data_path=os.path.join(args.data_path, 'mmbench.jsonl'),
    trans=model_loader.image_processor,
    model=args.model
)

mmbench_loader = torch.utils.data.DataLoader(
    mmbench_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=16,
    drop_last=False,
)

base_dir = "./results_generation/mmbench/" + args.model + '/' + os.path.basename(args.data_path)
if not os.path.exists('results_generation'):
    os.makedirs('results_generation')
if not os.path.exists("./results_generation/mmbench"):
    os.makedirs("./results_generation/mmbench")
if not os.path.exists("./results_generation/mmbench/" + args.model):
    os.makedirs("./results_generation/mmbench/" + args.model)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# dump metric file
file_parts = [
    f"mmbench_eval", 
    f"_{os.path.basename(args.data_path)}_tokens_{args.max_tokens}",
    "_sample" if args.sample else "",
    f"_beams_{args.beam}" if args.beam != 1 else "",
    f"_instruction_{args.alpha}_{args.start_instruction_layer}_{args.end_instruction_layer}" if args.use_instruction_attn else "", 
    f"_img_{args.beta}_{args.start_img_layer}_{args.end_img_layer}" if args.use_img_attn else "", 
    f"_contrast_{args.gamma}" if args.use_contrast else "", 
    f"_penalty_{args.ngram}" if args.ngram > 0 else "",
]

file_name = "".join(file_parts)
template = INSTRUCTION_TEMPLATE[args.model]

model_loader.llm_model.eval()
for batch_id, data in tqdm(enumerate(mmbench_loader), total=len(mmbench_loader)):
    index = data["index"]
    image = data["image"]  # [batch_size, 3, H, W]
    query = data["query"]  # [batch_size, 1]
    label = data["label"]  # [batch_size, 1]

    # prepare inputs for model
    # no image query with <ImageHere>, {"image": image_tensor, "input_ids": no image query token with -200}
    questions, kwargs = model_loader.prepare_inputs_for_model(
        template, query, image
    )
    llama_modify(
        model_loader.llm_model,
        args.start_img_layer,
        args.end_img_layer,
        args.start_instruction_layer,
        args.end_instruction_layer,
        args.use_img_attn,
        args.use_instruction_attn, 
        args.beta, 
        args.alpha,
        model_loader.img_start_idx,
        model_loader.img_end_idx,
        model_loader.instruction_start_idx,
        model_loader.instruction_end_idx,
    )

    logits_processor = (
        LogitsProcessorList([model_loader.init_contrast_processor(args.gamma, min(args.start_img_layer, args.start_instruction_layer), 
                                    max(args.end_img_layer, args.end_instruction_layer))])
        if args.use_contrast
        else LogitsProcessorList([])
    )

    with torch.inference_mode():
        outputs = model_loader.llm_model.generate(
            do_sample=args.sample,
            max_new_tokens=args.max_tokens,
            # max_new_tokens=1, 
            use_cache=True,
            num_beams=args.beam, 
            logits_processor=logits_processor, 
            stopping_criteria=model_loader.stopping_criteria, 
            output_hidden_states=False,
            return_dict_in_generate=True,
            return_dict=True, 
            **kwargs,
        )  # torch.LongTensor (token_id) if return_dict_in_generate=False
        # sequences, scores, attentions, hidden_states, past_key_values if return_dict_in_generate=True


    output_text = model_loader.decode(outputs[0], no_repeat_ngram_size=args.ngram)  # or outputs.sequences

    for i in range(len(output_text)):
        with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
            json.dump(
                {
                    "index": int(index[i]),
                    "query": query[i],
                    "label": label[i],
                    "ans": output_text[i],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
