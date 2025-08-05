import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from attention_new import llama_modify
from constants import INSTRUCTION_TEMPLATE, POPE_COCO_PATH, POPE_AOKVQA_PATH, POPE_GQA_PATH
from eval_data_loader import POPEChatDataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from tqdm import tqdm
from contrast import ContrastLogits
from transformers.generation.logits_process import LogitsProcessorList


def setup_seeds():
    seed = 2025

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--attn_ans", action="store_true", help="save attention each generated tokens, but the file will become very large")
parser.add_argument("--pope-dataset", type=str, help='coco, aokvqa, or gqa.')
parser.add_argument("--pope-type", type=str, help='random, popular, or adversarial.')
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
    default="coco/val2014/",  # images (gqa)
    help="data path",
)
parser.add_argument("--batch-size", type=int, default=1)

parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--use_instruction_attn", action="store_true")
parser.add_argument("--use_img_attn", action="store_true")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.93)
parser.add_argument("--start_img_layer", type=int, default=10)
parser.add_argument("--end_img_layer", type=int, default=30)
parser.add_argument("--start_instruction_layer", type=int, default=5)
parser.add_argument("--end_instruction_layer", type=int, default=20)
parser.add_argument("--use_contrast", action="store_true")
parser.add_argument("--gamma", type=float, default=1.2)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--ngram", type=int, default=4)  # no_repeat_ngram
args = parser.parse_known_args()[0]

setup_seeds()

disable_torch_init()

model_loader = ModelLoader(args.model)

if args.pope_dataset == 'coco':
    POPE_CHAT_PATH = POPE_COCO_PATH
elif args.pope_dataset == 'aokvqa':
    POPE_CHAT_PATH = POPE_AOKVQA_PATH
elif args.pope_dataset == 'gqa':
    POPE_CHAT_PATH = POPE_GQA_PATH
else:
    raise NotImplementedError('not implemented pope dataset: {}'.format(args.pope_dataset))

args.pope_path = POPE_CHAT_PATH[args.pope_type]


pope_dataset = POPEChatDataSet(
    pope_path=args.pope_path,
    data_path=args.data_path,
    trans=model_loader.image_processor,
    model=args.model
)

pope_loader = torch.utils.data.DataLoader(
    pope_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=16,
    drop_last=False,
)

base_dir = "./results_generation/pope_" + args.pope_dataset + "/" + args.model
if not os.path.exists('results_generation'):
    os.makedirs('results_generation')
if not os.path.exists("./results_generation/pope_" + args.pope_dataset):
    os.makedirs("./results_generation/pope_" + args.pope_dataset)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# dump metric file
file_parts = [
    f"pope_eval", 
    "_with_attn" if args.attn_ans else "",
    f"_{args.pope_type}_tokens_{args.max_tokens}",
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
for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
    image = data["image"]  # [batch_size, 3, H, W]
    queries = np.array(data["query"])  # [batch_size, list[query_size]]
    label = torch.stack(data["label"])  # [batch_size, list[query_size]]
    image_id = data["image_id"]
    kwargs = {}

    round = label.size()[0]

    for idx in range(round):
        # query = np.array([queries[idx, :][0] + " Please answer with one word: Yes or No."])
        query = queries[idx, :].tolist()
        lal = label[idx, :].tolist()

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
                output_attentions=args.attn_ans,
                output_hidden_states=False,
                return_dict_in_generate=True,
                return_dict=True, 
                **kwargs,
            )  # torch.LongTensor (token_id) if return_dict_in_generate=False
            # sequences, scores, attentions, hidden_states, past_key_values if return_dict_in_generate=True

        #### for attention statistics
        # Confirmatory Object Hallucination, attention of token ("yes" or "no")
        if args.attn_ans:
            num_generated = 0  # num of generated tokens
            for i, token in enumerate(outputs.attentions):
                num_generated += 1
                for j, layer in enumerate(token):
                        attn_instruction = torch.sum(layer[:, :, -1, model_loader.instruction_start_idx:model_loader.instruction_end_idx], dim=-1)
                        attn_img = torch.sum(layer[:, :, -1, model_loader.img_start_idx:model_loader.img_end_idx], dim=-1)
                        attn_instruction_wo_img = attn_instruction - attn_img
                        # [batch_size=1, head_size=32]
                        if j == 0:
                            token_attn_img = attn_img.unsqueeze(dim=1)
                            token_attn_text =  attn_instruction_wo_img.unsqueeze(dim=1)
                        else:
                            token_attn_img = torch.cat([token_attn_img, attn_img.unsqueeze(dim=1)], dim=1)
                            token_attn_text = torch.cat([token_attn_text, attn_instruction_wo_img.unsqueeze(dim=1)], dim=1)
                        # [batch_size=1, layersize=32, head_size=32]
                if i == 0:
                    generated_attn_img = token_attn_img.unsqueeze(dim=1)
                    generated_attn_text = token_attn_text.unsqueeze(dim=1)
                else:
                    generated_attn_img = torch.cat([generated_attn_img, token_attn_img.unsqueeze(dim=1)], dim=1)
                    generated_attn_text = torch.cat([generated_attn_text, token_attn_text.unsqueeze(dim=1)], dim=1)
        #### for attention statistics

        output_text = model_loader.decode(outputs[0], no_repeat_ngram_size=args.ngram)  # or outputs.sequences

        for i in range(len(output_text)):

            #### for attention statistics
            if args.attn_ans:
                word_ids = model_loader.words_to_ids(output_text[i], outputs[0][i], num_generated)
                assert int(word_ids[-1]) == num_generated, 'word tokenization in model_loader.words_to_ids is different from the model!'
            #### for attention statistics

            with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:

                #### for attention statistics
                if args.attn_ans:
                    json.dump(
                        {
                            "image_id": image_id[i],
                            "query": query[i],
                            "label": lal[i],
                            "ans": output_text[i],
                            "question": questions[i],
                            # "sys_start_idx": model_loader.sys_start_idx, 
                            # "sys_end_idx": model_loader.sys_end_idx, 
                            # "instruction_start_idx": model_loader.instruction_start_idx, 
                            # "instruction_end_idx": model_loader.instruction_end_idx, 
                            # "img_start_idx": model_loader.img_start_idx, 
                            # "img_end_idx": model_loader.img_end_idx, 
                            "word_ids": word_ids,
                            "generated_attn_img": generated_attn_img[i].tolist(), 
                            "generated_attn_text": generated_attn_text[i].tolist() 
                        },
                        f,
                    )
                    f.write("\n")
                #### for attention statistics

                else:
                    json.dump(
                        {
                            "image_id": image_id[i],
                            "query": query[i],
                            "label": lal[i],
                            "ans": output_text[i],
                            "question": questions[i],
                        },
                        f,
                    )
                    f.write("\n")
