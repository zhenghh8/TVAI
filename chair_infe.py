import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from attention_new import llama_modify
from constants import INSTRUCTION_TEMPLATE
from eval_data_loader import COCODataSet
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


parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--attn_ans", action="store_true", help="save attention each generated tokens, but the file will become very large")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
# TODO
parser.add_argument(
    "--data-path",
    type=str,
    default="./coco/val2014/",
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
parser.add_argument("--gamma", type=float, default=1.1)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--ngram", type=int, default=4)  # no_repeat_ngram
args = parser.parse_known_args()[0]

setup_seeds()

disable_torch_init()

model_loader = ModelLoader(args.model)

base_dir = "./results_generation/chair"
if not os.path.exists('results_generation'):
    os.makedirs('results_generation')
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
if not os.path.exists(base_dir + '/' + args.model):
    os.makedirs(base_dir + '/' + args.model)

coco_dataset = COCODataSet(data_path=args.data_path, trans=model_loader.image_processor, model=args.model)
# qwen-vl-chat re-initialize CUDA in image_processor (Cannot re-initialize CUDA in forked subprocess)
coco_loader = torch.utils.data.DataLoader(
    coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
)

file_parts = [
    "chair_eval",
    "_with_attn" if args.attn_ans else "",
    f"_tokens_{args.max_tokens}",
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
for batch_id, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):
    # if batch_id < 354:  # keep on running in batch # (id: # - 1)
    #     continue
    if batch_id == 500:
        break
    img_id = data["img_id"]
    image = data["image"]

    batch_size = img_id.shape[0]

    # chair query
    query = ["Please help me describe the image in detail."] * batch_size

    questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)

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

    # check weight and data type
    # for name, param in model_loader.llm_model.named_parameters():
    #     print(f"{name}: {param.dtype} {param.device}")
    # print(kwargs["input_ids"].dtype, kwargs["images"].dtype)
    # raise NotImplementedError

    # # https://github.com/huggingface/transformers/issues/34304
    # pre_expansion_embeddings = model_loader.llm_model.lm_head.weight.data
    # mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    # n = pre_expansion_embeddings.size()[0]
    # sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    # dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # num_new_tokens = 1 # 1 for special `image` token
    # lm_head_weights = model_loader.llm_model.lm_head.weight

    # new_token_embedding = torch.stack(tuple(dist.sample() for _ in range(num_new_tokens)), dim=0).to(device=lm_head_weights.device, dtype=lm_head_weights.dtype)
    # lm_head_weights.data = torch.cat([lm_head_weights.data, new_token_embedding], dim=0)
    # lm_head_weights.num_embeddings = lm_head_weights.data.shape[0]


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
            # repetition_penalty=1.1,  # https://github.com/huggingface/transformers/issues/34304
            # no_repeat_ngram_size=3, 
            **kwargs,
        )
        
    # check
    # k = 0
    # print(model_loader.tokenizer.eos_token_id)
    # print(outputs[0])
    # for i in outputs.attentions:
    #     k += 1
    # print(k)
    # output_ids = outputs[0].clone()
    # output_ids[output_ids == -200] = torch.tensor(0, dtype=output_ids.dtype, device=output_ids.device)
    # print(model_loader.tokenizer.batch_decode(output_ids, skip_special_tokens=False))
    # raise NotImplementedError

    #### for attention statistics
    # Conjured Object Hallucination, attention of object tokens
    if args.attn_ans:
        num_generated = 0  # num of generated tokens
        # declarative object hallucination
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
            # [batch_size=1, generated_size, layersize=32, head_size=32]
    #### for attention statistics
    
    # print(k)
    # print(outputs[0])
    output_text = model_loader.decode(outputs[0], no_repeat_ngram_size=args.ngram)  # or outputs.sequences
    # print(output_text)
    # raise NotImplementedError

    for i in range(len(output_text)):

        #### for attention statistics
        if args.attn_ans:
            word_ids = model_loader.words_to_ids(output_text[i], outputs[0][i], num_generated)
            assert int(word_ids[-1]) == num_generated, 'word tokenization in model_loader.words_to_ids is different from the model!'
        #### for attention statistics

        with open(os.path.join(base_dir, args.model, file_name + ".jsonl"), "a") as f:
            
            #### for attention statistics
            if args.attn_ans:
                json.dump({"image_id": int(img_id[i]), 
                            "caption": output_text[i], 
                            "word_ids": word_ids,
                            "generated_attn_img": generated_attn_img[i].tolist(), 
                            "generated_attn_text": generated_attn_text[i].tolist()
                            }, f)
                f.write("\n")
            #### for attention statistics

            else:
                json.dump({"image_id": int(img_id[i]), 
                            "caption": output_text[i]
                            }, f)
                f.write("\n")
