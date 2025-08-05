import os
from collections import namedtuple

import torch
import yaml
from constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    IMAGE_TOKEN_LENGTH,
    MINIGPT4_IMAGE_TOKEN_LENGTH,
    SHIKRA_IMAGE_TOKEN_LENGTH,
    SHIKRA_IMG_END_TOKEN,
    SHIKRA_IMG_START_TOKEN, 
    SYSTEM_MESSAGE, 
)
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from minigpt4.common.eval_utils import init_model
from shikra.models.builder.build_shikra import load_pretrained_shikra
import nltk
from transformers import StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from contrast import ContrastLogits
import re

# from model_loader import load_pretrained_qwen_vl
# tokenizer, model, _, _ = load_pretrained_qwen_vl('models/Qwen-VL-Chat')

def load_pretrained_qwen_vl(model_path, device_map="cuda", bf16=False, fp16=True, load_in_4bit=False):
    '''
    # special tokens
    IMSTART='<|im_start|>' # 151644
    IMEND='<|im_end|>' # 151645
    ENDOFTEXT='<|endoftext|>' # 151643
    '''
    # bf16 or fp16=True -> device_map="auto"
    
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=['lm_head', 'attn_pool.attn']
            )
    else:
        quantization_config = None
    
    kwargs = {"device_map": device_map, 
              "bf16": bf16,
              "fp16": fp16, 
              "quantization_config": quantization_config}

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)  # generation_config.json

    tokenizer.eos_token_id = model.generation_config.eos_token_id
    tokenizer.pad_token_id =  model.generation_config.pad_token_id
    # tokenizer.im_start_id: 151644 '<|im_start|>'
    # tokenizer.im_end_id: 151645 '<|im_end|>'

    def image_processor(x): # image path
        return x

    return tokenizer, model, image_processor, model


def load_model_args_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    ModelArgs = namedtuple("ModelArgs", data["ModelArgs"].keys())
    TrainingArgs = namedtuple("TrainingArgs", data["TrainingArgs"].keys())

    model_args = ModelArgs(**data["ModelArgs"])
    training_args = TrainingArgs(**data["TrainingArgs"])

    if data["load_in_8bit"]:
        quantization_kwargs = dict(
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            )
        )
    else:
        quantization_kwargs = dict()

    return model_args, training_args, quantization_kwargs


def load_llava_model(model_path):
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    return tokenizer, model, image_processor, model


def load_minigpt4_model(cfg_path):
    cfg = MiniGPT4Config(cfg_path)
    model, vis_processor = init_model(cfg)
    # TODO:
    # model.eval()
    return model.llama_tokenizer, model, vis_processor, model.llama_model


def load_shikra_model(yaml_path):
    model_args, training_args, quantization_kwargs = load_model_args_from_yaml(yaml_path)
    model, preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs)

    # if not getattr(model, 'is_quantized', False): 
    #     model.to(dtype=torch.float16, device=torch.device('cuda'))
    # if not getattr(model.model.vision_tower[0], 'is_quantized', False): 
    #     model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))

    return preprocessor["text"], model, preprocessor["image"], model


class MiniGPT4Config:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.options = None
        

def prepare_llava_inputs(template, query, image, tokenizer):
    # template = SYSTEM_MESSAGE + ' ' + template
    image_tensor = image["pixel_values"][0]
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_system = (
        tokenizer(
            SYSTEM_MESSAGE,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id 
    )

    sys_start_idx = 1
    sys_end_idx = sys_start_idx + len(token_system[0])

    instruction_start_idx = sys_end_idx
    instruction_end_idx = instruction_start_idx + len(token_before[0]) + IMAGE_TOKEN_LENGTH + len(token_after[0])

    img_start_idx = instruction_start_idx + len(token_before[0])
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH

    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_system, token_before, image_token, token_after], dim=1)

    kwargs = {}
    kwargs["images"] = image_tensor.half()
    kwargs["input_ids"] = input_ids

    # no image query with <ImageHere>, img_start_idx, img_end_idx, {"image": image_tensor, "input_ids": no image query token with -200}
    return qu, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs

def prepare_minigpt4_inputs(template, query, image, model):
    # no SYSTEM_MESSAGE
    image_tensor = image.to("cuda")
    qu = [template.replace("<question>", q) for q in query]
    # ['###Human: <Img><ImageHere></Img> Please help me describe the image in detail. ###Assistant:']
    # tokenizer.encode: [batch_size, 25]
    batch_size = len(query)

    img_embeds, atts_img = model.encode_img(image_tensor.to("cuda"))
    # torch.Size([1, 32, 4096]), torch.Size([1, 32])

    # '###Human: <Img>': [batch_size, 7]
    # '</Img> Please help me describe the image in detail. ###Assistant:': [batch_size, 16]
    inputs_embeds, attention_mask = model.prompt_wrap(
        img_embeds=img_embeds, atts_img=atts_img, prompts=qu
    )
    # torch.Size([1, 55, 4096]) torch.Size([1, 55])

    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device=inputs_embeds.device)
        * model.llama_tokenizer.bos_token_id
    )
    bos_embeds = model.embed_tokens(bos)
    # torch.Size([1, 1, 4096])
    atts_bos = attention_mask[:, :1]

    # '<', 'Img', '><', 'Image', 'Here', '></', 'Img', '>'
    # tensor([[  529, 25518,  5299,  2940, 10605,  2565, 25518, 29958]])
    # '<', 'Image', 'Here', '>'
    # tensor([[  529,  2940, 10605, 29958]])

    instruction_start_idx = 1
    instruction_end_idx = instruction_start_idx + attention_mask.size(dim=-1)

    # add 1 for bos token
    img_start_idx = (
        model.llama_tokenizer(
            qu[0].split("<ImageHere>")[0], 
            return_tensors="pt", 
            padding="longest", 
            add_special_tokens=False
        )
        .input_ids.shape[-1]
        + 1
    )
    img_end_idx = img_start_idx + MINIGPT4_IMAGE_TOKEN_LENGTH

    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

    kwargs = {}
    kwargs["inputs_embeds"] = inputs_embeds
    kwargs["attention_mask"] = attention_mask

    return qu, img_start_idx, img_end_idx, instruction_start_idx, instruction_end_idx, kwargs


def prepare_shikra_inputs(template, query, image, tokenizer): 
    image_tensor = image["pixel_values"][0]

    # "<im_patch>" * 256
    # 32000: '<im_patch>'
    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * SHIKRA_IMAGE_TOKEN_LENGTH
    qu = [template.replace("<question>", q) for q in query]
    qu = [p.replace("<ImageHere>", replace_token) for p in qu]

    token_system = tokenizer(
            SYSTEM_MESSAGE,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to("cuda").input_ids
    # 30 tokens

    token_input = tokenizer(
        qu, 
        return_tensors="pt", 
        padding="longest", 
        add_special_tokens=False
    ).to("cuda").input_ids

    # "USER: <im_start>"
    # [3148, 1001, 29901, 29871, 32001]
    # 5 tokens

    # "<im_end> Please help me describe the image in detail. ASSISTANT:"
    # [32002, 29871, 3529, 1371, 592, 8453, 278, 1967, 297, 9493, 29889, 319, 1799, 9047, 13566, 29901]
    # 16 tokens

    bs = len(query)
    bos = torch.ones([bs, 1], dtype=torch.int64, device="cuda") * tokenizer.bos_token_id
    input_ids = torch.cat([bos, token_system, token_input], dim=1)

    sys_start_idx = 1
    sys_end_idx = sys_start_idx + len(token_system[0])

    instruction_start_idx = sys_end_idx
    instruction_end_idx = instruction_start_idx + len(token_input[0])

    img_start_idx = torch.where(input_ids == SHIKRA_IMG_START_TOKEN)[1][0].item() + 1 # 32001: <im_start>
    img_end_idx = torch.where(input_ids == SHIKRA_IMG_END_TOKEN)[1][0].item() # 32002: <im_end>

    kwargs = {}
    kwargs["input_ids"] = input_ids
    kwargs["images"] = image_tensor.to("cuda")

    # idx: 36 292 1 31 31 308
    return qu, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs

def prepare_qwenvlchat_inputs(template, query, image, tokenizer):
    '''
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <img>/home/zhenghh8/hallucination/PAI/coco/val2014/COCO_val2014_000000000042.jpg</img>
    Describe it.<|im_end|>
    <|im_start|>assistant

    '''

    # image_replace = IMAGE_PAD_TAG * IMG_TOKEN_SPAN # 256 * <imgpad>
    qu = [template.replace('<question>', q) for q in query]
    chunks = [p.split('<imagehere>') for p in qu]

    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    system_raw = '<|im_start|>system\n' + SYSTEM_MESSAGE + '<im_end>\n'
    token_system = tokenizer.encode(
            system_raw,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to("cuda")

    token_before = tokenizer(
        chunk_before, 
        return_tensors="pt", 
        padding="longest", 
        add_special_tokens=False
    ).to("cuda").input_ids

    image = ['<img>' + img + '</img>' for img in image]  # 256 + 2
    token_img = tokenizer(
        image, 
        return_tensors="pt", 
        padding="longest", 
        add_special_tokens=False
    ).to("cuda").input_ids

    token_after = tokenizer(
        chunk_after, 
        return_tensors="pt", 
        padding="longest", 
        add_special_tokens=False
    ).to("cuda").input_ids

    input_ids = torch.cat([token_system, token_before, token_img, token_after], dim=1)

    # include special token 
    sys_start_idx = 0
    sys_end_idx = sys_start_idx + len(token_system[0])

    instruction_start_idx = sys_end_idx
    instruction_end_idx = instruction_start_idx + len(token_before[0]) + len(token_img[0]) + len(token_after[0])

    img_start_idx = instruction_start_idx + len(token_before[0])
    img_end_idx = instruction_start_idx + len(token_before[0]) + len(token_img[0])

    kwargs = {}
    kwargs["input_ids"] = input_ids

    return qu, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs



# Example usage:
# prepare_inputs_for_model(args, image, model, tokenizer, kwargs)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop):] == stop).item():
                return True
        return False
    
def stop_word_to_criteria(stop_word_ids):
    stop_words_ids = [torch.tensor(ids).to(device='cuda') for ids in stop_word_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria


class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.load_model()

    def load_model(self):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser("/path/to/llava-v1.5-7b")
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )
            self.stopping_criteria = stop_word_to_criteria(stop_word_ids=[[2]])  # </s>


        elif self.model_name == "minigpt4":
            cfg_path = "./minigpt4/eval_config/minigpt4_eval.yaml"
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_minigpt4_model(cfg_path)
            )
            self.stopping_criteria = stop_word_to_criteria(stop_word_ids=[[835], [2277, 29937]])  # [###], ['#', '##']

        elif self.model_name == "shikra":
            yaml_path = "./shikra/config/config.yml"
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_shikra_model(yaml_path)
            )
            self.stopping_criteria = stop_word_to_criteria(stop_word_ids=[[2]])  # </s>
        
        elif self.model_name == "qwen-vl-chat":
            model_path = os.path.expanduser("/path/to/Qwen-VL-Chat")

            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_pretrained_qwen_vl(model_path)
            )
            
            self.stopping_criteria = stop_word_to_criteria(stop_word_ids=[[151643, 151644, 151645]])
            # IMSTART='<|im_start|>' # 151644
            # IMEND='<|im_end|>' # 151645
            # ENDOFTEXT='<|endoftext|>' # 151643

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def prepare_inputs_for_model(self, template, query, image):
        if self.model_name == "llava-1.5":
            # no image query with <ImageHere>, img_start_idx, img_end_idx, {"image": image_tensor, "input_ids": no image query token with -200}
            questions, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs = prepare_llava_inputs(
                template, query, image, self.tokenizer
            )
        elif self.model_name == "minigpt4":
            questions, img_start_idx, img_end_idx, instruction_start_idx, instruction_end_idx, kwargs = prepare_minigpt4_inputs(
                template, query, image, self.vlm_model
            )
            sys_start_idx, sys_end_idx = None, None
        elif self.model_name == "shikra":
            questions, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs = prepare_shikra_inputs(
                template, query, image, self.tokenizer
            )
        elif self.model_name == 'qwen-vl-chat':
            questions, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs = prepare_qwenvlchat_inputs(
                template, query, image, self.tokenizer
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx
        self.sys_start_idx = sys_start_idx
        self.sys_end_idx = sys_end_idx
        self.instruction_start_idx = instruction_start_idx
        self.instruction_end_idx = instruction_end_idx

        self.input = kwargs

        # no image query, {"image": image_tensor, "input_ids": no image query token}
        return questions, kwargs
    
    def init_contrast_processor(self, gamma=1.1, start_layer=0, end_layer=32):
        logits_processor = ContrastLogits(self.input, gamma, self.llm_model, start_layer=start_layer, end_layer=end_layer)

        return logits_processor
    
    def decode(self, output_ids, no_repeat_ngram_size=None):
        # get outputs
        if self.model_name == "llava-1.5":
            # replace image token by pad token
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=False
            )

            output_text = [
                text.split("ASSISTANT: ")[-1].strip() 
                for text in output_text
            ]
            # [319, 1799, 9047, 13566, 29901]

        elif self.model_name == "minigpt4":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [
                text.split("###")[0].split("Assistant:")[-1].strip() # remove the stop sign '###'
                for text in output_text
            ]

        elif self.model_name == "shikra":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        elif self.model_name == "qwen-vl-chat":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=False
            )
            output_text = [text.split('assistant\n')[-1].strip() for text in output_text]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        def remove_repeats(text, ngram_size=4):
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            filtered_sentences = []
            all_previous_ngrams = set()

            for sentence in sentences:
                if not sentence:
                    continue
                
                words = sentence.split()
                if len(words) < ngram_size:
                    filtered_sentences.append(sentence)
                
                has_repeat = False
                current_sentence_ngrams = []
                
                for i in range(len(words) - ngram_size + 1):
                    current_ngram = tuple(words[i:i+ngram_size])
                    current_sentence_ngrams.append(current_ngram)
                    if current_ngram in all_previous_ngrams:
                        has_repeat = True
                        break
                
                if not has_repeat:
                    filtered_sentences.append(sentence)
                    all_previous_ngrams.update(current_sentence_ngrams)
            
            return ' '.join(filtered_sentences)

        if no_repeat_ngram_size:
            for idx, text in enumerate(output_text):
                output_text[idx] = remove_repeats(text, no_repeat_ngram_size)

        return output_text

    def words_to_ids(self, output_text, output_ids, num_generated):
        one_token_word = ' newline '
        # words_to_token_ids
        
        if self.model_name == 'qwen-vl-chat':
            import re
            def is_english_or_hyphen(s):
                return bool(re.match(r'^[a-zA-Z]+(-[a-zA-Z]+)*$', s))
            # words = self.tokenizer.tokenize(output_text)
            # words = [word.decode('utf-8') if type(word) == bytes else word for word in words]
            special_one_qwen = ['."\n\n', '.\n\n', '\n\n', '.\n', '\n', '<|im_start|>', '<|im_end|>', '<|endoftext|>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '."', ',"', '".', 'cannot']
            for special in special_one_qwen:
                output_text = output_text.replace(special, one_token_word)

            words = nltk.word_tokenize(output_text)
            for i, word in enumerate(words):  # qwen's encoding logic
                # special example
                # "holding a Wii remote"
                # if word.islower() and word.isalpha(): 
                #     words[i] = ' ' + word
                if i == 0:
                    continue
                if is_english_or_hyphen(word) and words[i-1] not in ['"', '``', ]:
                    words[i] = ' ' + word

        elif self.model_name == 'llava-1.5':
            special_one_llava = ['\n', '."', '</s>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',"']
            special_two_llava = ['Additionally', 'Inside', 'Allen']
            special_three_llava = ['Experts', 'Several', 'Metropolitan', 'Houston']

            for special in special_one_llava:
                output_text = output_text.replace(special, one_token_word)
            special_llama = ['newline nd']  # '2nd'
            for special in special_llama:
                output_text = output_text.replace(special, 2 * one_token_word)
            words = nltk.word_tokenize(output_text)
        else:
            NotImplementedError
        i = 0
        ids = []
        # if self.model == 'minigpt4':  # first generated token id is 0 (<unk>)
        #     i += 1 
        # attention generated in minigpt4 auto skip the first token
        ids.append(i)
        token_len = len(output_ids)
        for word in words:
            if self.model_name == 'qwen-vl-chat':
                # >>> tokenizer.encode(' Topology')
                # [6909, 2449]
                # >>> tokenizer.encode('Topology')
                # [60954]
                if output_ids[token_len-num_generated+i] == 220:  # qwen space: ' '
                    i += 1
                if word != 'newline': 
                    if output_ids[token_len-num_generated+i-1] in [4710, 382]:  # ' \n\n', '.\n\n'
                        word = word.strip()
                    token = self.tokenizer.encode(word, 
                                        return_tensors='pt', 
                                        padding='longest', 
                                        add_special_tokens=False)
                    i += token.size(dim=-1)
                    ids.append(i)
                else:
                    i += 1
                    ids.append(i)
            elif self.model_name == 'llava-1.5':
                # llava-1.5
                # special example
                # >>> tokenizer.decode([13])
                # '\n'
                # >>> tokenizer.encode('\n')
                # [29871, 13]
                # >>> tokenizer.decode([1213])
                # '."'
                # >>> tokenizer.encode('."')
                # [869, 29908]
                # >>> tokenizer.decode([1252, 546, 1372])
                # 'Experts'
                # >>> tokenizer.encode('Experts')
                # [28224, 1372]
                # >>> tokenizer.decode([2528, 17658])
                # 'Additionally'
                # >>> tokenizer.encode('Additionally')
                # [19814]
                # >>> tokenizer.decode([29903,  1310,   284])
                # 'Several'
                # >>> tokenizer.encode('Several')
                # [21882]
                # >>> tokenizer.decode([10095, 10759,  8929])
                # 'Metropolitan'
                # >>> tokenizer.encode('Metropolitan')
                # [28788]
                # >>> tokenizer.decode([797,  2975])
                # 'Inside'
                # >>> tokenizer.encode('Inside')
                # [22804]
                # >>> tokenizer.decode([1699])
                # ',"'
                # >>> tokenizer.encode(',"')
                # [29871, 1699]
                # >>> tokenizer.decode([3596, 264])
                # 'Allen'
                # >>> tokenizer.encode('Allen')
                # [16092]
                # >>> tokenizer.decode([29950, 283,  7352])
                # 'Houston'
                # >>> tokenizer.encode('Houston')
                # [24327]

                if output_ids[token_len-num_generated+i] == 29871:  # LLAMA SPIECE_UNDERLINE: ''
                    i += 1
                if word != 'newline': 
                    if word in special_two_llava and output_ids[token_len-num_generated+i] in [2528, 797, 3596]:
                        i += 2
                        ids.append(i)
                    elif word in special_three_llava and output_ids[token_len-num_generated+i] in [1252, 29903, 10095, 29950]:
                        i += 3
                        ids.append(i)
                    else:
                        token = self.tokenizer.encode(word, 
                                            return_tensors='pt', 
                                            padding='longest', 
                                            add_special_tokens=False)
                        i += token.size(dim=-1)
                        ids.append(i)
                else:
                    i += 1
                    ids.append(i)
            else:
                NotImplementedError
        return ids
