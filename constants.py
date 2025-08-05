IMAGE_TOKEN_INDEX = -200 # LLaVA
IMAGE_TOKEN_LENGTH = 576  

MINIGPT4_IMAGE_TOKEN_LENGTH = 32  # resampler

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
SHIKRA_IMAGE_TOKEN_LENGTH = 256  # SHIKRA
SHIKRA_IMG_START_TOKEN = 32001
SHIKRA_IMG_END_TOKEN = 32002

IMAGE_PAD_TAG = '<imgpad>' # Qwen-VL-Chat
IMG_TOKEN_SPAN = 256
QWEN_IMG_START_TOKEN = 151857
QWEN_IMG_END_TOKEN = 151858


INSTRUCTION_TEMPLATE = {
    "qwen-vl-chat": "<|im_start|>user\n<imagehere>\n<question><|im_end|>\n<|im_start|>assistant\n",
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
}

INSTRUCTION_TEMPLATE_NO_IMG = {
    "minigpt4": "###Human: <question> ###Assistant:",
    "shikra": "USER: <question> ASSISTANT:",
    "llava-1.5": "USER: <question> ASSISTANT:",
}

SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

POPE_COCO_PATH = {
    "random": "pope_dataset/pope_coco/pope_coco_random.json",
    "popular": "pope_dataset/pope_coco/pope_coco_popular.json",
    "adversarial": "pope_dataset/pope_coco/pope_coco_adversarial.json",
}

POPE_AOKVQA_PATH = {
    "random": "pope_dataset/pope_aokvqa/pope_aokvqa_random.json",
    "popular": "pope_dataset/pope_aokvqa/pope_aokvqa_popular.json",
    "adversarial": "pope_dataset/pope_aokvqa/pope_aokvqa_adversarial.json",
}

POPE_GQA_PATH = {
    "random": "pope_dataset/pope_gqa/pope_gqa_random.json",
    "popular": "pope_dataset/pope_gqa/pope_gqa_popular.json",
    "adversarial": "pope_dataset/pope_gqa/pope_gqa_adversarial.json",
}
