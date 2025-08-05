import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

class ContrastLogits(LogitsProcessor):
    def __init__(
        self,
        input,
        guidance_scale,
        model,
        start_layer=0,
        end_layer=32,
    ):
        self.guidance_scale = guidance_scale
        self.input = input
        self.model = model
        self.out = None
        self.start_layer = start_layer
        self.end_layer = end_layer

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)

        if self.guidance_scale == 1:
            return scores
        
        for i in range(self.start_layer, self.end_layer + 1):
            self.model.model.layers[i].self_attn.use_contrast = True

        if self.out is None:
            self.out = self.model(
                use_cache=True, 
                **self.input
            )
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )

        for i in range(self.start_layer, self.end_layer + 1):
            self.model.model.layers[i].self_attn.use_contrast = False

        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)
        
        cutoff = torch.log(torch.tensor(0.1)) + scores.max(dim=-1, keepdim=True).values
        out = (
            self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        )

        c_logits = out.masked_fill(scores < cutoff, -float("inf"))
        return c_logits