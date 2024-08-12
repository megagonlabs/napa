from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModel


class CTC:
    def __init__(self,
                 model_name: str = "roberta-large",
                 device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    @torch.no_grad()
    def __call__(self,
                 src: List[str],
                 hyp: str,
                 ref: Union[str, List[str]]):
        self.model.eval()
        src_tensor = self.tokenizer(src, return_tensors="pt", padding=True).to(self.device)
        src_emb = self.model(**src_tensor).last_hidden_state
        src_emb.div_(torch.norm(src_emb, dim=-1, keepdim=True))
        src_mask = src_tensor.attention_mask.clone()
        src_mask[src_tensor.input_ids == self.tokenizer.bos_token_id] = 0
        src_mask[src_tensor.input_ids == self.tokenizer.eos_token_id] = 0
        src_emb = src_emb[src_mask.bool()]

        hyp_tensor = self.tokenizer(hyp, return_tensors="pt", ).to(self.device)
        hyp_emb = self.model(**hyp_tensor).last_hidden_state[0, 1:-1]
        hyp_emb.div_(torch.norm(hyp_emb, dim=-1, keepdim=True))

        consistency = (hyp_emb @ src_emb.T).max(dim=-1).values.mean().item()

        ref_tensor = self.tokenizer(ref, return_tensors="pt", padding=True).to(self.device)  # Multiple reference
        ref_emb = self.model(**ref_tensor).last_hidden_state
        ref_emb.div_(torch.norm(ref_emb, dim=-1, keepdim=True))
        ref_mask = ref_tensor.attention_mask.clone()
        ref_mask[ref_tensor.input_ids == self.tokenizer.bos_token_id] = 0
        ref_mask[ref_tensor.input_ids == self.tokenizer.eos_token_id] = 0
        ref_emb = ref_emb[ref_mask.bool()]
        alignments = (ref_emb @ hyp_emb.T).max(dim=-1).values
        i = 0
        s = []
        ref_length = ref_mask.sum(dim=1)
        for r_len in ref_length:
            s.append(alignments[i:i + r_len].mean())
            i += r_len

        relevance = (max(s) * consistency).item()

        return consistency, relevance
