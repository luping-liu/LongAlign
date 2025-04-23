# Copyright 2024 Luping Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from torchvision import transforms
from transformers import AutoProcessor, AutoModel


processor = extract = None
processor_plus = extract_plus = None
_image_processor_reward = transforms.Compose([
    transforms.RandomCrop((504, 504)),  # for blocky blur
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])


def pick_score(image: torch.Tensor, text, do_ortho=False, accelerator=None):
    global processor, extract
    if extract is None:
        processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', torch_dtype=image.dtype)
        extract = AutoModel.from_pretrained('yuvalkirstain/PickScore_v1').to(image)
        extract.eval()
        extract.requires_grad_(False)

    pred_x_0_fea = extract.get_image_features(pixel_values=_image_processor_reward(image))
    pred_x_0_fea = pred_x_0_fea / pred_x_0_fea.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        text_inputs_pick = processor(
            text=text, padding="max_length", truncation=True, max_length=77, return_tensors="pt",
        ).to(image.device)
        text_embs_pick = extract.get_text_features(**text_inputs_pick)
        text_embs_pick = text_embs_pick / text_embs_pick.norm(dim=-1, keepdim=True)

        if do_ortho:
            ortho_tg = text_embs_pick.mean(dim=0)
            if accelerator is not None:
                ortho_tg = accelerator.gather(ortho_tg[None]).mean(dim=0)
            ortho_tg = (ortho_tg / ortho_tg.norm())[None]
            text_embs_pick_ = text_embs_pick - (1 - 0.3) * torch.sum(text_embs_pick * ortho_tg, dim=-1, keepdim=True) * ortho_tg
            # text_embs_pick_ = text_embs_pick_ + weight * ortho_tg
        else:
            text_embs_pick_ = text_embs_pick

    text_logits = text_embs_pick_ @ pred_x_0_fea.T

    return torch.diagonal(text_logits)
    # return F.cosine_similarity(text_embs_pick_.float(), pred_x_0_fea.float()).mean()


def clip_score(image: torch.Tensor, text, do_ortho=False, accelerator=None):
    global processor, extract
    if extract is None:
        processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', torch_dtype=image.dtype)
        extract = AutoModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K').to(image)
        extract.eval()
        extract.requires_grad_(False)

    pred_x_0_fea = extract.get_image_features(pixel_values=_image_processor_reward(image))
    pred_x_0_fea = pred_x_0_fea / pred_x_0_fea.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        text_inputs_pick = processor(
            text=text, padding="max_length", truncation=True, max_length=77, return_tensors="pt",
        ).to(image.device)
        text_embs_pick = extract.get_text_features(**text_inputs_pick)
        text_embs_pick = text_embs_pick / text_embs_pick.norm(dim=-1, keepdim=True)

        if do_ortho:
            ortho_tg = text_embs_pick.mean(dim=0)
            if accelerator is not None:
                ortho_tg = accelerator.gather(ortho_tg[None]).mean(dim=0)
            ortho_tg = (ortho_tg / ortho_tg.norm())[None]
            text_embs_pick_ = text_embs_pick - (1 - 0.3) * (torch.sum(text_embs_pick * ortho_tg, dim=-1, keepdim=True)) * ortho_tg
            # text_embs_pick_ = text_embs_pick_ + weight * ortho_tg
        else:
            text_embs_pick_ = text_embs_pick

    text_logits = text_embs_pick_ @ pred_x_0_fea.T

    return torch.diagonal(text_logits)
    # return F.cosine_similarity(text_embs_pick_.float(), pred_x_0_fea.float()).mean()


weight = 0.0
def dense_score(image: torch.Tensor, text_list, batch_ids, only_init=False, return_split=False,
                return_logit=False, do_ortho=False, accelerator=None):
    global processor, extract, weight
    if extract is None:
        weight = float(os.environ.get('REWEIGHT', 0.3))
        processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', torch_dtype=image.dtype)
        # token_ = os.environ.get('HF_TOKEN', None)
        extract = AutoModel.from_pretrained('luping-liu/Denscore').to(image)
        extract.eval()
        extract.requires_grad_(False)
        # print('logit_scale:', extract.logit_scale.item())

    if only_init:
        return

    pred_x_0_fea = extract.get_image_features(pixel_values=_image_processor_reward(image))
    pred_x_0_fea = pred_x_0_fea / pred_x_0_fea.norm(dim=-1, keepdim=True) 

    with torch.no_grad():
        text_inputs_pick = processor(
            text=text_list, padding="max_length", truncation=True, max_length=77, return_tensors="pt",
        ).to(image.device)
        text_embs_pick = extract.get_text_features(**text_inputs_pick)
        text_embs_pick = text_embs_pick / text_embs_pick.norm(dim=-1, keepdim=True)

        if not return_split:
            text_embs_pick_ = torch.zeros_like(text_embs_pick[:max(batch_ids) + 1])
            batch_ids = torch.tensor(batch_ids, device=image.device)
            for i in range(max(batch_ids) + 1):
                text_embs_pick_[i] = text_embs_pick[batch_ids == i].mean(dim=0)
            text_embs_pick = text_embs_pick_
        else:
            assert max(batch_ids) == 0

    if do_ortho:
        ortho_tg = text_embs_pick.mean(dim=0)
        if accelerator is not None:
            ortho_tg = accelerator.gather(ortho_tg[None]).mean(dim=0)
        ortho_tg = (ortho_tg / ortho_tg.norm())[None]
        # [0.8320, 0.8867,0.8281,0.8555, 0.8711, 0.9062, 0.8516, 0.8672]
        text_embs_pick_ = text_embs_pick -  (1 - weight) * torch.sum(text_embs_pick * ortho_tg, dim=-1, keepdim=True) * ortho_tg
        # text_embs_pick_ = text_embs_pick_ + weight * ortho_tg
    else:
        text_embs_pick_ = text_embs_pick

    text_logits = text_embs_pick_ @ pred_x_0_fea.T

    if return_logit:
        return text_logits
    else:
        return torch.diagonal(text_logits)
