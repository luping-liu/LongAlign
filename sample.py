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
import random
import re
import sys

import torch
import numpy
import platform
import argparse
from glob import glob
from PIL import Image

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, LCMScheduler
from transformers import AutoTokenizer, T5EncoderModel, CLIPTextModel

from tools import caption2embed
from modules.lora import monkeypatch_or_replace_lora_extended, collapse_lora, monkeypatch_remove_lora
from modules.adapters import TextAdapter

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="LongAlign Sampling")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--pretrained_decoder", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--token_length", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--ckpt_path", type=str, default="./model/LongSD/sd15-reward-3750.pt")
    parser.add_argument("--sample_method", type=str, default='unipc')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda", args.local_rank)
    weight_dtype = torch.bfloat16

    # args.ckpt_path = "./model/LongSDsd15-rewardori-lori3/sd15-reward-3750.pt"
    args.platform = platform.node()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(args.pretrained_decoder, subfolder="vae", torch_dtype=weight_dtype)
    vis = UNet2DConditionModel.from_pretrained(args.pretrained_decoder, subfolder="unet", torch_dtype=weight_dtype)

    tokenizer_clip = AutoTokenizer.from_pretrained(args.pretrained_decoder, subfolder="tokenizer",
                                                    torch_dtype=weight_dtype, use_fast=False)
    text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_decoder, subfolder="text_encoder",
                                                        torch_dtype=weight_dtype)

    tokenizer_t5 = AutoTokenizer.from_pretrained("google-t5/t5-large", torch_dtype=weight_dtype,
                                                 model_max_length=512)
    text_encoder_t5 = T5EncoderModel.from_pretrained("google-t5/t5-large", torch_dtype=weight_dtype)
    adapter = TextAdapter.from_pretrained('./model/LaVi-Bridge')

    VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}

    # for ckpt_dir in ['results/sd15-align-ct5f/']:
    # if os.path.exists(os.path.join(ckpt_dir, f"s{step}_adapter")):
    ckpt_dir = "model/LongSD/"
    adapter = TextAdapter.from_pretrained(os.path.join(ckpt_dir, f"s28750_adapter"), use_safetensors=True)
    monkeypatch_or_replace_lora_extended(
        vis,
        torch.load(os.path.join(ckpt_dir, f"s28750_lora_vis.pt"), map_location="cpu"),
        r=32,
        target_replace_module=VIS_REPLACE_MODULES,
    )
    collapse_lora(vis, VIS_REPLACE_MODULES)
    monkeypatch_remove_lora(vis)

    # for ckpt_dir in ['results/' + args.ckpt_dir]:
    #     if os.path.exists(os.path.join(ckpt_dir, f"s{step}_adapter")):
    # adapter = TextAdapter.from_pretrained(os.path.join(ckpt_dir, f"s{step}_adapter"), use_safetensors=True)
    monkeypatch_or_replace_lora_extended(
        vis,
        torch.load(args.ckpt_path, map_location="cpu"),
        r=32,
        target_replace_module=VIS_REPLACE_MODULES,
    )

    # merge LoRA
    collapse_lora(vis, VIS_REPLACE_MODULES)
    monkeypatch_remove_lora(vis)

    vae.to(device, weight_dtype)
    vis.to(device, weight_dtype)
    text_encoder_clip.to(device, weight_dtype)
    text_encoder_t5.to(device, weight_dtype)
    adapter.to(device, weight_dtype)
    vae.eval()
    vis.eval()
    text_encoder_clip.eval()
    text_encoder_t5.eval()
    adapter.eval()

    token_, text_ = [tokenizer_clip, tokenizer_t5], [text_encoder_clip, text_encoder_t5]
    caption2embed_simple = lambda captions: caption2embed(captions, token_, text_, args, device, weight_dtype)

    pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_decoder,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            unet=vis,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=weight_dtype,
        )

    if args.sample_method == 'unipc':
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        add_kwargs = {"num_inference_steps": 25, "guidance_scale": 7.5}
    else:
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        def fn_(t):
            return 0, 1
        pipeline.scheduler.get_scalings_for_boundary_condition_discrete = fn_
        add_kwargs = {"num_inference_steps": 4, "guidance_scale": 0.0}

    pipeline = pipeline.to(device)
    # pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(42)

    args.validation_prompts = [args.prompt.strip(), ]

    with torch.no_grad():
        # global validation_embeds
        # if validation_embeds is None:
        validation_embeds = caption2embed_simple(args.validation_prompts + [''] * len(args.validation_prompts))
        encoder_hidden_states = []
        if 'encoder_hidden_states_clip_concat' in validation_embeds:
            encoder_hidden_states.append(validation_embeds["encoder_hidden_states_clip_concat"])
        if 'encoder_hidden_states_t5' in validation_embeds:
            encoder_hidden_states.append(adapter(validation_embeds["encoder_hidden_states_t5"]).sample)
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)
        # encoder_hidden_states = torch.cat([encoder_hidden_states_clip, encoder_hidden_states_t5], dim=1)
        validation_embeddings, validation_embeddings_uc = \
            encoder_hidden_states.split([len(args.validation_prompts), len(args.validation_prompts)], dim=0)

    # images = []
    # for i in range(len(args.validation_prompts)):

    with torch.no_grad():
        with torch.autocast("cuda"):
            # note: high guidance_scale will lead to 过饱和
            images = pipeline(prompt_embeds=validation_embeddings,
                              negative_prompt_embeds=validation_embeddings_uc,  # [i:i + 1]
                              **add_kwargs, generator=generator).images
    
    flag = 0
    for image in images:
        image.save(os.path.join(save_path, f"{flag}-output.png"))
        flag += 1
