# Copyright 2022 Luping Liu
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
import sys
import math
import json
import inspect
import random
import itertools
import argparse
import platform
import numpy as np
import wandb.util
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, T5EncoderModel, CLIPTextModel

from modules.lora import inject_trainable_lora_extended, save_lora_weight
from modules.lora import monkeypatch_or_replace_lora_extended, collapse_lora, monkeypatch_remove_lora
from modules.adapters import TextAdapter
from tools import load_dataset, collate_fn, caption2embed, sample_images
from tools import scalings_for_boundary_conditions, get_predicted_original, DDIMSolver, append_dims


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="LongAlign Training")
    parser.add_argument("--pretrained_decoder", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--token_length", type=int, default=77)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--loss_type", type=str, default="l2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_train_epochs", type=int, default=1)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1250)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulation_steps,
        log_with='wandb' if not args.debug else None,
    )
    if accelerator.is_main_process and not args.debug:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Blocks to inject LoRA
    # note: result without CrossAttention is poor
    VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}

    # Modules of T2I diffusion models
    vae = AutoencoderKL.from_pretrained(args.pretrained_decoder, subfolder="vae", torch_dtype=weight_dtype)
    vis = UNet2DConditionModel.from_pretrained(args.pretrained_decoder, subfolder="unet", torch_dtype=weight_dtype)
    teacher_vis = UNet2DConditionModel.from_pretrained(args.pretrained_decoder, subfolder="unet",
                                                       torch_dtype=weight_dtype)

    tokenizer_clip = AutoTokenizer.from_pretrained(args.pretrained_decoder, subfolder="tokenizer",
                                                   torch_dtype=weight_dtype, use_fast=False)
    text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_decoder, subfolder="text_encoder",
                                                      torch_dtype=weight_dtype)

    tokenizer_t5 = AutoTokenizer.from_pretrained("google-t5/t5-large", torch_dtype=weight_dtype, model_max_length=512)
    text_encoder_t5 = T5EncoderModel.from_pretrained("google-t5/t5-large", torch_dtype=weight_dtype)
    
    # download from https://huggingface.co/shihaozhao/LaVi-Bridge/tree/main/t5_unet/adapter --> ./model/LaVi-Bridge
    adapter = TextAdapter.from_pretrained('./model/LaVi-Bridge')

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_decoder, subfolder="scheduler",
                                                    torch_dtype=weight_dtype)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    args.num_ddim_timesteps = 50
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(),
                        timesteps=noise_scheduler.config.num_train_timesteps,
                        ddim_timesteps=args.num_ddim_timesteps)

    if args.ckpt_dir != "":
        args.ckpt_dir = args.ckpt_dir.split(", ")

        for ckpt_dir in args.ckpt_dir:
            if os.path.exists(os.path.join(ckpt_dir, f"adapter")):
                adapter = TextAdapter.from_pretrained(os.path.join(ckpt_dir, f"adapter"))
            # else:
            #     print(f"adapter ckpt not found in {args.ckpt_dir}")

            # LoRA
            monkeypatch_or_replace_lora_extended(
                vis,
                torch.load(os.path.join(ckpt_dir, f"lora_vis.pt"), map_location="cpu"),
                r=32,
                target_replace_module=VIS_REPLACE_MODULES,
            )

            # merge LoRA
            collapse_lora(vis, VIS_REPLACE_MODULES)
            monkeypatch_remove_lora(vis)
    teacher_vis.load_state_dict(vis.state_dict())

    vae.requires_grad_(False)
    vis.requires_grad_(False)
    teacher_vis.requires_grad_(False)
    text_encoder_clip.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)

    # LoRA injection
    vis_lora_params, _ = inject_trainable_lora_extended(
        vis,
        r=args.lora_rank,
        target_replace_module=VIS_REPLACE_MODULES,
    )

    if args.gradient_checkpointing:
        vis.enable_gradient_checkpointing()

    # Dataset and dataloader
    train_dataset, args.validation_prompts = load_dataset(args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, prefetch_factor=3,
                                                   num_workers=6 if not args.debug else 6, collate_fn=collate_fn,
                                                   shuffle=True, drop_last=True, pin_memory=True)
    args.max_train_steps = len(train_dataset) * args.max_train_epochs // \
                           (args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps)

    # Optimizer and scheduler
    optimizer_class = torch.optim.AdamW
    params_to_optimize = ([
        {"params": itertools.chain(*vis_lora_params)},
        # {"params": adapter.parameters()},
    ])
    # todo check 1e-6 and 1e-8 optim_eps
    optimizer = optimizer_class(params_to_optimize, lr=args.learning_rate, betas=(0.9, 0.999),
                                weight_decay=1e-2, eps=1e-8)
    lr_scheduler = get_scheduler("constant_with_warmup", optimizer=optimizer,
                                 num_warmup_steps=2000 * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes)

    vae.to(accelerator.device, weight_dtype)
    teacher_vis.to(accelerator.device, weight_dtype)
    text_encoder_clip.to(accelerator.device, weight_dtype)
    text_encoder_t5.to(accelerator.device, weight_dtype)
    alpha_schedule = alpha_schedule.to(accelerator.device, weight_dtype)
    sigma_schedule = sigma_schedule.to(accelerator.device, weight_dtype)
    get_predicted_original_sample = lambda model_output, timesteps, sample: get_predicted_original(
        model_output, timesteps, sample, noise_scheduler.config.prediction_type, alpha_schedule, sigma_schedule)
    solver = solver.to(accelerator.device, weight_dtype)
    vae.eval()
    teacher_vis.eval()
    text_encoder_clip.eval()
    text_encoder_t5.eval()
    vis, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vis, optimizer, train_dataloader, lr_scheduler, )

    caption2embed_simple = lambda captions: caption2embed(captions, [tokenizer_clip, tokenizer_t5], [text_encoder_clip, text_encoder_t5],
                                                          args, accelerator.device, weight_dtype)
    uncond_prompt_embeds_pre = caption2embed_simple([''] * args.batch_size)
    uncond_hidden_states_clip = uncond_prompt_embeds_pre['encoder_hidden_states_clip_concat']
    uncond_hidden_states_t5 = uncond_prompt_embeds_pre["encoder_hidden_states_t5"]
    uncond_hidden_states_t5 = adapter(uncond_hidden_states_t5).sample
    uncond_hidden_states_ct5 = torch.cat([uncond_hidden_states_clip, uncond_hidden_states_t5], dim=1)

    if not os.path.exists(f"{args.output_dir}/latest_status"):
        args.resume = False
    if args.resume:
        accelerator.load_state(f"{args.output_dir}/latest_status")
        resume_json = json.load(open(f"{args.output_dir}/latest_status/resume.json", 'r'))
        global_step = resume_json['global_step']
        last_save = global_step // accelerator.gradient_accumulation_steps
        num_train_epochs = args.max_train_epochs - global_step // len(train_dataset)
        wandb_id = resume_json['wandb_id']
    else:
        global_step = last_save = 0
        num_train_epochs = args.max_train_epochs
        wandb_id = wandb.util.generate_id()
        resume_json = {'global_step': global_step, 'wandb_id': wandb_id}

    # Log
    if accelerator.is_main_process:
        if not args.debug:
            tracker_config = dict(vars(args))
            accelerator.init_trackers(project_name='dense-cap-sea', config=tracker_config,
                                      init_kwargs={'wandb': {'name': args.output_dir.split('/')[-1],
                                                             'resume': args.resume, 'id': wandb_id}, })

            # adapter.eval()
            vis.eval()
            sample_images(vae, adapter, caption2embed_simple, vis, args, accelerator, weight_dtype, global_step=0)

        print(f"Comuting node = {platform.node()}")
        print(f"Num examples = {len(train_dataset)}")
        print(f"Total batch size = {args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps}")
        print(f"Num Epochs = {num_train_epochs}")
        print(f"Total optimization steps = {args.max_train_steps}")

    accelerator.wait_for_everyone()

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
    progress_bar.set_description("Steps")
    if args.resume:
        progress_bar.update(global_step // accelerator.gradient_accumulation_steps)

    # Training
    for _ in range(num_train_epochs):
        # adapter.train()
        vis.train()

        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vis):
                with torch.no_grad():
                    # Latent preparation
                    latents = []
                    batch['pixel_values'] = batch['pixel_values'].to(weight_dtype)
                    for i in range(0, batch['pixel_values'].shape[0], 8):
                        latents.append(vae.encode(batch['pixel_values'][i: i + 8]).latent_dist.sample())
                    latents = torch.cat(latents, dim=0)
                    # latents = vae.encode(batch['pixel_values'].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    bsz = latents.shape[0]

                    start_timesteps = torch.randint(10, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                                    device=latents.device).long()
                    timesteps = start_timesteps - solver.step_ratio
                    timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                    # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                    c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
                    c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]

                    c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                    c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                    noise = torch.randn_like(latents)
                    noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
                    # noisy_model_input_pre = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Note that for LCM-LoRA distillation it is not necessary to use a guidance scale embedding
                    # args.w_min, args.w_max = 4, 10
                    args.w_min = args.w_max = 7.5
                    w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                    w = w.reshape(bsz, 1, 1, 1)
                    w = w.to(device=latents.device, dtype=latents.dtype)

                    # 6. Prepare prompt embeds and unet_added_conditions
                    # captions_ = [c if random.random() > 0.1 else '' for c in batch['caption']]
                    # captions_ = batch['caption']
                    captions_ = [batch['caption_split'], batch['caption_index']]
                    # encoder_hidden_states_pre = text_encoder(batch['input_ids'])[0]
                    encoder_hidden_states_pre = caption2embed_simple(captions_)
                    encoder_hidden_states_clip = encoder_hidden_states_pre['encoder_hidden_states_clip_concat']
                    encoder_hidden_states_t5 = encoder_hidden_states_pre["encoder_hidden_states_t5"]

                    encoder_hidden_states_t5 = adapter(encoder_hidden_states_t5).sample
                    encoder_hidden_states_ct5 = torch.cat([encoder_hidden_states_clip, encoder_hidden_states_t5], dim=1)

                # import pdb; pdb.set_trace()
                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                noise_pred = vis(noisy_model_input, start_timesteps, encoder_hidden_states_ct5).sample
                pred_x_0_s = get_predicted_original_sample(noise_pred, start_timesteps, noisy_model_input)
                model_pred_s = c_skip_start * noisy_model_input + c_out_start * pred_x_0_s

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0
                with torch.no_grad():
                    with torch.autocast(accelerator.device.type):
                        # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                        cond_pred_noise = teacher_vis(noisy_model_input, start_timesteps, encoder_hidden_states_ct5).sample
                        cond_pred_x0 = get_predicted_original_sample(cond_pred_noise, start_timesteps, noisy_model_input)

                        # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                        uncond_pred_noise = teacher_vis(noisy_model_input, start_timesteps, uncond_hidden_states_ct5).sample
                        uncond_pred_x0 = get_predicted_original_sample(uncond_pred_noise, start_timesteps, noisy_model_input)

                        # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                        # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                        pred_x0 = uncond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = uncond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                        # 4. Run one step of the ODE solver to estimate the next point x_prev
                        # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                        x_prev = solver.ddim_step(pred_x0, pred_noise, start_timesteps)

                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                # Note that we do not use a separate target network for LCM-LoRA distillation.
                with torch.no_grad():
                    with torch.autocast(accelerator.device.type):
                        # x_prev = noisy_model_input_pre
                        target_noise_pred = vis(x_prev.to(weight_dtype), timesteps, encoder_hidden_states_ct5).sample
                    pred_x_0_t = get_predicted_original_sample(target_noise_pred, timesteps, x_prev)
                    model_pred_t = c_skip * x_prev + c_out * pred_x_0_t

                # Optimization
                if args.loss_type == "huber":
                    args.huber_c = 0.001
                    # loss = torch.sqrt((pred_x_0_s.float() - pred_x_0_t.float()) ** 2 + args.huber_c ** 2).mean() - args.huber_c
                    loss = torch.sqrt((model_pred_s.float() - model_pred_t.float()) ** 2 + args.huber_c ** 2).mean() - args.huber_c
                    # note: cannot directly combine with mse_loss of noise
                    # loss_1 = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                else:
                    # loss = F.mse_loss(pred_x_0_s.float(), pred_x_0_t.float(), reduction="mean")
                    loss = F.mse_loss(model_pred_s.float(), model_pred_t.float(), reduction="mean")
                # if args.pretrained_decoder == "stabilityai/stable-diffusion-2-1":
                #     loss = loss * 0.2
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (itertools.chain(vis.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, 0.3)
                optimizer.step()
                lr_scheduler.step()
                global_step += 1
                optimizer.zero_grad()

            # Saving
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step_ = global_step // accelerator.gradient_accumulation_steps

                if accelerator.is_main_process:  # accelerator.sync_gradients and
                    if global_step_ % 5 == 0:
                        accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},
                                        step=global_step_)

                    if global_step_ % args.validation_steps == 0:
                        # adapter.eval()
                        vis.eval()
                        sample_images(vae, adapter, caption2embed_simple, vis, args, accelerator, weight_dtype,
                                      global_step_)
                        # adapter.train()
                        vis.train()

                    if global_step_ - last_save >= args.save_steps or global_step_ >= args.max_train_steps:
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(accelerator.unwrap_model).parameters.keys())
                        extra_args = ({"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {})
                        save_lora_weight(
                            accelerator.unwrap_model(vis, **extra_args),
                            f"{args.output_dir}/s{global_step_}_lora_vis.pt",
                            target_replace_module=VIS_REPLACE_MODULES,
                        )
                        # accelerator.unwrap_model(adapter, **extra_args).save_pretrained(
                        #     f"{args.output_dir}/s{global_step_}_adapter")
                        last_save = global_step_

                        # training status
                        accelerator.save_state(f"{args.output_dir}/latest_status")
                        resume_json['global_step'] = global_step
                        json.dump(resume_json, open(f"{args.output_dir}/latest_status/resume.json", 'w'))

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    # accelerator.log(logs, step=global_step)

                if global_step_ >= args.max_train_steps:
                    accelerator.wait_for_everyone()
                    accelerator.end_training()

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
