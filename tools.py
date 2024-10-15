import os
import random
import re
import sys
import torch
import numpy
import wandb
import datasets
from glob import glob
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import custom_bwd, custom_fwd


drop_ratio = float(os.environ.get('MY_ENVIRON_DROP', 0.1))
sample_method = os.environ.get('MY_ENVIRON_SAMPLE', 'lcm') # unipc, lcm
print(f'current drop_ratio of caption: {drop_ratio}; sample_method: {sample_method}')
validation_embeds = None


def sample_images(vae, adapter, caption2embed_simple, unet, args, accelerator, weight_dtype, global_step):
    print("Running validation ... ")
    from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, LCMScheduler

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_decoder,
        vae=accelerator.unwrap_model(vae),
        text_encoder=None,
        tokenizer=None,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=weight_dtype,
    )

    if sample_method == 'unipc':
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        add_kwargs = {"num_inference_steps": 25, "guidance_scale": 7.5}
    else:
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        def fn_(t):
            return 0, 1
        pipeline.scheduler.get_scalings_for_boundary_condition_discrete = fn_
        add_kwargs = {"num_inference_steps": 4, "guidance_scale": 0.0}
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    with torch.no_grad():
        global validation_embeds
        if validation_embeds is None:
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
    with torch.autocast("cuda"):
        # note: high guidance_scale will lead to oversaturation
        images = pipeline(prompt_embeds=validation_embeddings,
                          negative_prompt_embeds=validation_embeddings_uc,  # [i:i + 1]
                          **add_kwargs, generator=generator).images

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = numpy.stack([numpy.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {"validation": [
                    wandb.Image(image, caption=f"{i}: {args.validation_prompts[i % len(args.validation_prompts)]}")
                    for i, image in enumerate(images)]}
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def load_dataset(args):
    data_path = "xxx/dataset/"

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [Image.open(data_path + path).convert("RGB") for path in examples['path']]
        examples["sizes"] = [image.size for image in images]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["caption"] = [caption.replace('\n\n', '').strip() for caption in examples['caption']]

        return examples

    # token_ = os.environ.get('HF_TOKEN', None)
    dataset = datasets.load_dataset("luping-liu/LongAlign", num_proc=8)  # , token=token_
    train_dataset = dataset["train"].with_transform(preprocess_train)
    validation_prompts = []
    for i in range(8):
        validation_prompts.append(dataset['train'][i]['caption'].replace('\n\n', ' ').strip())

    return train_dataset, validation_prompts


pattern = re.compile(r'"!|\.|\?|;"')
pad_embed = None


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # import pdb; pdb.set_trace()

    sentence_list, sentence_index = [], []
    sentence_remain = []
    # class-free guidance
    captions = [example["caption"] if random.random() >= drop_ratio else '' for example in examples]
    for ic, caption in enumerate(captions):
        sentence_list_ = re.split(pattern, caption)
        sentence_list_ = [sent + '.' for sent in sentence_list_ if len(sent) > 0]
        if len(sentence_list_) == 0:
            sentence_list_ = [caption]
        sentence_index += [ic] * len(sentence_list_)
        sentence_list += sentence_list_

        cap_index = sorted(random.sample(range(len(sentence_list_)), min(len(sentence_list_), 4)))  # choose 4 sentences
        cap_ = [sentence_list_[ii].strip() for ii in cap_index]
        sentence_remain.append(' '.join(cap_))

    # input_ids = torch.stack([example["input_ids"] for example in examples])
    # attention_masks = torch.stack([example["attention_mask"] for example in examples])
    return {"pixel_values": pixel_values, "caption": captions,
            "caption_split": sentence_list, "caption_index": sentence_index, "caption_remain": sentence_remain}


@torch.no_grad()
def caption2embed(captions, tokenizer, text_encoder, args, device, weight_dtype):
    results = dict()

    if tokenizer[0] is not None:
        if isinstance(captions[0], list) and len(captions) == 2:
            sentence_list, sentence_index = captions
        else:
            assert isinstance(captions[0], str)
            sentence_list, sentence_index = [], []
            # import pdb; pdb.set_trace()
            for ic, caption in enumerate(captions):
                sentence_list_ = re.split(pattern, caption)
                sentence_list_ = [sent + '.' for sent in sentence_list_ if len(sent) > 0]
                if len(sentence_list_) == 0:
                    sentence_list_ = [caption]
                sentence_list += sentence_list_  # [:1]
                sentence_index += [ic] * len(sentence_list_)

        tokens_clip = tokenizer[0](sentence_list, max_length=tokenizer[0].model_max_length,
                                   padding=True, truncation=True, return_tensors="pt").to(device)
        results["input_ids_clip"], results["attention_mask_clip"] = tokens_clip.input_ids, tokens_clip.attention_mask
        results["clip_sentence_index"] = torch.tensor(sentence_index).to(device)
        # check the use of attention_mask_clip
        # results["encoder_hidden_states_clip"] = text_encoder[0](results["input_ids_clip"], results["attention_mask_clip"])[0]
        results["encoder_hidden_states_clip"] = text_encoder[0](results["input_ids_clip"])[0]
        # import pdb; pdb.set_trace()

        results["encoder_hidden_states_clip_concat"] = []
        for i in range(max(sentence_index) + 1):
            encoder_hidden_states_clip_ = results["encoder_hidden_states_clip"][results["clip_sentence_index"] == i]
            attention_mask_clip_ = results["input_ids_clip"][results["clip_sentence_index"] == i]
            attention_mask_clip_ = ((attention_mask_clip_ != tokenizer[0].pad_token_id) *
                                    (attention_mask_clip_ != tokenizer[0].eos_token_id))

            e_concat_ = encoder_hidden_states_clip_.reshape(-1, encoder_hidden_states_clip_.shape[-1])
            m_concat_ = attention_mask_clip_.reshape(-1)
            encoder_hidden_states_concat = e_concat_[m_concat_]

            if len(encoder_hidden_states_concat) > args.token_length:
                encoder_hidden_states_concat = encoder_hidden_states_concat[:args.token_length]
            else:
                global pad_embed
                if pad_embed is None:
                    pad_embed = tokenizer[0]([''], max_length=tokenizer[0].model_max_length, padding='max_length',
                                             return_tensors="pt").to(device)
                    # note: check the use of attention_mask_clip
                    # pad_embed = text_encoder[0](**pad_embed)[0]
                    pad_embed = text_encoder[0](pad_embed.input_ids)[0]
                    # import pdb; pdb.set_trace()
                    pad_embed = pad_embed[0, -60:].mean(dim=0, keepdim=True)
                    pad_embed = pad_embed.to(device, weight_dtype)
                pad_embed_ = pad_embed.repeat(args.token_length - len(encoder_hidden_states_concat), 1)
                encoder_hidden_states_concat = torch.cat([encoder_hidden_states_concat, pad_embed_], dim=0)
            results["encoder_hidden_states_clip_concat"].append(encoder_hidden_states_concat)
        results["encoder_hidden_states_clip_concat"] = torch.stack(results["encoder_hidden_states_clip_concat"], dim=0)

    if tokenizer[1] is not None:
        tokens_t5 = tokenizer[1](captions, max_length=args.token_length or tokenizer[1].model_max_length,
                                 padding="max_length", truncation=True, return_tensors="pt").to(device)
        results["input_ids_t5"], results["attention_mask_t5"] = tokens_t5.input_ids, tokens_t5.attention_mask
        results["encoder_hidden_states_t5"] = text_encoder[1](results["input_ids_t5"])[0]  # results["attention_mask_t5"]

    return results


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data ** 2 / (scaled_timestep ** 2 + sigma_data ** 2)
    c_out = scaled_timestep / (scaled_timestep ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original(model_output, timesteps, sample, prediction_type, alphas, sigmas, clamp=False):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    if clamp:
        pred_x_0 = pred_x_0.clamp(-1., 1.)

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        self.step_ratio = timesteps // ddim_timesteps
        self.alpha_cumprods = alpha_cumprods

    def to(self, device, dtype):
        self.alpha_cumprods = torch.tensor(self.alpha_cumprods, device=device, dtype=dtype)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep, is_prev=False):
        if not is_prev:
            timestep_prev = timestep - self.step_ratio
        else:
            timestep_prev = timestep
        timestep_prev = torch.where(timestep_prev < 0, torch.zeros_like(timestep_prev), timestep_prev)
        alpha_cumprod_prev = extract_into_tensor(self.alpha_cumprods, timestep_prev, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
