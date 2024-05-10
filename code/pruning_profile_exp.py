import torch
import wandb
from torch import nn
from transformers import CLIPTokenizer
import torch.nn.utils.prune as prune
from torchmetrics.functional.multimodal import clip_score
from functools import partial

from stable_diffusion.model_loader import load_from_standard_weights
from stable_diffusion import *
from utility import percentage_pruned
import time


def run_pruning_profile_exp(structured_pruning=False):
    clip = CLIP()
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()

    state_dict = load_from_standard_weights('./data/weights/v1-5-pruned-emaonly.ckpt')

    clip.load_state_dict(state_dict['clip'], strict=True)
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    tokenizer = CLIPTokenizer("./data/tokenizer_vocab.json", merges_file="./data/tokenizer_merges.txt")

    # prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    uncond_prompt = ""  # Also known as negative prompt
    do_cfg = True
    cfg_scale = 8
    strength = 0.9
    num_inference_steps = 40
    seed = 42
    sampler = 'ddpm'
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pruning_levels = [i / 100 for i in range(0, 41, 5)]

    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "A high tech solarpunk utopia in the Amazon rainforest",
        "A pikachu fine dining with a view to the Eiffel Tower",
        "A mecha robot in a favela in expressionist style",
        "an insect robot preparing a delicious meal",
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ]

    table = wandb.Table(columns=["Experiment Description", "Prompt", "Non-Zero Parameters in Diffusion Unit", "Zero-valued Parameters in Diffusion unit (%)", "CLIP Score", "Image", "CLIP Time", "Diffusion Time", "Decoder Time", "Total Generation Time"])

    for p in pruning_levels:
        diff = Diffusion()
        diff.load_state_dict(state_dict['diffusion'], strict=True)
        output_images = []

        if p > 0:
            for module_name, module in diff.named_modules():
                if 'unet' in module_name and (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                    # If the module is a convolutional or linear layer, prune it
                    if structured_pruning:
                        prune.ln_structured(module, name='weight', amount=p, n=2, dim=0)
                    else:
                        prune.l1_unstructured(module, name='weight', amount=p)
                    prune.remove(module, 'weight')
                    if module.bias is not None and not structured_pruning:
                        prune.l1_unstructured(module, name='bias', amount=p)
                        prune.remove(module, 'bias')
            print('Model Pruned!')

        params, pp = percentage_pruned(diff)

        models = {
            "clip": clip,
            "encoder": encoder,
            "decoder": decoder,
            "diffusion": diff,
        }

        experiment_type = 'Baseline Model, no pruning' if p == 0 else f'UNet Pruned - {p * 100}%'

        for prompt in prompts:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            output_img, ct, dft, dt = generate(
                prompt=prompt,
                uncond_prompt=uncond_prompt,
                input_image=None,
                strength=strength,
                do_cfg=do_cfg,
                cfg_scale=cfg_scale,
                sampler_name=sampler,
                n_inference_steps=num_inference_steps,
                seed=seed,
                models=models,
                device=device,
                idle_device="cpu",
                tokenizer=tokenizer,
                profiling=True
            )

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            images_int = (output_img * 255).astype("uint8")
            output_images.append(torch.from_numpy(images_int).permute(2, 0, 1))
            clip_score_val = clip_score_fn(torch.from_numpy(images_int).permute(2, 0, 1), prompt).detach()
            clip_score_val = round(float(clip_score_val), 4)

            table.add_data(experiment_type,
                           prompt,
                           params,
                           pp,
                           clip_score_val,
                           wandb.Image(output_img),
                           ct,
                           dft,
                           dt,
                           end_time - start_time)

    if structured_pruning:
        wandb.log({"Profiling Results for Structurally Pruning Convolutional and Linear Layers of UNet": table})
    else:
        wandb.log({"Profiling Results for Pruning Convolutional and Linear Layers of UNet": table})

    return

