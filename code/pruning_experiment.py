import torch
import wandb
from torch import nn
from transformers import CLIPTokenizer
import torch.nn.utils.prune as prune
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from torchmetrics.image.fid import FrechetInceptionDistance

from stable_diffusion.model_loader import load_from_standard_weights
from stable_diffusion import *
from utility import percentage_pruned, get_real_images


def run_pruning_exp():
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

    pruning_levels = [i/100 for i in range(0, 96, 5)]

    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "A high tech solarpunk utopia in the Amazon rainforest",
        "A pikachu fine dining with a view to the Eiffel Tower",
        "A mecha robot in a favela in expressionist style",
        "an insect robot preparing a delicious meal",
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ]

    prompts_fid = [
        "cassette player",
        "chainsaw",
        "chainsaw",
        "church",
        "gas pump",
        "gas pump",
        "gas pump",
        "parachute",
        "parachute",
        "tench",
    ]
    real_images = get_real_images()

    table = wandb.Table(columns=["Experiment Description", "Prompt", "Non-Zero Parameters in Diffusion Unit", "Zero-valued Parameters in Diffusion unit (%)", "CLIP Score", "Image"])
    table2 = wandb.Table(columns=["Experiment Description", "Non-Zero Parameters in Diffusion Unit",
                                  "Zero-valued Parameters in Diffusion unit (%)", "CLIP Score", "FID Score"])

    for p in pruning_levels:
        diff = Diffusion()
        diff.load_state_dict(state_dict['diffusion'], strict=True)
        output_images = []
        fake_images = []

        if p > 0:
            for module_name, module in diff.named_modules():
                if 'unet' in module_name and (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                    # If the module is a convolutional or linear layer, prune it
                    prune.l1_unstructured(module, name='weight', amount=p)
                    prune.remove(module, 'weight')
                    if module.bias is not None:
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

            output_img = generate(
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
            )

            images_int = (output_img * 255).astype("uint8")
            output_images.append(torch.from_numpy(images_int).permute(2, 0, 1))
            clip_score_val = clip_score_fn(torch.from_numpy(images_int).permute(2, 0, 1), prompt).detach()
            clip_score_val = round(float(clip_score_val), 4)

            table.add_data(experiment_type,
                           prompt,
                           params,
                           pp,
                           clip_score_val,
                           wandb.Image(output_img))

        for prompt in prompts_fid:
            output_img = generate(
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
            )

            images_int = (output_img * 255).astype("uint8")
            fake_images.append(torch.from_numpy(images_int).permute(2, 0, 1))

        fake_images = torch.stack(fake_images, dim=0)
        fid = FrechetInceptionDistance(normalize=True)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

        clip_score_net = clip_score_fn(output_images, prompts).detach()
        clip_score_net = round(float(clip_score_net), 4)

        table2.add_data(experiment_type, params, pp, clip_score_net, round(float(fid.compute()), 4))

    wandb.log({"Results for Pruning Convolutional and Linear Layers of UNet": table})
    wandb.log({"CLIP Score and FID for Pruning Convolutional and Linear Layers of UNet": table2})

    return

