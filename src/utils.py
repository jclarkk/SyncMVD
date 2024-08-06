import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from torchvision import transforms
from torchvision.transforms import Resize, InterpolationMode

'''
	Encoding and decoding functions similar to diffusers library implementation
'''


@torch.no_grad()
def encode_latents(vae, imgs):
    imgs = (imgs - 0.5) * 2
    latents = vae.encode(imgs).latent_dist.sample()
    latents = vae.config.scaling_factor * latents
    return latents


@torch.no_grad()
def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents

    image = vae.decode(latents, return_dict=False)[0]
    torch.cuda.current_stream().synchronize()

    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.permute(0, 2, 3, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()

    return image


# A fast decoding method based on linear projection of latents to rgb
@torch.no_grad()
def latent_preview(x):
    # adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
    v1_4_latent_rgb_factors = torch.tensor([
        #   R        G        B
        [0.298, 0.207, 0.208],  # L1
        [0.187, 0.286, 0.173],  # L2
        [-0.158, 0.189, 0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ], dtype=x.dtype, device=x.device)
    image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()
    return image


# Decode each view and bake them into a rgb texture
def get_rgb_texture(vae, uvp_rgb, latents, refine=False):
    # Decode latents to images
    result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    if refine is True:
        # Use SDXL Refiner
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        refiner.to("cuda")

        refiner.set_progress_bar_config(disable=True)

        refined_views = []
        for view in result_views:
            view_pil = transforms.ToPILImage()(view / 2 + 0.5)

            refined_view = refiner(
                prompt="high quality, detailed image, photorealistic",
                image=view_pil,
                num_inference_steps=30,
                strength=0.3,
                guidance_scale=7.5
            ).images[0]

            refined_view = transforms.ToTensor()(refined_view) * 2 - 1
            refined_views.append(refined_view)

        refined_views = torch.stack(refined_views).to(latents.device)

        if refined_views.shape[-2:] != (uvp_rgb.render_size, uvp_rgb.render_size):
            resize = Resize((uvp_rgb.render_size,) * 2, interpolation=InterpolationMode.BICUBIC, antialias=True)
            result_views = resize(refined_views / 2 + 0.5).clamp(0, 1).unbind(0)
        else:
            result_views = (refined_views / 2 + 0.5).clamp(0, 1).unbind(0)

    # Bake texture
    textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[],
                                                                                  exp=6, noisy=False)
    result_tex_rgb_output = result_tex_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]

    return result_tex_rgb, result_tex_rgb_output
