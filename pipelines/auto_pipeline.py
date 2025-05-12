import math
import os
from diffusers import (
  AutoencoderKL,
  AutoPipelineForImage2Image,
  AutoPipelineForText2Image,
  ControlNetModel,
  DEISMultistepScheduler,
  DDIMScheduler,
  DDPMScheduler,
  DPMSolverMultistepScheduler,
  EulerDiscreteScheduler,
  UniPCMultistepScheduler,
)
from PIL import Image, PngImagePlugin
import torch
import random

def auto_pipeline(
  model: str,
  prompt: str,
  negative_prompt: str = None,
  image: str = None,
  controlnet: list[str] = [],
  lora: str = None,
  seed: int = None,
  count: int = 1,
  num_inference_steps: int = 2,
  guidance_scale: float = 7.5,
  output_dir: str = 'output',
  progress: bool = False,
  width: int = 512,
  height: int = 512,
  strength: float = None,
  denoising_start: float = 0,
  denoising_end: float = 1.0,
  controlnet_conditioning_scale: float = 0.5,
  control_guidance_start: float = 0.0,
  control_guidance_end: float = 1.0,
  clip_skip: int = None,
  sampler: str = 'unipc',
  device: str = 'cuda',
  torch_dtype: str = 'float16',
  **kwargs,
):
  torch_dtype = torch.float16 if torch_dtype == 'float16' else torch.float32
  controlnet_model = []
  if isinstance(controlnet, str):
    controlnet = controlnet.split(',')
  for index in range(len(controlnet)):
    preprocessor = controlnet[index]
    if 'anyline' in preprocessor:
      controlnet_model.append(ControlNetModel.from_pretrained("TheMistoAI/MistoLine", variant="fp16", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'canny' in preprocessor and model in ["stable-diffusion-v1-5/stable-diffusion-v1-5", "Lykon/AnyLoRA", "admruul/anything-v3.0", "Lykon/dreamshaper-7", "Lykon/dreamshaper-8", "proximasanfinetuning/fantassified_icons_v2"]:
      controlnet_model.append(ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'hed' in preprocessor and model in ["stable-diffusion-v1-5/stable-diffusion-v1-5", "Lykon/AnyLoRA", "admruul/anything-v3.0", "Lykon/dreamshaper-7", "Lykon/dreamshaper-8", "proximasanfinetuning/fantassified_icons_v2"]:
      controlnet_model.append(ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'pose' in preprocessor and model in ["stable-diffusion-v1-5/stable-diffusion-v1-5", "Lykon/AnyLoRA", "admruul/anything-v3.0", "Lykon/dreamshaper-7", "Lykon/dreamshaper-8", "proximasanfinetuning/fantassified_icons_v2"]:
      controlnet_model.append(ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'zoe' in preprocessor and model in ["stable-diffusion-v1-5/stable-diffusion-v1-5", "Lykon/AnyLoRA", "admruul/anything-v3.0", "Lykon/dreamshaper-7", "Lykon/dreamshaper-8", "proximasanfinetuning/fantassified_icons_v2"]:
      controlnet_model.append(ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'canny' in preprocessor and model == "stabilityai/stable-diffusion-xl-base-1.0":
      controlnet_model.append(ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'hed' in preprocessor and model == "stabilityai/stable-diffusion-xl-base-1.0":
      controlnet_model.append(ControlNetModel.from_pretrained("Eugeoter/noob-sdxl-controlnet-softedge_hed", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'pose' in preprocessor and model == "stabilityai/stable-diffusion-xl-base-1.0":
      controlnet_model.append(ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch_dtype))
    elif 'zoe' in preprocessor and model == "stabilityai/stable-diffusion-xl-base-1.0":
      controlnet_model.append(ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch_dtype, use_safetensors=True))
    else:
      controlnet[index] = None
  controlnet = [preprocessor for preprocessor in controlnet if preprocessor is not None]
    
  pipe_kwargs = {}
  if len(controlnet):
    pipe_kwargs["controlnet"] = controlnet_model
  if model == "stabilityai/stable-diffusion-xl-base-1.0":
    pipe_kwargs["vae"] = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype, use_safetensors=True)
  
  pipe = (AutoPipelineForImage2Image if image is not None else AutoPipelineForText2Image).from_pretrained(
    model,
    torch_dtype=torch_dtype,
    use_safetensors=True if model != "admruul/anything-v3.0" else False,
    safety_checker=None,
    **pipe_kwargs,
  ).to(device)

  if lora is not None:
    for lora in lora.split(','):
      pipe.load_lora_weights(lora)
  
  if clip_skip is not None and hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder.config, "clip_skip"):
    pipe.text_encoder.config.clip_skip = clip_skip

  if sampler == "ddim":
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  elif sampler == "euler":
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
  elif sampler == "dpmpp-2m":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type = "dpmsolver++"
    pipe.scheduler.use_karras_sigmas = False
  elif sampler == "dpmpp-2m-karras":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type = "dpmsolver++"
    pipe.scheduler.use_karras_sigmas = True
  elif sampler == "dpmpp-2m-sde":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type = "sde-dpmsolver++"
    pipe.scheduler.use_karras_sigmas = False
  elif sampler == "dpmpp-2m-sde-karras":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type = "sde-dpmsolver++"
    pipe.scheduler.use_karras_sigmas = True
  elif sampler == "ddpm":
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
  elif sampler == "unipc":
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  elif sampler == "deis":
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
  else:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  
  if strength is None:
    strength = 1.0 - denoising_start

  output = []
  for i in range(count):
    current_seed = random.randint(1, 2**32 - 1) if seed is None else seed
    generator = torch.manual_seed(current_seed)
    last_num_inference_steps = 0

    for num_inference_steps in range(1 if progress else num_inference_steps, num_inference_steps + 1):
      curr_num_inference_steps = math.floor(num_inference_steps * strength)
      if curr_num_inference_steps == last_num_inference_steps:
        continue
      elif curr_num_inference_steps >= 1:
        last_num_inference_steps = curr_num_inference_steps
        meta = PngImagePlugin.PngInfo()
        meta.add_text("pipeline", "img2img" if image is not None else "txt2img")
        meta.add_text("model", model)
        meta.add_text("lora", f"{lora}")
        meta.add_text("prompt", prompt)
        meta.add_text("negative_prompt", f"{negative_prompt}")
        meta.add_text("image", f"{image}")
        meta.add_text("control_image", f"{controlnet}")
        meta.add_text("num_inference_steps", f"{num_inference_steps}")
        meta.add_text("guidance_scale", f"{guidance_scale}")
        meta.add_text("seed", f"{current_seed}")
        meta.add_text("strength", f"{strength}")
        meta.add_text("denoising_end", f"{denoising_end}")
        meta.add_text("controlnet_conditioning_scale", f"{controlnet_conditioning_scale}")
        meta.add_text("control_guidance_start", f"{control_guidance_start}")
        meta.add_text("control_guidance_end", f"{control_guidance_end}")
        output.append({
          "image": os.path.join(output_dir, f"seed_{current_seed}_step_{num_inference_steps:02}.png")
        })
        pipe(
          prompt=prompt,
          negative_prompt=negative_prompt,
          image=Image.open(image).convert("RGB") if image is not None else [Image.open(preprocessor).convert("RGB") for preprocessor in controlnet],
          control_image=[Image.open(preprocessor).convert("RGB") for preprocessor in controlnet],
          num_inference_steps=num_inference_steps,
          guidance_scale=guidance_scale,
          generator=generator,
          width=width,
          height=height,
          strength=strength,
          denoising_end=denoising_end,
          controlnet_conditioning_scale=[controlnet_conditioning_scale for preprocessor in controlnet],
          control_guidance_start=control_guidance_start,
          control_guidance_end=control_guidance_end,
        ).images[0].save(output[-1]["image"], pnginfo=meta)
  return output