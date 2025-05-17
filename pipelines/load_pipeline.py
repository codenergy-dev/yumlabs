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
import torch

def load_pipeline(
  type: str,
  model: str,
  controlnet: list[str] = [],
  lora: str = None,
  clip_skip: int = None,
  scheduler: str = 'unipc',
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
    elif 'scribble' in preprocessor and model in ["stable-diffusion-v1-5/stable-diffusion-v1-5", "Lykon/AnyLoRA", "admruul/anything-v3.0", "Lykon/dreamshaper-7", "Lykon/dreamshaper-8", "proximasanfinetuning/fantassified_icons_v2", "Anomaly-Games-Inc/aziibpixelmix-json"]:
      controlnet_model.append(ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'shuffle' in preprocessor and model in ["stable-diffusion-v1-5/stable-diffusion-v1-5", "Lykon/AnyLoRA", "admruul/anything-v3.0", "Lykon/dreamshaper-7", "Lykon/dreamshaper-8", "proximasanfinetuning/fantassified_icons_v2", "Anomaly-Games-Inc/aziibpixelmix-json"]:
      controlnet_model.append(ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_shuffle", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'seg' in preprocessor and model in ["stable-diffusion-v1-5/stable-diffusion-v1-5", "Lykon/AnyLoRA", "admruul/anything-v3.0", "Lykon/dreamshaper-7", "Lykon/dreamshaper-8", "proximasanfinetuning/fantassified_icons_v2"]:
      controlnet_model.append(ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch_dtype, use_safetensors=True))
    elif 'tile' in preprocessor and model in ["stable-diffusion-v1-5/stable-diffusion-v1-5", "Lykon/AnyLoRA", "admruul/anything-v3.0", "Lykon/dreamshaper-7", "Lykon/dreamshaper-8", "proximasanfinetuning/fantassified_icons_v2", "Anomaly-Games-Inc/aziibpixelmix-json"]:
      controlnet_model.append(ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch_dtype))
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
  
  pipe = (AutoPipelineForImage2Image if type == "img2img" else AutoPipelineForText2Image).from_pretrained(
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

  if scheduler == "ddim":
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  elif scheduler == "euler":
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
  elif scheduler == "dpmpp-2m":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type = "dpmsolver++"
    pipe.scheduler.use_karras_sigmas = False
  elif scheduler == "dpmpp-2m-karras":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type = "dpmsolver++"
    pipe.scheduler.use_karras_sigmas = True
  elif scheduler == "dpmpp-2m-sde":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type = "sde-dpmsolver++"
    pipe.scheduler.use_karras_sigmas = False
  elif scheduler == "dpmpp-2m-sde-karras":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type = "sde-dpmsolver++"
    pipe.scheduler.use_karras_sigmas = True
  elif scheduler == "ddpm":
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
  elif scheduler == "unipc":
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  elif scheduler == "deis":
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
  else:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  
  return {
    "pipe": pipe,
    "model": model,
    "controlnet": controlnet,
    "lora": lora,
  }