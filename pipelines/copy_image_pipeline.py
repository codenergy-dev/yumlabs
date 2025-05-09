import os
from PIL import Image

def copy_image_pipeline(
  image: str,
  name: str = None,
  output_dir: str = "output",
  **kwargs,
):
  image_file = Image.open(image)
  output = os.path.join(output_dir, (image if name is None else name).split("/")[-1])
  os.makedirs(output_dir, exist_ok=True)
  image_file.save(output)
  return { "image": output }