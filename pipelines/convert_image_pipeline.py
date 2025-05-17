from PIL import Image
import os

def convert_image_pipeline(
  image: str,
  output_format: str,
  output_dir: str = "output",
  **kwargs,
):
  valid_formats = ['PNG', 'JPEG']
  if output_format not in valid_formats:
    raise ValueError(f"Invalid output format. Choose from: {valid_formats}")

  img = Image.open(image).convert("RGB")  # Ensure compatibility
  ext = '.png' if output_format == 'PNG' else '.jpg'

  if output_dir is None:
    output_dir = os.path.dirname(image)

  output_path = os.path.join(output_dir, f"convert_image_pipeline{ext}")
  img.save(output_path, output_format)
  return output_path