import shutil

def copy_image_pipeline(
  image: str,
  target: str,
  **kwargs,
):
  shutil.copy2(image, target)
  return { "image": target }