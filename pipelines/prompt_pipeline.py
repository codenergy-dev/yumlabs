def prompt_pipeline(
  prompt: str,
  negative_prompt: str = None,
  **kwargs,
):
  return { "prompt": prompt, "negative_prompt": negative_prompt }