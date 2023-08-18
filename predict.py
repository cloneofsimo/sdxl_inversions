from typing import Optional
from cog import BasePredictor, Input, Path
import time
import subprocess
import os
from pnp_pipeline import SDXLDDIMPipeline
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import PIL

SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        if not os.path.exists("sdxl-cache"):
            download_weights(SDXL_URL, "./sdxl-cache")

        self.custom_pipe = SDXLDDIMPipeline.from_pretrained(
            "./sdxl-cache",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "./sdxl-cache",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def predict(
        self,
        original_image: Path = Input(
            description="base image",
        ),
        original_prompt: str = Input(
            description="description of image",
        ),
        original_negative_prompt: str = Input(
            description="negative description of image",
            default=None,
        ),
        num_inversion_steps: int = Input(
            description="Number of steps to calculate original latents",
            ge=1,
            le=500,
            default=50,
        ),
        new_prompt: str = Input(description="new prompt"),
        new_negative_prompt: str = Input(
            description="new negative prompt", default=None
        ),
        num_inference_steps: int = Input(
            description="denoising steps to generate new image",
            ge=1,
            le=500,
            default=50,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=3.5
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
    ) -> Path:
        init_image = PIL.Image.open(original_image)

        x = self.custom_pipe(
            prompt=original_prompt,
            negative_prompt=original_negative_prompt,
            image=init_image,
            num_inference_steps=num_inversion_steps,
        )

        if not apply_watermark:
            watermark_cache = self.pipe.watermark
            self.pipe.watermark = None

        result = self.pipe(
            prompt=new_prompt,
            negative_prompt=new_negative_prompt,
            latents=x[0].clone(),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        if not apply_watermark:
            self.pipe.watermark = watermark_cache

        result.save("out.png")
        return Path("out.png")
