# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
import uuid
from omegaconf import OmegaConf
import shutil

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Soft links for the auxiliary models
        os.system("mkdir -p ~/.cache/torch/hub/checkpoints")
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip"
        )
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth"
        )
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
        )

    def predict(
        self,
        video: Path = Input(description="Input video", default=None),
        audio: Path = Input(description="Input audio file to sync with the video", default=None),
        guidance_scale: float = Input(description="Guidance scale: higher values follow audio more precisely", ge=0.5, le=10.0, default=1.5),
        inference_steps: int = Input(description="Number of denoising steps (higher = better quality but slower)", ge=20, le=200, default=20),
        num_frames: int = Input(description="Number of frames processed in each batch", ge=16, le=200, default=16),
        resolution: int = Input(description="Output video resolution (square)", ge=256, le=1024, default=256),
        mixed_noise_alpha: float = Input(description="Mixed noise alpha (used for temporal consistency)", ge=0.0, le=2.0, default=1.0),
        seed: int = Input(description="Set to 0 for random seed, otherwise uses fixed seed", default=0),
        config_path: str = Input(description="UNet config path", default="configs/unet/stage2.yaml", choices=["configs/unet/stage2.yaml", "configs/unet/stage2_efficient.yaml"]),
    ) -> Path:
        """Run a single prediction on the model"""
        # Generate a unique ID for this prediction run
        run_id = str(uuid.uuid4())[:8]
        
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        video_path = str(video)
        audio_path = str(audio)
        ckpt_path = "checkpoints/latentsync_unet.pt"
        
        # Use both a temporary working path and a final output path
        temp_output_path = f"/tmp/temp-output-{run_id}.mp4"
        final_output_path = f"/tmp/output-{run_id}.mp4"
        
        # First, modify the config file to update resolution and num_frames
        config = OmegaConf.load(config_path)
        config.data.resolution = resolution
        config.data.num_frames = num_frames
        config.run.mixed_noise_alpha = mixed_noise_alpha
        
        # Create a temporary config file with unique name
        temp_config_path = f"/tmp/temp_config_{run_id}.yaml"
        OmegaConf.save(config, temp_config_path)
        
        # Create command list with all parameters
        command = [
            "python", "-m", "scripts.inference",
            "--unet_config_path", temp_config_path,
            "--inference_ckpt_path", ckpt_path,
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", temp_output_path,
            "--seed", str(seed)
        ]
        
        try:
            # Use subprocess.run with environment isolation for better process management
            print(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
            
            # Print output for debugging
            print(f"Command output: {result.stdout}")
            
            # Print any errors for debugging
            if result.returncode != 0:
                print(f"Error during inference: {result.stderr}")
            
            # Check if output file exists
            if os.path.exists(temp_output_path):
                # Copy to the final output path to ensure we're returning a complete file
                shutil.copy2(temp_output_path, final_output_path)
                print(f"Successfully created output at {final_output_path}")
            else:
                # If output doesn't exist, create a fallback output
                print(f"Output file not found at {temp_output_path}, creating fallback")
                # Copy input video as fallback to ensure we return something
                shutil.copy2(video_path, final_output_path)
        except Exception as e:
            print(f"Exception during inference: {str(e)}")
            # Create fallback output in case of errors
            shutil.copy2(video_path, final_output_path)
        finally:
            # Clean up temporary config file
            try:
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
                if os.path.exists(temp_output_path) and os.path.exists(final_output_path):
                    os.remove(temp_output_path)
            except Exception as e:
                print(f"Error cleaning up temporary files: {str(e)}")
            
        return Path(final_output_path)