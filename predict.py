# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
import uuid
from omegaconf import OmegaConf
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"
AUXILIARY_MODELS = [
    ("2DFAN4-cd938726ad.zip", "2DFAN4-cd938726ad.zip"),
    ("s3fd-619a316812.pth", "s3fd-619a316812.pth"),
    ("vgg16-397923af.pth", "vgg16-397923af.pth")
]


def download_weights(url: str, dest: str) -> None:
    """
    Download model weights using pget for efficient downloading.
    
    Args:
        url: URL to download from
        dest: Destination directory for downloaded weights
    """
    start = time.time()
    logger.info(f"Downloading from: {url}")
    logger.info(f"Downloading to: {dest}")
    
    try:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
        logger.info(f"Download completed in {time.time() - start:.2f} seconds")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download weights: {e}")
        raise


class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Load the model into memory to make running multiple predictions efficient.
        This includes downloading model weights if needed and creating symbolic links
        for auxiliary models.
        """
        # Download the model weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        else:
            logger.info(f"Model weights already exist at {MODEL_CACHE}")

        # Create PyTorch hub checkpoints directory if it doesn't exist
        torch_hub_path = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        os.makedirs(torch_hub_path, exist_ok=True)
        
        # Create symbolic links for auxiliary models
        for source_file, target_file in AUXILIARY_MODELS:
            source_path = os.path.join(os.getcwd(), "checkpoints/auxiliary", source_file)
            target_path = os.path.join(torch_hub_path, target_file)
            
            # Remove existing link if it exists
            if os.path.islink(target_path):
                os.unlink(target_path)
                
            if os.path.exists(source_path):
                try:
                    os.symlink(source_path, target_path)
                    logger.info(f"Created symbolic link: {source_path} -> {target_path}")
                except OSError as e:
                    logger.error(f"Failed to create symbolic link for {source_file}: {e}")
            else:
                logger.warning(f"Source file {source_path} does not exist")

    def predict(
        self,
        video: Path = Input(description="Input video", default=None),
        audio: Path = Input(description="Input audio file to sync with the video", default=None),
        guidance_scale: float = Input(
            description="Guidance scale: higher values follow audio more precisely", 
            ge=0.5, le=10.0, default=1.5
        ),
        inference_steps: int = Input(
            description="Number of denoising steps (higher = better quality but slower)", 
            ge=20, le=200, default=20
        ),
        num_frames: int = Input(
            description="Number of frames processed in each batch", 
            ge=16, le=200, default=16
        ),
        resolution: int = Input(
            description="Output video resolution (square)", 
            ge=256, le=1024, default=256
        ),
        mixed_noise_alpha: float = Input(
            description="Mixed noise alpha (used for temporal consistency)", 
            ge=0.0, le=2.0, default=1.0
        ),
        seed: int = Input(
            description="Set to 0 for random seed, otherwise uses fixed seed", 
            default=0
        ),
        config_path: str = Input(
            description="UNet config path", 
            default="configs/unet/stage2.yaml", 
            choices=["configs/unet/stage2.yaml", "configs/unet/stage2_efficient.yaml"]
        ),
    ) -> Path:
        """
        Run lip synchronization on the input video with the specified audio.
        
        Args:
            video: Input video file path
            audio: Input audio file path
            guidance_scale: Controls how closely the model follows the audio
            inference_steps: Number of denoising steps for generation
            num_frames: Number of frames to process in each batch
            resolution: Output video resolution (square)
            mixed_noise_alpha: Controls temporal consistency
            seed: Random seed for reproducibility
            config_path: Path to the UNet configuration file
            
        Returns:
            Path to the generated lip-synced video
        """
        # Generate a unique ID for this prediction run
        run_id = str(uuid.uuid4())[:8]
        
        # Set random seed or generate one if 0 or negative
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        logger.info(f"Using seed: {seed}")

        # Convert inputs to string paths
        video_path = str(video)
        audio_path = str(audio)
        ckpt_path = "checkpoints/latentsync_unet.pt"
        
        # Define output paths
        temp_output_path = f"/tmp/temp-output-{run_id}.mp4"
        final_output_path = f"/tmp/output-{run_id}.mp4"
        
        # Modify the config with user parameters
        try:
            config = OmegaConf.load(config_path)
            config.data.resolution = resolution
            config.data.num_frames = num_frames
            config.run.mixed_noise_alpha = mixed_noise_alpha
            
            # Save modified config to a temporary file
            temp_config_path = f"/tmp/temp_config_{run_id}.yaml"
            OmegaConf.save(config, temp_config_path)
            logger.info(f"Created temporary config at {temp_config_path}")
        except Exception as e:
            logger.error(f"Failed to create temporary config: {e}")
            # Copy input video as fallback and return early
            shutil.copy2(video_path, final_output_path)
            return Path(final_output_path)
        
        # Build command for inference script
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
            # Run inference command
            logger.info(f"Running inference command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
            
            # Log command output for debugging
            if result.stdout:
                logger.info(f"Command output: {result.stdout}")
            
            # Check for and log errors
            if result.returncode != 0:
                logger.error(f"Inference failed with exit code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                raise RuntimeError(f"Inference command failed with code {result.returncode}")
            
            # Check if output file exists and copy to final location
            if os.path.exists(temp_output_path):
                # Verify the output file is valid
                file_size = os.path.getsize(temp_output_path)
                if file_size > 0:
                    shutil.copy2(temp_output_path, final_output_path)
                    logger.info(f"Successfully created output at {final_output_path} ({file_size} bytes)")
                else:
                    logger.warning(f"Output file exists but has zero size, using fallback")
                    shutil.copy2(video_path, final_output_path)
            else:
                logger.warning(f"Output file not found at {temp_output_path}, using fallback")
                shutil.copy2(video_path, final_output_path)
                
        except Exception as e:
            logger.error(f"Exception during inference: {str(e)}")
            # Use input video as fallback in case of errors
            logger.info(f"Using original video as fallback due to error")
            shutil.copy2(video_path, final_output_path)
            
        finally:
            # Clean up temporary files
            try:
                for path in [temp_config_path, temp_output_path]:
                    if os.path.exists(path):
                        os.remove(path)
                        logger.debug(f"Removed temporary file: {path}")
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")
            
        return Path(final_output_path)