import torch
import torch.distributed as dist
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import fsspec
import webdataset as wds
import json
import typer
import os
from numpyencoder import NumpyEncoder

import logging
import copy
from diffusers import AutoencoderKL
from diffusers.models.vae import DiagonalGaussianDistribution
from functools import partial
import io
from collections import defaultdict

from multiprocessing import Process
from augmentations import create_timm_transform
from typing import List
from torch.cuda.amp import autocast
import time

app = typer.Typer()

class VAEEncoder(torch.nn.Module):
    def __init__(self, latent_encoder_name="madebyollin/sdxl-vae-fp16-fix", dtype=torch.float32):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(latent_encoder_name, torch_dtype=dtype)
        self.dtype = dtype
        self.vae_encoder = copy.deepcopy(vae.encoder)
        self.vae_quant_conv = copy.deepcopy(vae.quant_conv)
        self.vae_encoder.eval()
        self.vae_quant_conv.eval()
        del vae 

    def forward(self, x):
        h = self.vae_encoder(x)
        moments = self.vae_quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.sample()
        return latents


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def to_dtype(x, dtype):
    return x.to(dtype)

def make_collatable(x):
    return [x]

def get_filesystem(url):
    if url.startswith("s3://"):
        return fsspec.filesystem("s3")
    else:
        return fsspec.filesystem("file")

def get_the_writer(output_file: str):
    fs = get_filesystem(output_file)
    if output_file.startswith("s3://"):
        tar_fd = fs.open(output_file, "wb", s3={"profile": "writer"})
    else:
        tar_fd = fs.open(output_file, "wb")
    return wds.TarWriter(tar_fd)

def collate_fn(batch):
    dict = defaultdict(list)
    for sample in batch:
        for key, value in sample.items():
            dict[key].append(value)
    dict["image"] = torch.stack(dict["image"], dim=0)
    return dict

def write_result(shard, target_url, target_name):
    """ with s3 datasets this one ends up blocking after '3' """
    # write results to target_url
    out_fs, output_dir = fsspec.core.url_to_fs(target_url)
    if output_dir is not None and not out_fs.exists(output_dir):
        out_fs.mkdir(output_dir)
    sample_idx = 0
    with get_the_writer(os.path.join(target_url, target_name)) as sink:
        keys = list(shard[0].keys())
        for batch in shard:
            batch_size = len(batch['__key__'])
            for idx in range(batch_size):
                sample = {key: batch[key][idx] for key in keys}
                del sample["image"]
                buffer =  io.BytesIO()
                torch.save(sample["latent.pt"].cpu().detach().clone(), buffer)
                sample["json"] = json.dumps(sample["json"], cls=NumpyEncoder).encode("utf-8")
                sample["latent.pt"] = buffer.getvalue()
                sink.write(sample)
                sample_idx += 1
                if sample_idx % 100 == 0:
                    logging.info(f"Writer: Written {sample_idx} samples...")
    logging.info(f"Writer: Finished writing {target_url}{target_name}...")


def determine_batch_size(model, input_shape, lower_bound=1, upper_bound=1024, dtype=torch.float32):
    """
    Determine the maximum batch size for a given model using binary search.

    Parameters:
    - model: The PyTorch model for which to determine the batch size.
    - input_shape: Shape of a single input sample (without batch size).
    - lower_bound: The minimum batch size to consider.
    - upper_bound: The maximum batch size to consider.

    Returns:
    - The maximum batch size that can be used without running out of memory.
    """
    
    device = next(model.parameters()).device  # Get the device of the model
    model.eval()  # Set the model to evaluation mode

    while lower_bound <= upper_bound:
        # Take the midpoint of the current range as the current batch size
        mid_point = (lower_bound + upper_bound) // 2
        
        # Create a batch of random noise
        noise = torch.randn([mid_point] + list(input_shape)).to(dtype).to(device)
        
        try:
            # Try to forward pass the noise through the model
            with torch.no_grad():
                model(noise)
            # If successful, set the lower bound to the midpoint for the next iteration
            lower_bound = mid_point + 1
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # If there's an out-of-memory error, set the upper bound to the midpoint for the next iteration
                upper_bound = mid_point - 1
            else:
                # If there's another type of error, raise it
                raise e

    # The maximum batch size is the upper bound after the loop finishes
    return upper_bound


@app.command()
def process_data(
    input_urls: List[str] = typer.Argument(..., help="Path to the shard to fix", min=1),
    target_url: str = typer.Option(..., help="Target URL (S3 or local) to write the processed data."),
    batch_size: int = typer.Option(None, help="Batch size."),
    fp16: bool = typer.Option(False, help="Whether to use fp16."),
    image_size: int = typer.Option(512, help="Image size."),
    num_workers: int = typer.Option(32, help="Number of workers."),
    num_samples_per_shard: int = typer.Option(10000, help="Number of samples per shard."),
):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    writer_processes = []
    # Instantiate the model and move it to GPU
    dtype = torch.float32
    if fp16:
        dtype = torch.float16
    
    model = VAEEncoder(dtype=dtype)
    model = nn.DataParallel(model)
    model.eval()
    model.cuda()

    if batch_size is None:
        logging.info("Determining batch size...")
        batch_size = determine_batch_size(model, [3, image_size, image_size], dtype=dtype)
        logging.info(f"Batch size not specified. Using {batch_size} instead.")

    logging.info(f"Processing {len(input_urls)} shards...") 
    shard_idx = 0
    for input_url in input_urls:
        target_name = input_url.split('/')[-1]
        if input_url.startswith("s3://"):
            # actually this is not the recommended use anymore...
            input_url = f"pipe: aws s3 cp {input_url} -"

        logging.info(f"Processing {input_url}...")
        try: 
            dataset = (wds.WebDataset(input_url)
                        .select(filter_no_caption_or_no_image)
                        .decode("pilrgb", handler=log_and_continue)
                        .rename(image="jpg;png;jpeg;webp")
                        .map_dict(image=transforms.Compose([create_timm_transform(image_size),
                                                            transforms.ToTensor(), 
                                                            partial(to_dtype, dtype=dtype),
                                                            transforms.Normalize(mean=0.5, std=0.5)]))
            )

            dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
            batch_idx = 0
            shard = []
            shard_size = 0
            for d in dataloader:
                with torch.no_grad():
                    #with autocast(enabled=fp16):
                    outputs = model(d["image"].cuda())
                d['latent.pt'] = outputs.cpu()
                shard += [d]
                shard_size += len(d["__key__"])
                logging.info(f"Processed batch {batch_idx}")
                batch_idx += 1
                if shard_size > num_samples_per_shard:
                    target_name = f"{shard_idx:06d}.tar"
                    logging.info(f"Writing results to {target_url}{target_name}...")
                    p = Process(target=write_result, args=(shard, target_url, target_name))
                    p.start()
                    writer_processes.append(p)
                    shard = []
                    shard_size = 0
                    shard_idx += 1
            if shard_size > 0:
                target_name = f"{shard_idx:06d}.tar"
                logging.info(f"Writing results to {target_url}{target_name}...")
                p = Process(target=write_result, args=(shard, target_url, target_name))
                p.start()
                writer_processes.append(p)
                shard = []
                shard_size = 0
                shard_idx += 1
        except Exception as e:
            logging.error(f"Error processing {input_url}: {e}. Ignoring...")
            continue
    for p in writer_processes:
        p.join()

if __name__ == "__main__":
    app()
