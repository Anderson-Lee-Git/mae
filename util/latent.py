import os

import numpy as np
import torch
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

def save_latent(latent_batch: torch.Tensor, paths: list[str]):
    latent_batch = latent_batch.cpu().detach().numpy()
    for idx, latent in enumerate(latent_batch):
        # save latent
        save_path = process_latent_path(paths[idx])
        np.save(save_path, latent)

def process_latent_path(image_path):
    if image_path.endswith("/"):
        image_path = image_path[:-1]
    tokens = image_path.split("/")
    image_fname = tokens[-1]
    class_id = tokens[-2]
    assert image_fname.find(".JPEG") != -1
    latent_fname = image_fname[:image_fname.find(".JPEG")] + ".npy"
    assert latent_fname.startswith(class_id)
    class_folder = os.path.join(config("TINYIMAGENET_TRAIN_LATENT_PATH"), class_id)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    return os.path.join(class_folder, latent_fname)