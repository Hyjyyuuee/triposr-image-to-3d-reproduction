import os
import logging
import numpy as np
import torch
import rembg
import trimesh

from PIL import Image
from scipy.spatial import cKDTree

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# =========================
# CONFIG
# =========================

DATASET_DIR = "ShapeNetCore"
OUTPUT_DIR = "evaluation_output"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MC_RESOLUTION = 256
FOREGROUND_RATIO = 0.85

NUM_SAMPLE_POINTS = 2048

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# =========================
# LOAD MODEL
# =========================

logging.info("Loading TripoSR model...")

model = TSR.from_pretrained(
    "./checkpoints",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

model.renderer.set_chunk_size(8192)
model.to(DEVICE)

logging.info("Model loaded.")

# =========================
# REMBG
# =========================

rembg_session = rembg.new_session()

# =========================
# FUNCTIONS
# =========================

def preprocess_image(image_path, remove_bg=True):

    if remove_bg:

        image = remove_background(
            Image.open(image_path),
            rembg_session
        )

        image = resize_foreground(
            image,
            FOREGROUND_RATIO
        )

        image = np.array(image).astype(np.float32) / 255.0

        image = (
            image[:, :, :3] * image[:, :, 3:4]
            + (1 - image[:, :, 3:4]) * 0.5
        )

        image = Image.fromarray(
            (image * 255.0).astype(np.uint8)
        )

    else:

        image = Image.open(image_path).convert("RGB")

    return image


def reconstruct_mesh(image):

    with torch.no_grad():

        scene_codes = model([image], device=DEVICE)

        meshes = model.extract_mesh(
            scene_codes,
            True,
            resolution=MC_RESOLUTION
        )

    return meshes[0]


def sample_points_from_mesh(mesh, num_points=2048):

    points, _ = trimesh.sample.sample_surface(
        mesh,
        num_points
    )

    return points


def chamfer_distance(points1, points2):

    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)

    cd = np.mean(dist1 ** 2) + np.mean(dist2 ** 2)

    return cd


def evaluate_setting(remove_bg=True):

    setting_name = (
        "REMOVE_BG"
        if remove_bg
        else "NO_REMOVE_BG"
    )

    logging.info(f"========== {setting_name} ==========")

    cd_list = []

    sample_dirs = sorted(os.listdir(DATASET_DIR))

    for idx, sample_name in enumerate(sample_dirs):

        logging.info(
            f"[{idx+1}/{len(sample_dirs)}] {sample_name}"
        )

        sample_path = os.path.join(
            DATASET_DIR,
            sample_name
        )

        image_path = os.path.join(
            sample_path,
            "image.jpg"
        )

        gt_obj_path = os.path.join(
            sample_path,
            "model.obj"
        )

        # preprocess
        image = preprocess_image(
            image_path,
            remove_bg=remove_bg
        )

        # reconstruct
        pred_mesh = reconstruct_mesh(image)

        # save mesh
        pred_mesh_path = os.path.join(
            OUTPUT_DIR,
            f"{sample_name}_{setting_name}.obj"
        )

        pred_mesh.export(pred_mesh_path)

        # load GT mesh
        gt_mesh = trimesh.load(
            gt_obj_path,
            force='mesh'
        )

        # sample points
        pred_points = sample_points_from_mesh(
            pred_mesh,
            NUM_SAMPLE_POINTS
        )

        gt_points = sample_points_from_mesh(
            gt_mesh,
            NUM_SAMPLE_POINTS
        )

        # compute CD
        cd = chamfer_distance(
            pred_points,
            gt_points
        )

        cd_list.append(cd)

        logging.info(
            f"{setting_name} CD = {cd:.6f}"
        )

    mean_cd = np.mean(cd_list)

    print("\n==========================")
    print(f"{setting_name}")
    print(f"Mean Chamfer Distance: {mean_cd:.6f}")
    print("==========================\n")

    return mean_cd


# =========================
# RUN EVALUATION
# =========================

cd_remove_bg = evaluate_setting(
    remove_bg=True
)

cd_no_remove_bg = evaluate_setting(
    remove_bg=False
)

# =========================
# FINAL COMPARISON
# =========================

print("\n===================================")
print("FINAL RESULT")
print("===================================")

print(f"With Background Removal    : {cd_remove_bg:.6f}")
print(f"Without Background Removal : {cd_no_remove_bg:.6f}")

print("===================================")