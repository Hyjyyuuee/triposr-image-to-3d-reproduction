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

# ============================================
# CONFIG
# ============================================

DATASET_DIR = "ShapeNetCore"

OUTPUT_DIR = "evaluation_output"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MC_RESOLUTION = 256

REMOVE_BG = False
FOREGROUND_RATIO = 0.85

NUM_SAMPLE_POINTS = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ============================================
# LOAD MODEL
# ============================================

logging.info("Loading TripoSR model...")

model = TSR.from_pretrained(
    "./checkpoints",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

model.renderer.set_chunk_size(8192)

model.to(DEVICE)

logging.info("Model loaded.")

# ============================================
# BACKGROUND REMOVAL
# ============================================

if REMOVE_BG:
    rembg_session = rembg.new_session()
else:
    rembg_session = None

# ============================================
# FUNCTIONS
# ============================================

def preprocess_image(image_path):

    if REMOVE_BG:

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


def postprocess_mesh(mesh):

    mesh.remove_duplicate_faces()

    mesh.remove_degenerate_faces()

    mesh.remove_unreferenced_vertices()

    mesh.fill_holes()

    mesh = trimesh.smoothing.filter_laplacian(
        mesh,
        iterations=5
    )

    mesh.process(validate=True)

    return mesh


def load_mesh(mesh_path):

    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.Scene):

        mesh = trimesh.util.concatenate(
            tuple(
                g for g in mesh.geometry.values()
            )
        )

    return mesh


def sample_points_from_mesh(mesh, num_points=10000):

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

    chamfer = (
        np.mean(dist1 ** 2)
        + np.mean(dist2 ** 2)
    )

    return chamfer


# ============================================
# EVALUATION LOOP
# ============================================

sample_dirs = sorted(os.listdir(DATASET_DIR))

chamfer_scores = []

logging.info(f"Found {len(sample_dirs)} samples.")

for idx, sample_name in enumerate(sample_dirs):

    logging.info(
        f"[{idx+1}/{len(sample_dirs)}] Processing {sample_name}"
    )

    sample_path = os.path.join(
        DATASET_DIR,
        sample_name
    )

    image_path = os.path.join(
        sample_path,
        "image.jpg"
    )

    gt_mesh_path = os.path.join(
        sample_path,
        "model.obj"
    )

    # =====================================
    # FIND IMAGE
    # =====================================

    if not os.path.exists(image_path):

        logging.warning(
            f"Missing image: {image_path}"
        )

        continue

    if not os.path.exists(gt_mesh_path):

        logging.warning(
            f"Missing OBJ: {gt_mesh_path}"
        )

        continue

    # =====================================
    # PREPROCESS IMAGE
    # =====================================

    image = preprocess_image(image_path)

    # =====================================
    # RECONSTRUCTION
    # =====================================

    pred_mesh = reconstruct_mesh(image)

    # =====================================
    # POST-PROCESSING
    # =====================================

    pred_mesh = postprocess_mesh(pred_mesh)

    # =====================================
    # SAVE PREDICTED MESH
    # =====================================

    output_mesh_path = os.path.join(
        OUTPUT_DIR,
        f"{sample_name}_pred.obj"
    )

    pred_mesh.export(output_mesh_path)

    # =====================================
    # LOAD GT MESH
    # =====================================

    gt_mesh = load_mesh(gt_mesh_path)

    # =====================================
    # SAMPLE POINT CLOUDS
    # =====================================

    pred_points = sample_points_from_mesh(
        pred_mesh,
        NUM_SAMPLE_POINTS
    )

    gt_points = sample_points_from_mesh(
        gt_mesh,
        NUM_SAMPLE_POINTS
    )

    # =====================================
    # COMPUTE CHAMFER
    # =====================================

    chamfer = chamfer_distance(
        pred_points,
        gt_points
    )

    chamfer_scores.append(chamfer)

    logging.info(
        f"Chamfer Distance = {chamfer:.6f}"
    )

# ============================================
# FINAL RESULT
# ============================================

mean_chamfer = np.mean(chamfer_scores)

print("\n===================================")
print("FINAL RESULT")
print("===================================")

if REMOVE_BG:
    print("Background Removal: ENABLED")
else:
    print("Background Removal: DISABLED")

print(f"Mean Chamfer Distance: {mean_chamfer:.6f}")

print("===================================")