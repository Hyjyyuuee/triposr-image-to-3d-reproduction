# TripoSR Image-to-3D Reproduction

A reproducible single-image-to-3D workflow based on
[TripoSR](https://github.com/VAST-AI-Research/TripoSR). The project accepts a
PNG/JPG image, reconstructs a 3D mesh, and exports an OBJ or GLB file that can
be opened in Blender or MeshLab.

## Verified result

The workflow was reproduced on Google Colab with an NVIDIA Tesla T4 GPU:

- model initialization: about 8 seconds;
- image-to-scene inference: about 2 seconds;
- mesh extraction at resolution 256: about 3 seconds;
- exported OBJ successfully opened in Blender 4.5 LTS.

Actual timing varies with the input image and Colab runtime.

## Colab workflow

Open `TripoSR_Colab.ipynb` in Google Colab and select a GPU runtime. Run the
cells in order to prepare the project, install compatible dependencies, upload
an image, generate a mesh, and download `mesh.obj`.

For best results, use a clear image containing one complete object against a
simple background. Images with several objects are reconstructed as one scene.

## Local usage

Python 3.10 or 3.11 is recommended. An NVIDIA GPU with at least 6 GB VRAM is
strongly preferred.

```bash
pip install -r requirements.txt
python run.py examples/chair.png --output-dir output/
```

To bake a texture atlas instead of using vertex colors:

```bash
python run.py examples/chair.png --output-dir output/ --bake-texture
```

## Repository contents

- `run.py` — command-line image-to-3D generation.
- `gradio_app.py` — local interactive interface.
- `tsr/` — reconstruction model implementation.
- `examples/` — sample input images.
- `TripoSR_Colab.ipynb` — Colab reproduction notebook.

Large checkpoints, datasets, generated meshes, and local outputs are excluded
from Git. Model assets must be downloaded separately or prepared by the Colab
workflow.

## Credits and license

This reproduction is based on the official TripoSR project developed by Tripo
AI and Stability AI. The original code and model are released under the MIT
License. See `LICENSE` and the
[official TripoSR repository](https://github.com/VAST-AI-Research/TripoSR).
