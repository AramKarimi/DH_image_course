# Image Processing & AI for Digital Humanities

A hands-on Jupyter course for Digital Humanities students: from classic image
processing to AI models to fake-image generation and detection — all in the
context of **museums and cultural heritage**.

## Course structure

| # | Notebook | Topics | Heritage angle |
|---|----------|--------|----------------|
| 1 | `01_image_basics.ipynb` | pixels, crop/resize/rotate, color spaces, histograms, CLAHE, filtering | restoring faded photos, removing dust & scratches |
| 2 | `02_classic_cv.ipynb` | edges (Sobel/Canny), thresholding (Otsu/adaptive), morphology, contours, template & ORB feature matching | binarizing inscriptions for OCR/HTR, locating details in artworks |
| 3 | `03_ai_classification.ipynb` | pretrained ResNet/VGG, the domain gap, Grad-CAM explainability, CLIP zero-shot, fine-tuning on your own data | classifying collections with curatorial vocabulary |
| 4 | `04_object_detection.ipynb` | YOLOv8, Faster R-CNN, confidence thresholds, FiftyOne dataset exploration | iconography at scale, archive auto-tagging, ethics of detecting people |
| 5 | `05_fake_generation.ipynb` | copy–move, splicing, inpainting, train a GAN from scratch, diffusion text-to-image | forged provenance, "newly discovered" artworks, disclosure ethics |
| 6 | `06_fake_detection.ipynb` | EXIF/C2PA forensics, Error Level Analysis, noise residuals, copy–move detection, FFT, neural detectors | the museum as an institution of authenticity |

Notebooks 5 and 6 are paired: the fakes you create in 5 (saved to
`data/fakes/`) are the evidence you analyze in 6.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (or plain `pip`) and ~6 GB of disk
(environment + model weights).

```bash
# 1. Create the environment (Python 3.12)
uv venv --python 3.12 .venv
uv pip install -r requirements.txt

# 2. Download the sample images (public domain, from Wikimedia Commons)
.venv/bin/python scripts/download_data.py

# 3. Start Jupyter
.venv/bin/jupyter lab
```

Then open the notebooks in order. Model weights (ResNet, CLIP, YOLO,
Stable Diffusion, …) download automatically on first run and are cached
in `~/.cache`.

## Project layout

```
aram/
├── notebooks/            # the six teaching notebooks (.ipynb)
│   └── *.py              # jupytext source files (same content, easy to diff/edit)
├── scripts/
│   └── download_data.py  # fetches sample images from Wikimedia Commons
├── data/
│   ├── raw/              # downloaded heritage images
│   ├── my_collection/    # PUT YOUR OWN IMAGES HERE for fine-tuning (NB 3)
│   │   └── <class_name>/ #   one subfolder per class
│   └── fakes/            # forgeries created in NB 5, analyzed in NB 6
├── outputs/              # images students save during exercises
└── requirements.txt
```

## Using your own data (Notebook 3 fine-tuning)

Organize images into class-named subfolders:

```
data/my_collection/
├── paintings/    img001.jpg ...
├── sculptures/   ...
└── manuscripts/  ...
```

If `my_collection/` is empty, Notebook 3 auto-builds a small demo dataset
from the Wikimedia images, so the class demo always works.

## Editing the notebooks

The `.py` files in `notebooks/` are [jupytext](https://jupytext.readthedocs.io)
sources — edit them in any editor and regenerate the notebooks with:

```bash
.venv/bin/jupytext --to ipynb notebooks/0*.py
```

## Notes for instructors

- **Library choice**: the course standardizes on PyTorch (torchvision,
  ultralytics, Hugging Face) rather than mixing in TensorFlow — one framework,
  fewer environment problems for students.
- **Heavy cells** are marked with ⏳ in the notebooks. The diffusion cell in
  NB 5 can be skipped (`RUN_DIFFUSION = False`) on weak machines.
- **Apple Silicon** Macs use the GPU automatically (MPS); NVIDIA machines use
  CUDA; everything also runs on CPU, just slower.
- All sample images are public-domain works from Wikimedia Commons; swap in
  your own institution's images by editing `scripts/download_data.py`.
