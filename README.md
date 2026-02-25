# ControlNet for 2D Wireframe/Skeleton Images

This project implements a ControlNet model that uses 2D wireframe or skeleton images to control 2D image generation of arbitrary objects.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download data from Objaverse:
   ```bash
   python src/data/objaverse_downloader.py
   ```
3. Process data (render images and wireframes):
   ```bash
   python src/data/renderer.py
   ```
4. Start training:
   ```bash
   python src/train.py
   ```

## Project Structure
- `src/data/`: Data ingestion and processing.
- `src/model/`: ControlNet model definition.
- `src/`: Training and inference scripts.
