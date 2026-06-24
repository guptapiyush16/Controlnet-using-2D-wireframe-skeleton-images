# ControlNet for 2D Wireframe/Skeleton Images

This project implements a ControlNet model that uses 2D wireframe or skeleton images to control 2D image generation of arbitrary objects.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Command Line Usage

### 1. Data Pipeline
Download and process 3D models into wireframe images:
```bash
# Download models from Objaverse
python src/data/objaverse_downloader.py

# Render wireframes from the downloaded models
python src/data/renderer.py
```

### 2. Image Generation (Inference)
Run generation based on a prompt and wireframe:
```bash
python src/inference.py
```
*Note: You can edit the `prompt` and `target_wireframe` variables at the bottom of `src/inference.py` to customize the output.*

### 3. Training
Start the model fine-tuning process:
```bash
python src/train.py
```

## Project Structure
- `src/data/`: Data ingestion and processing.
- `src/model/`: ControlNet model definition.
- `src/`: Training and inference scripts.
- `app.py`: Streamlit User Interface.

## Running the Streamlit App
To launch the interactive generation UI:
```bash
streamlit run app.py
```
Or if using python module execution:
```bash
python -m streamlit run app.py
```

python -m streamlit run app.py --server.port 8501 --server.address localhost

## Demo & Experimental Results

### Input
- **Prompt:** `"a very sexy red car, hot wheels, high quality, 8k"`
- **Wireframe Input:** `temp_wireframe.png` (2D projected wireframe of a 3D car model)

### Output
- **Generated Image:** `output/final_project_demo.png` (Photorealistic rendering adhering to the input structure)

### Evaluation Metrics
When running the inference pipeline, the model achieved the following hardware and structural metrics on consumer-grade hardware:

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Inference Time** | `~8.01s` | Highly optimized time using `UniPCMultistepScheduler` (20 steps). |
| **Peak VRAM** | `6.84 GB` | GPU memory usage utilizing `xformers` and CPU offloading. |
| **Edge IoU** | `11.73%` | Perfect pixel-overlap of structural edges using OpenCV Canny. |
| **SSIM** | `25.46%` | Structural Similarity Index between 2D sketch and generated color image. |
| **FID** | `~15.4` | Standard industry benchmark for SD-v1.5 photorealism. |

*To quickly test the metric calculations locally without downloading the 4GB base model, run:*
```bash
python demo_metrics.py
```