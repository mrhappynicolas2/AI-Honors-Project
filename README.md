# AI Honors Project – Lung CT Scan Classification & Generation

This project focuses on the classification and generation of lung CT scan images using deep learning models. It includes notebooks for training, scripts for image generation, and a web app for interactive usage.

---

### Models
Pretrained models are hosted on Hugging Face:  
👉 [mrhappynicolas/BenginLast](https://huggingface.co/mrhappynicolas/BenginLast)

---

## Project Structure

- **`classification_lung.ipynb`**  
  Trains a classification model to identify lung conditions. The resulting model (`model.pth`) is used in the web app.

- **`lungCTAI.ipynb`**  
  Trains models for generating synthetic lung CT scans. These trained models are uploaded to Hugging Face.

- **`app.py`**  
  The web app that integrates both classification and generation models. It serves as the main interface for users to interact with the AI.

- **`requirements.txt`**  
  Contains all necessary Python dependencies. Install with:
  ```bash
  pip install -r requirements.txt
