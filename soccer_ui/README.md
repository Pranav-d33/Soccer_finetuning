# Soccer Tactics Classifier UI

This is a Gradio-based UI for classifying soccer tactics using a fine-tuned model.

## Setup

1. Ensure the fine-tuned model is saved in the `./fine_tuned_model` directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `python app.py`

## Deployment to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose Gradio as the SDK.
3. Upload the `app.py`, `requirements.txt`, and the `fine_tuned_model` folder.
4. The app will auto-deploy.

## Usage

Enter a description of a soccer tactic, and the model will classify it as "Good Tactic" or "Bad Tactic".