import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"  # Adjust path if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def classify_tactic(tactic_text):
    inputs = tokenizer(tactic_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    # Assuming 0 is negative, 1 is positive
    return "Good Tactic" if predicted_class == 1 else "Bad Tactic"

# Gradio interface
iface = gr.Interface(
    fn=classify_tactic,
    inputs=gr.Textbox(lines=5, placeholder="Enter soccer tactic description..."),
    outputs=gr.Textbox(),
    title="Soccer Tactics Classifier",
    description="Enter a description of a soccer tactic to classify it as good or bad."
)

if __name__ == "__main__":
    iface.launch()