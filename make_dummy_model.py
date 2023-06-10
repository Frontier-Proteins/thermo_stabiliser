from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
import os

#Check for output directory and create it if it doesn't exist
if not os.path.exists("models/"):
    os.mkdir("models")

if not os.path.exists("models/dummy"):
    os.mkdir("models/dummy")

#Load the model
# model_name = "esm2_t30_150M_UR50D"
# model_checkpoint = f"facebook/{model_name}"
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)
# model.save_pretrained("models/dummy", from_pt=True)

#For fine-tuned models:
model_path = "models/dummy/"
model_name = "dummy_model"
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)