from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig, AutoTokenizer
import os

#Generate a dummy fine-tuned ESM Model
# model_name = "esm2_t30_150M_UR50D"
# model_checkpoint = f"facebook/{model_name}"
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)
# model.save_pretrained("models/dummy", from_pt=True)

#For fine-tuned models:
model_path = "models/dummy/"
model_name = "dummy_model"
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

test_sequence = "MTNLDYF"

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
tokenized = tokenizer(test_sequence, return_tensors="pt")["input_ids"]

print(model(tokenized))