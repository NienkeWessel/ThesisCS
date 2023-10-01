'''
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
import random

# Set a seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# Load the tokenizer and pre-trained model
tokenizer = RobertaTokenizerFast.from_pretrained("javirandor/passgpt-10characters",
                                                 max_len=12, padding="max_length",
                                                 truncation=True, do_lower_case=False,
                                                 strip_accents=False, mask_token="<mask>",
                                                 unk_token="<unk>", pad_token="<pad>",
                                                 truncation_side="right")
model = RobertaForSequenceClassification.from_pretrained("javirandor/passgpt-10characters")

# Define your own classification head
num_classes = 2  # You can adjust this based on your classification task
classification_head = torch.nn.Sequential(
    torch.nn.Linear(model.config.hidden_size, num_classes),
)

# Set the model's classification head
model.classifier = classification_head

# Define your classification task data (replace this with your own dataset)
# Example: pairs of (text, label)
# Assuming "Password" is label 1 and "Normal" is label 0
training_data = [
    ("password123", 1),
    ("securepass", 1),
    ("example", 0),
    ("123456", 1),
    ("secret", 0),
]

# Tokenize and prepare the dataset
encoded_data = tokenizer(
    [example[0] for example in training_data],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

# Create a PyTorch dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = CustomDataset(encoded_data, [example[1] for example in training_data])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    seed=seed,
)

# Create a Trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model()

# You can now use the fine-tuned model for classification tasks
'''


from datasets import Dataset
import torch

# Sample data: list of embeddings and PyTorch tensors for labels
data = {
    "embeddings": [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ],
    "labels": [
        torch.tensor([1,0]),
        torch.tensor([0,1]),
        torch.tensor([1,0]),
    ],
}

# Create a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Print the dataset
print(dataset)