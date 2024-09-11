from datasets import load_metric
from datasets import load_dataset
from transformers import AdamW
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers import BertForSequenceClassification

dataset = load_dataset("liar")  # LIAR dataset is often used for fake news detection
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create data loaders
train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=8, collate_fn=data_collator)


# Load pre-trained BERT for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Move the model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")


metric = load_metric("accuracy")

# Evaluation loop
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

# Print accuracy
accuracy = metric.compute()
print(f"Accuracy: {accuracy['accuracy']}")

model.save_pretrained("./fake_news_classifier")
tokenizer.save_pretrained("./fake_news_classifier")