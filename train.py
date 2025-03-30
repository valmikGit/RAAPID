import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# Access secret
user_secrets = UserSecretsClient()
hugging_face_access_token = user_secrets.get_secret("hugging_face_access_token")

# Authenticate with Hugging Face
login(token=hugging_face_access_token)

# Load dataset
train_df = pd.read_csv('/kaggle/input/raapid-inc-task-a-and-b/inference_dataset/inference_dataset/train.tsv', sep='\t')
val_df = pd.read_csv('/kaggle/input/raapid-inc-task-a-and-b/inference_dataset/inference_dataset/dev.tsv', sep='\t')
test_df = pd.read_csv('/kaggle/input/raapid-inc-task-a-and-b/inference_dataset/inference_dataset/test.tsv', sep='\t')

def extract_raw_sentence(parsed_sentence):
    # Implement extraction logic here (Placeholder)
    return parsed_sentence  # Replace with actual parsing logic

# Extract raw sentences
train_df['premise'] = train_df['Sent1_parse'].apply(extract_raw_sentence)
train_df['hypothesis'] = train_df['Sent2_parse'].apply(extract_raw_sentence)
val_df['premise'] = val_df['Sent1_parse'].apply(extract_raw_sentence)
val_df['hypothesis'] = val_df['Sent2_parse'].apply(extract_raw_sentence)
test_df['premise'] = test_df['Sent1_parse'].apply(extract_raw_sentence)
test_df['hypothesis'] = test_df['Sent2_parse'].apply(extract_raw_sentence)

# Convert labels to integers, handling NaN values
label_mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
train_df['label'] = train_df['Label'].map(label_mapping)
val_df['label'] = val_df['Label'].map(label_mapping)
test_df['label'] = test_df['Label'].map(label_mapping)

# Drop rows with NaN labels before conversion
train_df.dropna(subset=['label'], inplace=True)
val_df.dropna(subset=['label'], inplace=True)
test_df.dropna(subset=['label'], inplace=True)

# Convert labels to integers
train_df['label'] = train_df['label'].astype(int)
val_df['label'] = val_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

# Convert to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df[['premise', 'hypothesis', 'label']])
val_dataset = Dataset.from_pandas(val_df[['premise', 'hypothesis', 'label']])
test_dataset = Dataset.from_pandas(test_df[['premise', 'hypothesis', 'label']])

# Load tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=128)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.config.problem_type = "single_label_classification"  # Explicitly set classification type

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    save_strategy="epoch",
    push_to_hub=False,
    report_to="none",
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
trainer.save_model("/kaggle/working/nli_bert_model")
tokenizer.save_pretrained("/kaggle/working/nli_bert_model")

# Evaluate model
val_results = trainer.evaluate(eval_dataset=val_dataset)
print(f"Validation Accuracy: {val_results.get('eval_accuracy', 'N/A')}")

test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A')}")

# Save predictions
predictions = trainer.predict(test_dataset)
pred_df = test_df[['Sent1_parse', 'Sent2_parse', 'Label']]
pred_df['Prediction'] = predictions.predictions.argmax(axis=1)
pred_df.to_csv('/kaggle/working/test_predictions.tsv', sep='\t', index=False)
