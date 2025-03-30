    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")

    ### STEP 1: LOAD DATASETS ###

    # Load validation dataset (dev.tsv)
    print("🔹 Loading validation dataset (dev.tsv)...")
    dev_df = pd.read_csv('./inference_dataset/inference_dataset/dev.tsv', sep='\t')
    print(f"✅ Loaded validation dataset with {len(dev_df)} samples.")

    # Load test dataset (test.tsv)
    print("🔹 Loading test dataset (test.tsv)...")
    test_df = pd.read_csv('./inference_dataset/inference_dataset/test.tsv', sep='\t')
    print(f"✅ Loaded test dataset with {len(test_df)} samples.")

    ### STEP 2: PREPROCESS DATA ###

    def extract_raw_sentence(parsed_sentence):
        return parsed_sentence  # Replace with actual parsing logic

    # Extract raw sentences
    print("🔹 Extracting raw sentences...")
    for df in [dev_df, test_df]:
        df['premise'] = df['Sent1_parse'].apply(extract_raw_sentence)
        df['hypothesis'] = df['Sent2_parse'].apply(extract_raw_sentence)

    # Convert labels
    print("🔹 Mapping labels...")
    label_mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    for df in [dev_df, test_df]:
        df['label'] = df['Label'].map(label_mapping)
        df.dropna(subset=['label'], inplace=True)
        df['label'] = df['label'].astype(int)

    print(f"✅ Labels mapped. Validation: {len(dev_df)} samples, Test: {len(test_df)} samples.")

    # Convert to Hugging Face Dataset object
    print("🔹 Converting to Hugging Face Dataset object...")
    dev_dataset = Dataset.from_pandas(dev_df[['premise', 'hypothesis', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['premise', 'hypothesis', 'label']])
    print("✅ Conversion complete.")

    ### STEP 3: LOAD TOKENIZER & MODEL ###

    model_path = "./NLI_bert_model"  # Change to your local model directory
    print(f"🔹 Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    print("✅ Model loaded.")

    def tokenize_function(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=128)

    # Tokenize datasets
    print("🔹 Tokenizing datasets (this may take some time)...")
    start_time = time.time()
    dev_dataset = dev_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    print(f"✅ Tokenization complete. Time taken: {time.time() - start_time:.2f}s")

    ### STEP 4: VALIDATION (ON dev.tsv) ###

    print("🔹 Running validation (on dev.tsv)...")
    trainer = Trainer(model=model)
    start_time = time.time()
    val_results = trainer.evaluate(dev_dataset)
    print(f"✅ Validation complete. Time taken: {time.time() - start_time:.2f}s")
    print(f"📊 Validation Accuracy: {val_results.get('eval_accuracy', 'N/A')}")

    ### STEP 5: TEST INFERENCE (ON test.tsv) ###

    print("🔹 Running inference on test dataset (test.tsv)...")
    start_time = time.time()
    test_results = trainer.predict(test_dataset)
    print(f"✅ Inference complete. Time taken: {time.time() - start_time:.2f}s")

    # Compute accuracy (since test set has labels)
    test_predictions = test_results.predictions.argmax(axis=1)
    test_accuracy = (test_predictions == test_df['label'].values).mean()
    print(f"📊 Test Accuracy: {test_accuracy:.4f}")

    ### STEP 6: SAVE TEST PREDICTIONS ###

    print("🔹 Saving predictions to test_predictions.tsv...")
    pred_df = test_df[['Sent1_parse', 'Sent2_parse', 'Label']].copy()
    pred_df['Prediction'] = test_predictions
    pred_df.to_csv('test_predictions.tsv', sep='\t', index=False)
    print("✅ Predictions saved.")
