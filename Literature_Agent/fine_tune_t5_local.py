import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Device setup
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[*] Using device: {device}")


# -------------------------------------------------
# Load dataset (small subset for testing)
# -------------------------------------------------
logger.info("[*] Loading dataset (ccdv/arxiv-summarization subset)...")
dataset = load_dataset("ccdv/arxiv-summarization")
train_data = dataset["train"].select(range(1000))
val_data = dataset["validation"].select(range(200))


# -------------------------------------------------
# Load base model and tokenizer
# -------------------------------------------------
model_name = "google/flan-t5-base"
logger.info(f"[*] Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# -------------------------------------------------
# Apply LoRA configuration
# -------------------------------------------------
logger.info("[*] Applying LoRA configuration...")
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q", "v"],  # attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# -------------------------------------------------
# Preprocess the data
# -------------------------------------------------
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(
        text_target=examples["abstract"],
        max_length=150,
        truncation=True
    )
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    return model_inputs


logger.info("[*] Tokenizing dataset...")
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=["article", "abstract"])
tokenized_val = val_data.map(preprocess_function, batched=True, remove_columns=["article", "abstract"])


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# -------------------------------------------------
# Define metrics
# -------------------------------------------------
rouge = evaluate.load("rouge")

import numpy as np

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    labels = [[(token if token != -100 else tokenizer.pad_token_id) for token in label] for label in labels]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v * 100, 4) for k, v in result.items()}




# -------------------------------------------------
# Training Arguments
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results_flan_t5_lora",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs_flan_t5_lora",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    report_to="none",
)


# -------------------------------------------------
# Trainer setup
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# -------------------------------------------------
# Train the model
# -------------------------------------------------
logger.info("[*] Starting LoRA fine-tuning...")
trainer.train()
logger.info("[*] Training complete!")


# -------------------------------------------------
# Save model & tokenizer
# -------------------------------------------------
save_dir = "./flan_t5_lora_finetuned"
logger.info(f"[*] Saving fine-tuned model to {save_dir}...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


logger.info("[✅] Done! LoRA fine-tuned model saved successfully.")
