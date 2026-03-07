
from omegaconf import DictConfig
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from datasets import load_dataset
import hydra
from evaluate import load
from jiwer import wer, cer
import os
from argparse import ArgumentParser
bleu = load("sacrebleu")


def preprocess_function(examples):
    input_strs = examples['sentence']
    model_inputs = tokenizer(input_strs, max_length=128, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        label_strs = examples['translation']
        labels = tokenizer(label_strs, max_length=128, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup WandB
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)

    # 3. Load Data
    dataset = load_dataset(cfg.data.hf_uri)
    dataset = dataset.rename_columns({
        cfg.data.columns.input_column: "input_text",
        cfg.data.columns.target_column: "output_text"
    })

def main():
    print("="*40)
    print("🚀 Tira Translation Model Training Setup")
    print("="*40)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")
    # Enable Mixed Precision (fp16) only if CUDA is available
    fp16_enabled = device == "cuda"

    args = parse_args()

    task_name = f'mbart_ft_{args.dataset}'
    os.environ['TASK_NAME'] = task_name
    os.environ['WANDB_PROJECT'] = 'tira_mt_asru2025'
    
    if fp16_enabled:
        print("FP16 Mixed Precision: ENABLED (Crucial for VRAM savings)")

    print("\n📂 Loading tokenized data from disk...")

    print(f"\n⬇️  Loading base mBART model: {MODEL_CHECKPOINT}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # Load tokenizer to save with the model later
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    print("\n🎯 Defining metrics...")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        blue_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)

        decoded_labels = [label[0] for label in decoded_labels]

        if args.print_outputs:
            print("\nSample Predictions vs References:")
            for i in range(min(5, len(decoded_preds))):
                print(f"Predicted: {decoded_preds[i]}")
                print(f"Reference: {decoded_labels[i]}\n")

        wer_score = wer(decoded_labels, decoded_preds)
        cer_score = cer(decoded_labels, decoded_preds)
        blue_score = {
            "bleu": blue_score["score"],
            "wer": wer_score,
            "cer": cer_score,
        }
        return blue_score

    print("\n⚙️  Configuring training parameters...")

    batch_size = 16
    gradient_accumulation = 2

    output_dir = os.path.join(MODEL_DIR, task_name)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=20,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=fp16_enabled,
        logging_steps=50,
        report_to="wandb",
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="bleu",
        run_name=task_name,
        resume_from_checkpoint=True,
        # dataloader_num_workers=num_workers
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"\n🔥 STARTING FINE-TUNING 🔥")
    print(f"Batch Size: {batch_size} | Accumulation: {gradient_accumulation}")
    trainer.train()

    print(f"\n✅ Training finished! Saving final model to: /{output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
