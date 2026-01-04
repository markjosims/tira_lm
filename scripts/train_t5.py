import hydra
import wandb
import os
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import numpy as np
from evaluate import load_metric

# Load metrics
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")
chrf_metric = load_metric("chrf")

def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple post-processing: strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Calculate metrics
    # ChrF expects a list of references (each reference is a list of strings)
    # but for single-reference tasks, evaluate handles the nested list internally
    chrf_results = chrf_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels,
        word_order=2 # This enables chrF++, which includes word n-grams
    )
    
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "wer": wer,
        "cer": cer,
        "chrf": chrf_results["score"],
    }

def preprocess_function(examples, prompt_template, tokenizer, max_length):
    inputs_w_prompt = [
        prompt_template.format(
            input_text=example['input_text'],
            translation=example.get('translation', '')
        ) for example in examples
    ]
    model_inputs = tokenizer(inputs_w_prompt, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=examples["output_text"], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup WandB
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)

    # 3. Load Data
    dataset = load_dataset(cfg.data.hf_uri)
    dataset = dataset.rename_columns({
        cfg.data.input_column: "input_text",
        cfg.data.target_column: "output_text"
    })
    
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, cfg.data.task.prompt, tokenizer, cfg.model.max_length), 
        batched=True
    )

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_acc,
        num_train_epochs=cfg.training.epochs,
        weight_decay=0.01,
        predict_with_generate=True,
        bf16=cfg.training.bf16,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.training.save_total_limit,
        report_to="wandb",
        run_name=cfg.wandb.run_name
    )

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda eval_preds: compute_metrics(tokenizer, eval_preds),
    )

    # 6. Train
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    main()