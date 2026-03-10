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
import evaluate
import re

# Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
chrf_metric = evaluate.load("chrf")
bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # Decode predictions and labels
    # BUG: `skip_special_tokens=True` causes an infinite loop
    # Remove special tokens manually
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

    preds_nosymbols = []
    labels_nosymbols = []

    for pred, label in zip(decoded_preds, decoded_labels):
        for special_token in tokenizer.all_special_tokens:
            pred = pred.replace(special_token, '')
            label = label.replace(special_token, '')
        preds_nosymbols.append(pred)
        labels_nosymbols.append(label)

    decoded_preds = preds_nosymbols
    decoded_labels = labels_nosymbols

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
    bleu_results = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "wer": wer,
        "cer": cer,
        "chrf": chrf_results["score"],
        "bleu": bleu_results["score"],
    }

def filter_texts(example) -> str:
    part_regex = re.compile('parts? of', re.IGNORECASE)
    return not part_regex.match(example['input_text'])

def preprocess_transcription(input_text: str) -> str:
    input_text = input_text.strip()
    return input_text

def preprocess_translation(output_text: str) -> str:
    output_text = output_text.strip().lower()
    parenthetical_rgx = r'\(.*?\)'
    output_text = re.sub(parenthetical_rgx, '', output_text)

    non_alphanum_rgx = r'[^A-Za-z0-9\s]'
    output_text = re.sub(non_alphanum_rgx, '', output_text)
    return output_text

def preprocess_texts(examples):
    examples['input_text'] = [
        preprocess_transcription(text)
        for text in examples['input_text']
    ]
    examples['output_text'] = [
        preprocess_translation(text)
        for text in examples['output_text']
    ]
    return examples

def preprocess_prompt(
        examples,
        prompt_template,
        expected_fields,
        tokenizer,
        max_length
):
    examples = preprocess_texts(examples)
    inputs_w_prompt = []
    batch_size = len(examples[expected_fields[0]])
    for i in range(batch_size):
        prompt_fields = {field: examples[field][i] for field in expected_fields}
        prompt = prompt_template.format(**prompt_fields)
        inputs_w_prompt.append(prompt)
    model_inputs = tokenizer(inputs_w_prompt, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=examples['output_text'], max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_mbart(examples, tokenizer, max_length):
    examples = preprocess_texts(examples)
    input_strs = examples['input_text']
    label_strs = examples['output_text']
    model_inputs = tokenizer(
        input_strs,
        text_target=label_strs,
        max_length=max_length,
        padding="max_length",
        truncation=True
    )

    return model_inputs

@hydra.main(version_base="1.3", config_path="../../conf/mbart", config_name="translation")
def main(cfg: DictConfig):
    # 1. Setup WandB
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)

    if cfg.data.task.format == "prompt_template":
        preprocess_fn = lambda examples: preprocess_prompt(
            examples=examples,
            prompt_template=cfg.data.task.prompt,
            expected_fields=cfg.data.task.expected_fields,
            tokenizer=tokenizer,
            max_length=cfg.model.max_length,
        )
    elif cfg.data.task.format == "mbart":
        tokenizer.src_lang = cfg.data.task.src_lang_code
        tokenizer.tgt_lang = cfg.data.task.tgt_lang_code
        preprocess_fn = lambda examples: preprocess_mbart(examples, tokenizer, cfg.model.max_length)
    else:
        raise ValueError(f"Unsupported task format: {cfg.data.task.format}")


    # 3. Load Data
    dataset = load_dataset(cfg.data.hf_uri)
    dataset = dataset.rename_columns({
        cfg.data.columns.input_column: "input_text",
        cfg.data.columns.target_column: "output_text"
    })
    dataset = dataset.filter(filter_texts)

    tokenized_dataset = dataset.map(
        lambda x: preprocess_fn(x),
        batched=True
    )

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_acc,
        num_train_epochs=cfg.training.epochs,
        weight_decay=cfg.training.weight_decay,
        predict_with_generate=True,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        eval_strategy="epoch",
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
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda eval_preds: compute_metrics(tokenizer, eval_preds),
    )

    # 6. Train
    trainer.evaluate()
    trainer.train()

    # save model separate from checkpoints for better organization
    print("Saving final model...")
    model_save_dir = os.path.join(cfg.training.output_dir, "model")
    trainer.save_model(model_save_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
