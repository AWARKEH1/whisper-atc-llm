"""LoRA fine-tuning of TinyLlama on ATC transcripts (domain LM for fusion / reranking).

Trains on the TRAIN split, early-stops on the VALIDATION split; the test split
is never seen.

    python -m atc_asr.finetune_lm --metadata atcosim.csv --out models/tinyllama-atcosim
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    args = p.parse_args()

    import pandas as pd
    import torch
    from datasets import Dataset, DatasetDict
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    df = pd.read_csv(args.metadata)
    df = df[df["text"].notna()]
    split = df["split"].astype(str).str.lower()
    data = DatasetDict({name: Dataset.from_dict({"text": df[split == name]["text"].astype(str).tolist()})
                        for name in ("train", "validation")})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # a pad token different from EOS, so that the collator does not mask the EOS the model must learn
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    def tokenize(batch):
        return tokenizer([t + tokenizer.eos_token for t in batch["text"]],
                         max_length=args.max_length, truncation=True)

    data = data.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.rank,
                                             lora_alpha=4 * args.rank, lora_dropout=0.05, bias="none"))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, num_train_epochs=args.epochs, eval_strategy="epoch",
            save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
            logging_steps=50, report_to=[]),
        train_dataset=data["train"], eval_dataset=data["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
