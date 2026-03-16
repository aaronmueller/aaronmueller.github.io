import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M" 

def load_and_finetune(output_dir="./smollm2_finetuned"):
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load WMT dataset (mocking the specific dataset loading line)
    # In reality you would load the formatted WMT data here.
    # dataset = load_dataset('wmt14', 'de-en', split='train[:10000]') 
    
    # TODO: implement a function that loads the WMT data in a format that
    # the Trainer can handle.
    # STUDENT START -----------------------------------------
    def preprocess_function(examples):
        # This would mimic the X ≡|≡ Y format used in the homework
        inputs = [ex['en'] + " ≡|≡ " + ex['de'] for ex in examples['translation']]
        return tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    # STUDENT END ------------------------------------------

    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    dev_dataset = dev_dataset.map(preprocess_function, batched=True)

    # Feel free to modify these hyperparameters.
    # To make things easier, we recommend only tuning the
    # learning_rate and max_steps hyperparameters.
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,  # [cite: 142]
        num_train_epochs=1,
        max_steps=1000,      # [cite: 142]
        per_device_train_batch_size=4,
        save_steps=500,
        logging_steps=100,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # Pass processed dataset here
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting fine-tuning...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    load_and_finetune()