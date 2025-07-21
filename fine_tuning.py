import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from functions.create_data_file import create_data_file

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, but recommended


def get_model(model_name):
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY not set in environment or .env file.")
    login(api_key)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)
    return model,tokenizer

def formatting_func(example):
    return f"<s>[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']}</s>"


def crete_trainer(model):
    # Config
    training_args = SFTConfig(
        output_dir="weights",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        logging_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_strategy="epoch",   # or "epoch" / "steps"
        report_to=[],  # disable wandb if not using it
        remove_unused_columns=False,
        
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        
        formatting_func=formatting_func,
    )
    return trainer


if __name__ == "__main__":
    create_data_file()
    dataset = load_dataset("json", data_files="qlora_ready_data.json")
    dataset = dataset["train"]
    # Shuffle dataset
    dataset = dataset.shuffle(seed=42)
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    model, = get_model(model_name)
    trainer = crete_trainer(model)
    trainer.train()
