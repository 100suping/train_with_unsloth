from utils import get_bnb_config, get_current_gpu_memory
from data import get_dataset

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, set_seed
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only


def get_model_and_tokenizer(config):
    
    if config.quant_bit:
        bnb_config = get_bnb_config(bit=config.quant_bit)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            dtype=None,
            quantization_config=bnb_config,
            # max_seq_length=2048,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            dtype=None,
            # max_seq_length=2048,
        )
    if config.verbose:
        
        print(model)
        print(f'unsloth model type: {type(model)}')
        print(f'unsloth tokenizer type: {type(tokenizer)}')
        print(f"eos_token: {tokenizer.eos_token}")
        print(f"special_tokens_map: {tokenizer.special_tokens_map}")
    get_current_gpu_memory("Model Loaded", 0)
    return model, tokenizer

def reconstruct_model(config, model):
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = config.lora_alpha,
        lora_dropout = config.lora_dropout, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = config.seed,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model

def get_trainer(config, model, tokenizer, dataset):
    
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = config.max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
            per_device_train_batch_size = config.batch_size,
            gradient_accumulation_steps = config.gradient_accumulation_steps,
            warmup_steps = config.warmup_steps,
            num_train_epochs = config.epochs, # Set this for 1 full training run.
            # max_steps = config.max_steps,
            learning_rate = config.lr,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = config.logging_steps,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = config.seed,
            output_dir = config.output_dir,
            save_strategy="steps",
            save_steps=config.save_steps,
            report_to = config.report_to, # Use this for WandB etc
        ),
    )
    
    return trainer

def do_it_tuning(config):
    
    set_seed(config.seed)
    # 모델, 토크나이저 불러오기
    model, tokenizer = get_model_and_tokenizer(config=config)
    # 모델 재구축 - LoRA
    model = reconstruct_model(config=config, model=model)
    # 모델에 맞는 chat template 적용
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = config.model_type,
    )
    
    # 데이터셋 불러오기
    dataset = get_dataset(config=config, tokenizer=tokenizer)
    
    # SFTTrainer 생성
    trainer = get_trainer(config=config, model=model, tokenizer=tokenizer, dataset=dataset)
    
    # 데이터 중 LLM의 답변만 학습하도록 설정
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    
    trainer.train()