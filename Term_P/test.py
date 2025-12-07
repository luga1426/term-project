import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import os
os.environ["WANDB_DISABLED"] = "true"  # wandb 끄기

# 1. 설정
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-ko-en"
DATASET_NAME = "lemon-mint/korean_english_parallel_wiki_augmented_v1"
OUTPUT_DIR = "./ko-en-finetuned-model"

# 2. 데이터셋 로드
print("Dataset Loading")
raw_datasets = load_dataset(DATASET_NAME)

# 검증 데이터셋 나누기
if 'validation' not in raw_datasets:
    split = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
    raw_datasets = split
    raw_datasets['validation'] = raw_datasets.pop('test')

# 3. 전처리 (이 부분이 핵심 수정 사항입니다!)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess_function(examples):
    inputs = []
    targets = []
    
    # [수정 1] 데이터셋이 {'translation': {'ko': '...', 'en': '...'}} 구조일 경우 처리
    if "translation" in examples:
        inputs = [ex['ko'] for ex in examples["translation"]]
        targets = [ex['en'] for ex in examples["translation"]]
    else:
        # 혹시 구조가 다를 경우를 대비한 안전 장치
        inputs = examples.get('ko', examples.get('korean', []))
        targets = examples.get('en', examples.get('english', []))

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Data Preprocessing...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# 4. 모델 및 학습 설정
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",  # [수정 2] evaluation_strategy -> eval_strategy 로 변경
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,     # 빠른 테스트를 위해 1번만 학습
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 5. 학습 시작
print("Training Start!")
trainer.train()

# 6. 저장
save_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model has been saved: {save_path}")

