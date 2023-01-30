# 对模型进行MLM预训练
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertForPreTraining
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import math

model_dir = "/Users/admin/Desktop/bert-base-chinese-model"
train_file = "../travel_data/sentences2train.txt"
eval_file = "../travel_data/sentences2dev.txt"
max_seq_length = 512
out_model_path = "../model/pretrain.pkl"
train_epoches = 20
batch_size = 16

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
tokenizer = AutoTokenizer.from_pretrained(out_model_path, use_fast=True)

model = AutoModelForMaskedLM.from_pretrained(out_model_path)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=201,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=201,
)

training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

if __name__ == '__main__':
    trainer.train()

    trainer.save_model(out_model_path)
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
