#!/usr/bin/env python
# coding: utf-8

# # prompt 实战

# ## Step1 导入相关包

# In[ ]:


from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer


# ## Step2 加载数据集

# In[ ]:


ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
ds


# In[ ]:


ds[:3]


# In[ ]:


# print(len("以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"))


# ## Step3 数据集预处理

# In[ ]:


#tokenizer = AutoTokenizer.from_pretrained("D:\pycharm\modelscope\Pretrained_models\modelscope\Llama-2-7b-ms")
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/modelscope/Pretrained_models/modelscope/Llama-2-7b-ms")
tokenizer


# In[ ]:


tokenizer.padding_side = "right"  # 一定要设置padding_side为right，否则batch大于1时可能不收敛


# In[ ]:


tokenizer.pad_token_id = 2
'''
# LLaMA2 的特殊标记 ID
SPECIAL_TOKEN_IDS = {
    0: "<unk>",    # 未知标记
    1: "<s>",      # 开始标记（bos）
    2: "</s>",     # 结束标记（eos）
    32000: "<pad>" # 填充标记（但需要手动添加）
}
# 当设置 pad_token_id = 2 时：
# 实际上是将 </s>（结束标记）用作填充标记
# 这是合理的，因为 </s> 通常出现在序列末尾

所以也可以写成：tokenizer.pad_token = tokenizer.eos_token
'''


# In[ ]:


def process_func(example):
    MAX_LENGTH = 1024    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ", add_special_tokens=False)
    response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# In[ ]:


tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
tokenized_ds


# In[ ]:


print(tokenized_ds["input_ids"][0])


# In[ ]:


# tokenizer("abc " + tokenizer.eos_token)


# In[ ]:


tokenizer.decode(tokenized_ds["input_ids"][0])


# In[ ]:


# tokenizer("呀", add_special_tokens=False) # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性


# In[ ]:


tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"])))


# ## Step4 创建模型

# In[ ]:


import torch
# 多卡情况，可以去掉device_map="auto"，否则会将模型拆开
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/modelscope/Pretrained_models/modelscope/Llama-2-7b-ms", low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")


# In[ ]:


model.dtype


# ## Prompt

# ### PEFT Step1 配置文件

# In[ ]:


from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,   #告诉模型这是生成式任务（对话、续写等）
                            prompt_tuning_init=PromptTuningInit.TEXT,   #指定提示词初始化方式
                            prompt_tuning_init_text="下面是一段人与机器人的对话。",   #初始化的具体文本内容,这就是一个 Hard Prompt
                            num_virtual_tokens=len(tokenizer("下面是一段人与机器人的对话。")["input_ids"]),   #自动计算提示词的token数量
                            tokenizer_name_or_path="/root/autodl-tmp/modelscope/Pretrained_models/modelscope/Llama-2-7b-ms")


# ### PEFT Step2 创建模型

# In[ ]:


model = get_peft_model(model, config)


# In[ ]:


#config


# In[ ]:


model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
'''
gradient_checkpointng
梯度检查点是一种用时间换空间的优化技术，用于解决深度学习训练中的显存瓶颈问题核心思想,使训练多花30%的时间，但是显存需求更低2/3
• 在标准训练中，前向传播需要保存所有中间激活值用于反向传播
• 这些激活值占用大量显存，尤其是大型模型和长序列
梯度检查点只保存部分激活值，在反向传播时重新计算丢失的激活值

float32：32位浮点数，占用4字节
float16：16位浮点数，占用2字节
'''


# In[ ]:


# model = model.half()  # 当整个模型都是半精度时，需要将adam_epsilon调大
# torch.tensor(1e-8).half() 


# In[ ]:


model.print_trainable_parameters()


# In[ ]:


from torch.utils.data import DataLoader


# In[ ]:


dl = DataLoader(tokenized_ds, batch_size=2, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))


# In[ ]:


ipt = next(enumerate(dl))[1]


# In[ ]:


ipt


# In[ ]:


model(**ipt.to("cuda"))


# ## Step5 配置训练参数

# In[ ]:


args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=1,
    gradient_checkpointing=True
)


# ## Step6 创建训练器

# In[ ]:


trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds.select(range(6000)),
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)


# ## Step7 模型训练

# In[ ]:


trainer.train()


# ## Step8 模型推理

# In[ ]:


model.eval()
ipt = tokenizer("Human: {}\n{}".format("你好", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)


# In[ ]:




