from huggingface_hub import login
import os
from datasets import load_dataset, Dataset, concatenate_datasets


def get_conversation_data(examples):
    GENERAL_QUERY_PREFIX = "당신은 사용자의 입력을 MySQL 쿼리문으로 바꾸어주는 조직의 팀원입니다. 당신의 임무는 DB 이름 그리고 DB내 테이블의 메타 정보가 담긴 아래의 (context)를 이용해서 주어진 질문(user_question)에 걸맞는 MySQL 쿼리문을 작성하는 것입니다.\n\n(context)\n{context}\n"
    GENERATE_QUERY_INSTRUCTIONS = "\n주어진 질문(user_question)에 대해서 문법적으로 올바른 MySQL 쿼리문을 작성해 주세요.\n"
    questions = examples['question']
    schemas =examples['schema']
    sql_queries =examples['SQL']
    convos = []
    for question, schema, sql in zip(questions, schemas, sql_queries):
        conv = [
        {"role": "system", "content": GENERAL_QUERY_PREFIX.format(context=schema) + GENERATE_QUERY_INSTRUCTIONS},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "```sql\n"+sql+";\n```"}
        ]
        convos.append(conv)
    return {"conversation":convos,}


def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversation"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    
    return { "text" : texts, }


def get_tokenized_length(examples, tokenizer):
    convos = examples["conversation"]
    lengths = []
    for convo in convos:
        text = tokenizer.apply_chat_template(convo, tokenize = True, add_generation_prompt = False)
        lengths.append(len(text))
        
    return {"length": lengths,}


def get_dataset(config, tokenizer):
    login(token=os.getenv("HF_TOKEN"))
    kobird_dataset = load_dataset(config.dataset_path, split="train")
    won75 = load_dataset("won75/text_to_sql_ko", split="train")

    won75_selected = won75.select_columns(['TEXT', "MySQL", "Schema"])
    won75_selected = won75_selected.rename_column("TEXT", "question")
    won75_selected = won75_selected.rename_column("MySQL", "SQL")
    won75 = won75_selected.rename_column("Schema", "schema")

    # config의 test_run 값에 따라서 데이터를 얼마나 불러올지 결정
    if config.test_run:
        kobird_dataset = kobird_dataset.select(range(100))
        won75 = won75.select(range(100))

    combined_dataset = concatenate_datasets([kobird_dataset, won75])
    combined_dataset = combined_dataset.map(lambda x : get_conversation_data(x), batched=True)
    combined_dataset = combined_dataset.map(lambda x : formatting_prompts_func(x, tokenizer), batched=True)
    combined_dataset = combined_dataset.map(lambda x : get_tokenized_length(x, tokenizer), batched=True)
    
    
    df = combined_dataset.to_pandas()
    cleaned_df = df[df['length'] < config.max_seq_length].reset_index(drop=True)
    dataset = Dataset.from_pandas(cleaned_df).shuffle(config.seed)
    
    return dataset