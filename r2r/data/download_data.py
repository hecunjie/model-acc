import os
from datasets import load_dataset
import uuid

def prepare_dataset():
    # 你的目标路径
    output_path = "/mnt/ali-sh-1/dataset/zeus/hecunjie/models/data/BS-17K"
    
    print(f"Loading Bespoke-Stratos-17k...")
    # 这里我们加载 HuggingFaceH4/Bespoke-Stratos-17k
    # cache_dir 可以指向你的缓存目录，但 load_dataset 主要是为了获取对象
    ds = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train")

    print(f"Original columns: {ds.column_names}")

    def transform(example, idx):
        # 1. 提取 question
        question = ""
        if 'conversations' in example:
            for msg in example['conversations']:
                # 兼容不同的字段名
                role = msg.get('role') or msg.get('from')
                content = msg.get('content') or msg.get('value')
                
                if role == 'user':
                    question = content
                    break
        
        # 如果没找到 user 提问，回退处理（根据实际数据情况）
        if not question:
            question = "Error: No question found"

        # 2. 确保有 ID
        # 如果原始数据里没有唯一 id，我们可以生成一个
        example_id = example.get('system_id') or example.get('id') or str(uuid.uuid4())

        return {
            "id": str(example_id),
            "question": question
        }

    print("Mapping dataset to extract 'id' and 'question'...")
    # remove_columns 确保只保留我们需要的列，避免混淆
    processed_ds = ds.map(
        transform, 
        with_indices=True,
        remove_columns=ds.column_names
    )

    print(f"Processed columns: {processed_ds.column_names}")
    print(f"Example item: {processed_ds[0]}")

    print(f"Saving to disk at: {output_path}")
    # 这才是关键步骤：生成 load_from_disk 能读取的格式
    processed_ds.save_to_disk(output_path)
    
    print("Done! Now you can run step_0_llm_response.py with --dataset_path " + output_path)

if __name__ == "__main__":
    prepare_dataset()
