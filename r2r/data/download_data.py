from datasets import load_dataset
import os

# 指定你想要的下载位置（例如当前目录下的 data 文件夹）
download_path = os.path.join(os.getcwd(), "my_datasets")  # 或绝对路径如 "/mnt/data/datasets"

# 确保目录存在（可选，但推荐）
os.makedirs(download_path, exist_ok=True)

# 加载数据集，并指定缓存目录
ds = load_dataset(
    "HuggingFaceH4/Bespoke-Stratos-17k",
    cache_dir=download_path,          # 核心参数：下载和缓存都放这里
    # 可选：如果你只想下载，不立即处理成 Arrow 格式，可以加 download_mode
    # download_mode="force_redownload"  # 强制重新下载（用于测试）
)

print(ds)  # 查看数据集信息
print(f"数据集缓存路径示例：{download_path}")