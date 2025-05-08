import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm  # 新增进度条库

# 初始化文本分类模型

tokenizer1 = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
tokenizer2 = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# 模型输出到标签的映射
model_label_map = {
    'surprise': 'Surprise',
    'anger': 'Anger',
    'disgust': 'Disgust',
    'sadness': 'Sad',
    'joy': 'Happy',
    'fear': 'Fear',
    'neutral': 'Neutral'
}

def check_dataset(root_dir):
    mismatched = []
    
    # 遍历所有数据集分割目录
    for split in ['train']:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue
            
        # 遍历情感类别目录
        emotions = os.listdir(split_path)
        for emotion in tqdm(emotions, desc=f'Processing {split}', leave=False):
            class_dir = os.path.join(split_path, emotion)
            if not os.path.isdir(class_dir):
                continue
                
            # 检查每个pt文件
            files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
            for file in tqdm(files, desc=f'{emotion[:8]:<8}', leave=False):
                file_path = os.path.join(class_dir, file)
                try:
                    data = torch.load(file_path,weights_only=True)
                    # 获取模型预测结果
                    
                    llm_tensor = data['LLM'] 
                    decoded_text = tokenizer1.decode(
                        llm_tensor[0],
                    skip_special_tokens=True
                    )
                    inputs = tokenizer2(decoded_text, return_tensors="pt")
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        pred_idx = logits.argmax().item()
                        predicted_label = model.config.id2label[pred_idx]
                    
                    # 转换模型标签到我们的标签体系
                    mapped_label = model_label_map.get(predicted_label.lower(), None)
                    if mapped_label != emotion:
                        mismatched.append({
                            'file': file_path,
                        })
                except Exception as e:
                    print(f"\n🔥 Error processing {os.path.basename(file_path)}: {str(e)}")
    
    # 保存检查结果
    with open('/root/autodl-fs/dataset/mismatched_results.json', 'w') as f:
        json.dump(mismatched, f, indent=2)
        
    print(f"Found {len(mismatched)} mismatched samples. Results saved to /root/autodl-fs/dataset/mismatched_results.json")

if __name__ == "__main__":
    check_dataset("/root/autodl-fs/dataset/caer/fusion")