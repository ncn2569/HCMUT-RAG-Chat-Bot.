import pandas as pd
import json

def chunking(excel_path,output_path):
    df=pd.read_excel(excel_path)
    df=df[df['Question']!='Question']
    with open(output_path,'w',encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "text": f"Q: {row['Question']}\nA: {row['Answer']}"
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

chunking('../../data/raw/data cho box thông tin trường-completed.xlsx','../../data/processed/data-cleaned.jsonl')

print("JSONL created.")
