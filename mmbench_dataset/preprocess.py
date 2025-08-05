import pandas as pd
import os
from PIL import Image
import io
import json
import argparse

def extract_mmbench_data(
    data_path, 
    output_dir, 
    include_hint=True,
    include_options=True, 
    cn=True
):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    df = pd.read_parquet(data_path)

    if cn:
        prompt = "从选项{}中选出最佳答案，并用字母（如A）标示。"
    else:
        prompt = "Choose the best answer from the options {} and indicate it with a letter (e.g., A)."
    
    with open(os.path.join(output_dir, "mmbench.jsonl"), "w", encoding='utf-8') as f:
        for idx, row in df.iterrows():
            try:
                data_index = row['index']
                answer = row['answer']

                question_parts = []

                if include_hint and pd.notna(row['hint']) and str(row['hint']).strip() != "nan":
                    question_parts.append(row['hint'])

                question_parts.append(row['question'])
                
                valid_option_tags = []
                if include_options:
                    for tag in ['A', 'B', 'C', 'D']:
                        if pd.notna(row[tag]) and str(row[tag]).strip() != "nan" and str(row[tag]).strip() != "":
                            valid_option_tags.append(tag)
                    
                    if valid_option_tags:
                        options_placeholder = ", ".join(valid_option_tags)
                        if cn:
                            prompt_ = prompt.format(options_placeholder)
                        else:
                            prompt_ = prompt.format(options_placeholder)
                        
                        options = [f"{tag}. {row[tag]}" for tag in valid_option_tags]
                        question_parts.extend(options)
                
                question_parts.append(prompt_)
                
                full_question = "\n".join(question_parts)
                
                image_data = row['image']
                img_path = None
                try:
                    image = Image.open(io.BytesIO(image_data['bytes']))
                    img_filename = f"image_{data_index}.png"
                    img_path = os.path.join(images_dir, img_filename)
                    image.save(img_path)
                except Exception as e:
                    print(f"Error in（index {data_index}）: {str(e)}")

                json.dump(
                    {
                    'index': data_index,
                    'text': full_question,
                    'label': answer,
                    'image': img_path,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                
            except Exception as e:
                print(f"{e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMBench preprocess.")
    parser.add_argument("--data_path", type=str, help="The path of downloaded MMBench.")
    args = parser.parse_known_args()[0]

    parquet_file_path = os.path.join(args.data_path, "cn/dev-00000-of-00001.parquet")
    output_directory = "./mmbench_dataset/cn"
    cn = True

    extract_mmbench_data(
        data_path=parquet_file_path,
        output_dir=output_directory,
        include_hint=True,
        include_options=True, 
        cn=cn
    )


    parquet_file_path = os.path.join(args.data_path, "en/dev-00000-of-00001.parquet")
    output_directory = "./mmbench_dataset/en"
    cn = False

    extract_mmbench_data(
        data_path=parquet_file_path,
        output_dir=output_directory,
        include_hint=True,
        include_options=True, 
        cn=cn
    )
    