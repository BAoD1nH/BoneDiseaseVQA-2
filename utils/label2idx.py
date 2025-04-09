import json

def build_label_mappings(json_file, save_to=None):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_answers = set()
    for item in data:
        # Format: "diagnose, tình trạng condition"
        answer_text = f"{item['diagnose']}, tình trạng {item['condition']}"
        unique_answers.add(answer_text.strip())

    sorted_answers = sorted(unique_answers)  # for consistent indexing
    label2idx = {ans: i for i, ans in enumerate(sorted_answers)}
    idx2label = {str(i): ans for ans, i in label2idx.items()}  # keys as string for JSON

    if save_to:
        with open(f"{save_to}_label2idx.json", 'w', encoding='utf-8') as f:
            json.dump(label2idx, f, ensure_ascii=False, indent=2)
        with open(f"{save_to}_idx2label.json", 'w', encoding='utf-8') as f:
            json.dump(idx2label, f, ensure_ascii=False, indent=2)

    return label2idx, idx2label

# Example usage:
label2idx, idx2label = build_label_mappings('dataset/cleaned_output_bonedata.json', save_to='label_map')
print(f"Total unique labels: {len(label2idx)}")
