# CS505: NLP - Spring 2026
import sys

class Example:
    def __init__(self, label, text):
        # Labels:
        # -1 - Unknown (only used for the test data)
        # 0 - world
        # 1 - sports
        # 2 - business
        # 3 - tech
        LABEL_MAP = {"world": 0, "sports": 1, "business": 2, "tech": 3}
        self.label = LABEL_MAP.get(label, -1)
        self.text = text
        
    def __repr__(self):
        return f"Label: {self.label}, Text: {self.text[:30]}..."

def load_data(filename):
    # This function assumes that the file is a tab-separated values (TSV) file.
    data = []
    from collections import defaultdict
    label_counts = defaultdict(int)
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        label, text = parts
                        data.append(Example(label, text))
                        label_counts[label] += 1
                else:
                    text = line
                    data.append(Example(-1, text))
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)
            
    print(f"Loaded {len(data)} examples from {filename}")
    return data