from orkg import ORKG
import re

MAP = {
    "deepset/bert-base-cased-squad2": "BERT-Base",
    "deepset/bert-large-uncased-whole-word-masking-squad2": "BERT-Large",
    "distilbert-base-uncased-distilled-squad": "DistilBERT-Base",
    "ktrapeznikov/albert-xlarge-v2-squad-v2": "ALBERT-XL",
    "replydotai/albert-xxlarge-v1-finetuned-squad2": "ALBERT-XXL"
}
orkg = ORKG(host="https://sandbox.orkg.org/")
orkg.templates.materialize_template(template_id="R107811")


def parse_text_file(file_path):
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        model = ''
        for idx, line in enumerate(lines):
            if line.startswith('On model:'):
                model = line.strip().split(': ')[1]
                if model in MAP:
                    model = MAP[model]
                    data[model] = {}
                else:
                    model = ''
                    continue
            elif line.startswith('Jarvis:') and len(model) > 0:
                line = lines[idx+1]
                match = re.search(r'k=(\d+)\s+Precision: ([\d.]+),\s+Recall: ([\d.]+),\s+F1-Score: ([\d.]+)', line)
                k, precision, recall, f1_score = map(float, match.groups())
                if model not in data:
                    data[model] = {}
                k = int(k)
                data[model].update({
                    f'Precision@{k}': precision,
                    f'Recall@{k}': recall,
                    f'F1-Score@{k}': f1_score
                })
    return data


if __name__ == '__main__':
    dataset = "ORKG-QA"
    # TODO: Use resource_id for the dataset resource, and same for metrics
    for model, evaluation in parse_text_file("./eval-results/benchmark-results-exp2.txt").items():
        orkg.templates.benchmark(
            has_dataset=dataset,
            has_evaluation=""
        )

