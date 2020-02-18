import evaluate


def exp1():
    ds_path = './datasets/orkg/ORKG-QA-DS.csv'
    p, r, f1, _ = evaluate.evaluate_random_baseline(ds_path, 1)
    final_result = f'Random:\nPrecision: {p:.4f},\tRecall: {r:.4f},\tF1-Score: {f1:.4f}\n'
    print("Done with Random!!!")
    final_result = f'{final_result}{"=" * 40}\n'
    for k in [1, 3, 5, 10]:
        p, r, f1, _ = evaluate.evaluate_lucene_baseline(ds_path, k)
        final_result = f'{final_result}Lucene:\nk={k}\tPrecision: {p:.4f},\tRecall: {r:.4f},\tF1-Score: {f1:.4f}\n'
        print(f"Done with Lucene@{k}")
    final_result = f'{final_result}{"=" * 40}\n'
    for kind in ['normal', 'aggregation', 'related', 'similar']:
        for k in [1, 3, 5, 10]:
            p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, k, qtype=kind)
            final_result = f'{final_result}Jarvis {kind}:\nk={k}\tPrecision: {p:.4f},\tRecall: {r:.4f},\tF1-Score: {f1:.4f}\n'
            print(f"Done with Jarvis {kind}@{k}")
        final_result = f'{final_result}{"*" * 30}\n'
    for k in [1, 3, 5, 10]:
        p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, k)
        final_result = f'{final_result}Jarvis:\nk={k}\tPrecision: {p:.4f},\tRecall: {r:.4f},\tF1-Score: {f1:.4f}\n'
        print(f"Done with Jarvis overall@{k}")
        final_result = f'{final_result}{"=" * 40}\n'
    with open('benchmark-results-exp1.txt', 'w') as out_file:
        out_file.write(final_result)


if __name__ == '__main__':
    ds_path = './datasets/orkg/ORKG-QA-DS.csv'
    final_result = ''
    for model in ['bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking-finetuned-squad', 'deepset/bert-base-cased-squad2', 'deepset/bert-large-uncased-whole-word-masking-squad2', 'distilbert-base-uncased-distilled-squad', 'ktrapeznikov/albert-xlarge-v2-squad-v2', 'replydotai/albert-xxlarge-v1-finetuned-squad2']:
        final_result = f'{final_result}On model: {model}\n'
        print(f'Starting with model {model}')
        for kind in ['normal', 'aggregation', 'related', 'similar']:
            for k in [1, 3, 5, 10]:
                p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, k, qtype=kind, model_name=model)
                final_result = f'{final_result}Jarvis {kind}:\nk={k}\tPrecision: {p:.4f},\tRecall: {r:.4f},\tF1-Score: {f1:.4f}\n'
                print(f"Done with Jarvis {kind}@{k}")
            final_result = f'{final_result}{"*" * 30}\n'
        for k in [1, 3, 5, 10]:
            p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, k, model_name=model)
            final_result = f'{final_result}Jarvis:\nk={k}\tPrecision: {p:.4f},\tRecall: {r:.4f},\tF1-Score: {f1:.4f}\n'
            print(f"Done with Jarvis overall@{k}")
        final_result = f'{final_result}{"=" * 40}\n'
    with open('benchmark-results-exp2.txt', 'w') as out_file:
        out_file.write(final_result)
