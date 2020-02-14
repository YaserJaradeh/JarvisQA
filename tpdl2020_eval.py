import evaluate


def run_eval():
    pass


if __name__ == '__main__':
    ds_path = './datasets/orkg/ORKG-QA-DS.csv'
    p, r, f1, _ = evaluate.evaluate_random_baseline(ds_path, 1)
    final_result = f'Random:\nPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    print("Done with Random!!!")
    final_result = f'{final_result}{"="*40}\n'
    p, r, f1, _ = evaluate.evaluate_lucene_baseline(ds_path, 1)
    final_result = f'{final_result}Lucene:\nk=1\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    print("Done with Lucene@1")
    p, r, f1, _ = evaluate.evaluate_lucene_baseline(ds_path, 3)
    final_result = f'{final_result}Lucene:\nk=3\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    print("Done with Lucene@3")
    p, r, f1, _ = evaluate.evaluate_lucene_baseline(ds_path, 5)
    print("Done with Lucene@5")
    final_result = f'{final_result}Lucene:\nk=5\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    final_result = f'{final_result}{"=" * 40}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 1)
    print("Done with Jarvis overall@1")
    final_result = f'{final_result}Jarvis:\nk=1\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 3)
    print("Done with Jarvis overall@3")
    final_result = f'{final_result}Jarvis:\nk=3\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 5)
    print("Done with Jarvis overall@5")
    final_result = f'{final_result}Jarvis:\nk=5\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    final_result = f'{final_result}{"=" * 40}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 1, qtype='normal')
    print("Done with Jarvis normal@1")
    final_result = f'{final_result}Jarvis on Normal:\nk=1\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 3, qtype='normal')
    print("Done with Jarvis normal@3")
    final_result = f'{final_result}Jarvis on Normal:\nk=3\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 5, qtype='normal')
    print("Done with Jarvis normal@5")
    final_result = f'{final_result}Jarvis on Normal:\nk=5\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    final_result = f'{final_result}{"*" * 30}\n'
    p, r, f1 = evaluate.evaluate_jarvis(ds_path, 1, qtype='aggregation')
    print("Done with Jarvis aggregation@1")
    final_result = f'{final_result}Jarvis on aggregation:\nk=1\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 3, qtype='aggregation')
    print("Done with Jarvis aggregation@3")
    final_result = f'{final_result}Jarvis on aggregation:\nk=3\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 5, qtype='aggregation')
    print("Done with Jarvis aggregation@5")
    final_result = f'{final_result}Jarvis on aggregation:\nk=5\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    final_result = f'{final_result}{"*" * 30}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 1, qtype='related')
    print("Done with Jarvis related@1")
    final_result = f'{final_result}Jarvis on related:\nk=1\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 3, qtype='related')
    print("Done with Jarvis related@3")
    final_result = f'{final_result}Jarvis on related:\nk=3\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 5, qtype='related')
    print("Done with Jarvis related@5")
    final_result = f'{final_result}Jarvis on related:\nk=5\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    final_result = f'{final_result}{"*" * 30}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 1, qtype='similar')
    print("Done with Jarvis similar@1")
    final_result = f'{final_result}Jarvis on similar:\nk=1\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 3, qtype='similar')
    print("Done with Jarvis similar@3")
    final_result = f'{final_result}Jarvis on similar:\nk=3\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    p, r, f1, _ = evaluate.evaluate_jarvis(ds_path, 5, qtype='similar')
    print("Done with Jarvis similar@5")
    final_result = f'{final_result}Jarvis on similar:\nk=5\tPrecision: {p:.2f},\tRecall: {r:.2f},\tF1-Score: {f1:.2f}\n'
    with open('benchmark-results.txt') as out_file:
        out_file.write(final_result)
