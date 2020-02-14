import pandas as pd
from baselines import BaseSelector, RandomSelector, LuceneSelector
from sklearn.metrics import precision_recall_fscore_support
from time import time
from memory_profiler import profile


@profile
def evaluate_random_baseline(dataset_path, top_k=3, qtype=None):
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = []
    times = []
    baseline = RandomSelector(seed=12)
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        if qtype is not None and qtype in df.keys():
            if row['Type'] != qtype:
                continue
        y_true.append(real_answer if pd.notna(real_answer) else '')
        baseline.update_table(f'./datasets/orkg/csv/{row["Table"]}.csv')
        start = time()
        answers = baseline.answer_question(question, 4)[:top_k]
        end = time()
        times.append(end-start)
        if real_answer in answers:
            y_pred.append(real_answer)
        else:
            y_pred.append(answers[0])
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return p, r, f1, pd.np.np.mean(times) * 1000


@profile
def evaluate_lucene_baseline(dataset_path, top_k=3, qtype=None):
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = []
    times = []
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        if qtype is not None and qtype in df.keys():
            if row['Type'] != qtype:
                continue
        baseline = LuceneSelector(f'./datasets/orkg/csv/{row["Table"]}.csv')
        baseline.clean_index()
        baseline.index_table()
        y_true.append(real_answer if pd.notna(real_answer) else '')
        start = time()
        answers = baseline.answer_question(question, top_k)
        end = time()
        times.append(end - start)
        if real_answer in answers:
            y_pred.append(real_answer)
        else:
            y_pred.append(answers[0] if len(answers) > 0 else '')
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return p, r, f1, pd.np.np.mean(times) * 1000


if __name__ == '__main__':
    print(f'{"="*10} Random {"="*10}')
    print(evaluate_random_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=1))
    print(evaluate_random_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=2))
    print(evaluate_random_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=3))
    print(f'{"=" * 10} Lucene {"=" * 10}')
    print(evaluate_lucene_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=1))
    print(evaluate_lucene_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=2))
    print(evaluate_lucene_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=3))
