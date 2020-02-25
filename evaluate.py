import pandas as pd
from baselines import RandomSelector, LuceneSelector
from sklearn.metrics import precision_recall_fscore_support
from jarvis import JarvisQA
from time import time
from memory_profiler import profile


#@profile
def evaluate_random_baseline(dataset_path, top_k=3, qtype=None):
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = []
    times = []
    baseline = RandomSelector(seed=4)
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        if qtype is not None:
            if row['Type'] != qtype:
                continue
        y_true.append(real_answer if pd.notna(real_answer) else '')
        baseline.update_table(f'./datasets/orkg/csv/{row["Table"]}.csv')
        start = time()
        answers = baseline.answer_question(question, 10)[:top_k]
        end = time()
        times.append(end-start)
        if real_answer in answers:
            y_pred.append(real_answer)
        else:
            y_pred.append(answers[0])
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return p, r, f1, pd.np.np.mean(times) * 1000


def evaluate_random_baseline_efficient(dataset_path, top_k=3, qtype=None) -> list:
    if top_k > 10:
        raise ValueError(f"topK can't be more than 10, got {top_k}")
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = [[] for _ in range(top_k)]
    times = []
    results = []
    baseline = RandomSelector(seed=4)
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        if qtype is not None:
            if row['Type'] != qtype:
                continue
        y_true.append(real_answer if pd.notna(real_answer) else '')
        baseline.update_table(f'./datasets/orkg/csv/{row["Table"]}.csv')
        start = time()
        answers = baseline.answer_question(question, 10)[:top_k]
        end = time()
        times.append(end-start)
        for i in range(top_k):
            if real_answer in answers[:i+1]:
                y_pred[i].append(real_answer)
            else:
                y_pred[i].append(answers[0])
    for k in range(top_k):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred[k], average='macro', zero_division=0)
        results.append((k+1, p, r, f1))
    return results


def evaluate_lucene_baseline_efficient(dataset_path, top_k=3, qtype=None) -> list:
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = [[] for _ in range(top_k)]
    times = []
    results = []
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        if qtype is not None:
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
        for i in range(top_k):
            if real_answer in answers[:i+1]:
                y_pred[i].append(real_answer)
            else:
                y_pred[i].append(answers[0])
    for k in range(top_k):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred[k], average='macro', zero_division=0)
        results.append((k+1, p, r, f1))
    return results


#@profile
def evaluate_lucene_baseline(dataset_path, top_k=3, qtype=None):
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = []
    times = []
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        if qtype is not None:
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


def evaluate_jarvis_efficient(dataset_path, top_k=3, qtype=None, model_name='deepset/bert-large-uncased-whole-word-masking-squad2') -> list:
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = [[] for _ in range(top_k)]
    times = []
    results = []
    qa = JarvisQA(model=model_name, tokenizer=model_name)
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        if isinstance(real_answer, str):
            real_answer = real_answer.strip('().{},\'"')
        if qtype is not None:
            if row['Type'] != qtype:
                continue
        y_true.append(real_answer if pd.notna(real_answer) else '')
        start = time()
        answers = qa.answer_question(f'./datasets/orkg/csv/{row["Table"]}.csv', question, top_k)
        end = time()
        times.append(end - start)
        for i in range(top_k):
            if real_answer in answers[:i+1]:
                y_pred[i].append(real_answer)
            else:
                y_pred[i].append(answers[0])
    for k in range(top_k):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred[k], average='macro', zero_division=0)
        results.append((k+1, p, r, f1))
    return results


#@profile
def evaluate_jarvis(dataset_path, top_k=3, qtype=None, model_name='deepset/bert-large-uncased-whole-word-masking-squad2'):
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = []
    times = []
    qa = JarvisQA(model=model_name, tokenizer=model_name)
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        if isinstance(real_answer, str):
            real_answer = real_answer.strip('().{},\'"')
        if qtype is not None:
            if row['Type'] != qtype:
                continue
        y_true.append(real_answer if pd.notna(real_answer) else '')
        start = time()
        answers = qa.answer_question(f'./datasets/orkg/csv/{row["Table"]}.csv', question, top_k)
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
    #print(evaluate_random_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=3))
    #print(evaluate_random_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=5))
    #print(evaluate_random_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=10))
    #print(evaluate_random_baseline_efficient('./datasets/orkg/ORKG-QA-DS.csv', top_k=10))
    #print(f'{"=" * 10} Lucene {"=" * 10}')
    #print(evaluate_lucene_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=1))
    #print(evaluate_lucene_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=2))
    #print(evaluate_lucene_baseline('./datasets/orkg/ORKG-QA-DS.csv', top_k=3))
    #print(f'{"=" * 10} Jarvis base {"=" * 10}')
    #print(evaluate_jarvis('./datasets/orkg/ORKG-QA-DS.csv', top_k=1))
    #print(evaluate_jarvis('./datasets/orkg/ORKG-QA-DS.csv', top_k=3))
    #print(evaluate_jarvis('./datasets/orkg/ORKG-QA-DS.csv', top_k=5))
    #print(evaluate_jarvis_efficient('./datasets/orkg/ORKG-QA-DS.csv', top_k=5))
