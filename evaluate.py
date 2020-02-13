import pandas as pd
from baselines import BaseSelector, RandomSelector, LuceneSelector
import statistics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


def compute_metrics(true, pred, k):
    if len(true) != len(pred):
        raise ValueError("true and pred should be of the same length")
    if k < 1:
        raise ValueError(f"k should be at least 1, got {k}")
    for index in range(len(true)):
        predicted = pred[index][:k]
        truth = true[index]




def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result


def recall(actual, predicted, k):
    act_set = set(actual)
    if len(act_set) == 0:
        return 0.0
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result


def f_score(precision, recall):
    if precision+recall == 0.0:
        return 0.0
    return 2.0*((precision*recall)/(precision+recall))


def precision_recall_fscore(true, pred, k):
    ps = []
    rs = []
    f1s = []
    for index, predicated in enumerate(pred):
        p = precision(true[index], predicated, k)
        ps.append(p)
        r = recall(true[index], predicated, k)
        rs.append(p)
        f1 = f_score(p, r)
        f1s.append(f1)
    return statistics.mean(ps), statistics.mean(rs), statistics.mean(f1s)


def evaluate_random_baseline(dataset_path):
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred1 = []
    y_pred2 = []
    y_pred3 = []
    baseline = RandomSelector(seed=12)
    for index, row in df.iterrows():
        question = row['Question']
        real_answer = row['Answer']
        y_true.append(real_answer if pd.notna(real_answer) else '')
        baseline.update_table(f'./datasets/orkg/csv/{row["Table"]}.csv')
        answers = baseline.answer_question(question, 3)
        y_pred1.append([answers[0]])
        y_pred2.append(answers[:2])
        y_pred3.append(answers)
    multi = MultiLabelBinarizer()
    y_true = multi.fit(y_true).transform(y_true)
    y_pred1 = multi.transform(y_pred1)
    y_pred2 = multi.transform(y_pred2)
    y_pred3 = multi.transform(y_pred3)
    p1, r1, f1, _ = precision_recall_fscore_support(y_true, y_pred1, average='macro', zero_division=0)
    p2, r2, f2, _ = precision_recall_fscore_support(y_true, y_pred2, average='macro', zero_division=0)
    p3, r3, f3, _ = precision_recall_fscore_support(y_true, y_pred3, average='macro', zero_division=0)
    return p1, r1, f1, p1+p2, r1+r2, f1+f2, p1+p2+p3, r1+r2+r3, f1+f2+f3


def evaluate_lucene_baseline(dataset_path):
    df = pd.read_csv(dataset_path)
    y_true = []
    y_pred = []

    for index, row in df.iterrows():
        question = row['Question']
        y_true.append(row['Answer'])
        baseline = LuceneSelector(f'./datasets/orkg/csv/{row["Table"]}.csv')
        print(question)
        baseline.clean_index()
        baseline.index_table()
        answers = baseline.answer_question(question, 1)
        print(answers)
        print("="*40)
        y_pred.append(answers[0] if len(answers) > 0 else '')
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return p, r, f1


if __name__ == '__main__':
    print(evaluate_random_baseline('./datasets/orkg/ORKG-QA-DS.csv'))
    # print(evaluate_lucene_baseline('./datasets/orkg/ORKG-QA-DS.csv'))
    # df = pd.read_csv('./datasets/orkg/ORKG-QA-DS.csv')
    # y_true = []
    # y_pred = []
    #
    # for index, row in df.iterrows():
    #     question = row['Question']
    #     y_true.append(row['Answer'])
    #     baseline = LuceneSelector(f'./datasets/orkg/csv/{row["Table"]}.csv')
    #     print(question)
    #     baseline.clean_index()
    #     baseline.index_table()
    #     answers = baseline.answer_question(question, 3)
    #     print(answers)
    #     print("=" * 40)
    #     y_pred.append(answers)
    # p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
