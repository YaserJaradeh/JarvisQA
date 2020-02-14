from t2t import T2T
from qa_brain import QABrain


class JarvisQA:

    def __init__(self, model=None, tokenizer=None):
        self.t2t = T2T()
        self.brain = QABrain(model, tokenizer)

    def answer_question(self, csv_path: str, question: str, topk: int = 3) -> list:
        context = self.t2t.table_2_text(csv_path)
        return self.brain.answer(question, context, top_k=topk)
