from transformers import pipeline


class QABrain:

    def __init__(self, model=None, tokenizer=None):
        if model is None:
            self.nlp = pipeline('question-answering')
        if model is not None and tokenizer is None:
            self.nlp = pipeline('question-answering', model=model)
        if model is not None and tokenizer is not None:
            self.nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    @staticmethod
    def remove_duplicate(dicts, key):
        added = []
        results = []
        for dic in dicts:
            if dic[key] not in added:
                added.append(dic[key])
                results.append(dic)
        return results

    def answer(self, question, context, top_k=3, threshold=0.001):
        answers = self.nlp({
            'question': question,
            'context': context
        }, topk=top_k, max_seq_len=512)
        if not isinstance(answers, list):
            answers = [answers]
        clean_answers = [answer for answer in answers if answer['score'] >= threshold]
        clean_answers = self.remove_duplicate(clean_answers, 'start')
        clean_answers = self.remove_duplicate(clean_answers, 'end')
        clean_answers = self.remove_duplicate(clean_answers, 'answer')
        for answer in clean_answers:
            answer['answer'] = answer['answer'].strip('().{},\'"')
        return [a['answer'] for a in sorted(clean_answers, key=lambda x: x['score'], reverse=True)]
