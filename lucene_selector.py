import pandas as pd
import os
import pysolr

stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']


# you have to run this
# docker run -p 8983:8983 -t solr
# docker run -d -p 8983:8983 --name my_solr solr solr-precreate gettingstarted
class LuceneSelector:

    def __init__(self, path: str, solr_instance: str = 'http://localhost:8983/solr/gettingstarted/'):
        self.df = self.__read_csv(path)
        self.solr = pysolr.Solr(solr_instance, always_commit=True)
        self.count = 0

    @staticmethod
    def __read_csv(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise ValueError("Path doesn't exists")
        return pd.read_csv(path)

    @staticmethod
    def __clean_values(value):
        if value == 'T' or value == 'Y':
            return "True"
        if value == 'F' or value == 'N':
            return "False"
        return value

    def index_table(self):
        docs = []
        for _, row in self.df.iterrows():
            for predicate in row.keys()[1:]:
                doc_id = f'doc_{self.count}'
                value = self.__clean_values(row[predicate])
                content = f'{row[row.keys()[0]]}\'s {predicate} is "{value}"'
                docs.append({"id": doc_id, "segment": content, "value": value})
                self.count = self.count + 1
        self.solr.add(docs)

    def query_index(self, query, top_k=3):
        clean_query = ' '.join([w for w in query.split() if w.lower() not in stopwords])
        return [result['value'][0] for result in list(self.solr.search(f'segment:{clean_query}'))[:top_k]]

    def clean_index(self):
        self.solr.delete(q='*:*')


if __name__ == '__main__':
    x = LuceneSelector('./data/test.csv')
    #x.index_table()
    print(x.query_index("what is the data type of nanopublications?"))
    #x.clean_index()
