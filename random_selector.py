import pandas as pd
from random import Random
import os


class RandomSelector:

    def __init__(self, path: str, seed: int = 10):
        self.df = self.__read_csv(path)
        self.randomizer = Random(seed)

    @staticmethod
    def __read_csv(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise ValueError("Path doesn't exists")
        return pd.read_csv(path)

    def choose_answer_randomly(self) -> str:
        values = self.df.values
        x = self.randomizer.randint(1, values.shape[0]-1)
        y = self.randomizer.randint(1, values.shape[1]-1)
        return values[x, y]


if __name__ == '__main__':
    x = RandomSelector('./data/test.csv')
    for _ in range(5):
        print(x.choose_answer_randomly())
