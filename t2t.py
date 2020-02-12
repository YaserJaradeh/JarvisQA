import csv
import os
from typing import List
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from collections import Counter
from itertools import chain


class T2T:

    @staticmethod
    def read_csv(path: str) -> (List, List):
        if not os.path.exists(path):
            raise ValueError("Path doesn't exists")
        with open(path, 'r') as in_file:
            reader = csv.reader(in_file)
            result = [row for row in reader]
            return result[0], result[1:]

    @staticmethod
    def clean_predicate(predicate: str) -> str:
        if predicate[:3].lower() == 'has':
            return predicate[3:]
        return predicate

    @staticmethod
    def append_value(value: str) -> str:
        if len(value.split(',')) > 1:
            return f'are "{value}"'
        else:
            return f'is "{value}"'

    def append_aggregation_info(self, df: pd.DataFrame) -> str:
        info = []
        for column in df:
            column_df = df[column]
            if is_numeric_dtype(column_df):
                max_value = column_df.max()
                min_value = column_df.min()
                avg_value = column_df.mean()
                info.append(f'The maximum value of {column} is {max_value},'
                            f' the minimum value of {column} is {min_value},'
                            f' and the average value of {column} is {avg_value:.2f}')
            if is_string_dtype(column_df):
                counts = Counter([value.strip() for value in chain(*[value.split(',') for value in column_df.values])])
                max_occurrence = max(counts.values())
                if max_occurrence == 1:
                    # ignore one value occurrences
                    continue
                min_occurrence = min(counts.values())
                most_common = [k for k, v in counts.items() if v == max_occurrence]
                least_common = [k for k, v in counts.items() if v == min_occurrence]
                info.append(f'The most common {column} {self.append_value(",".join(most_common))},'
                            f' and the least common {self.append_value(",".join(least_common))}')
        return '\n'.join(info)

    def row_2_text(self, row: List, header: List, start_index: int = 0) -> str:
        text = f'{row[start_index]}'
        parts = [(header[index], value) for index, value in enumerate(row) if len(value.strip()) != 0][start_index+1:]
        text = f'{text}\'s {self.clean_predicate(parts[0][0])} {self.append_value(parts[0][1])}'
        for part in parts[1:-1]:
            text = f'{text}, its {self.clean_predicate(part[0])} {self.append_value(part[1])}'
        text = f'{text}, and its {self.clean_predicate(parts[-1][0])} {self.append_value(parts[-1][1])}.'
        return text

    def table_2_text(self, csv_path: str) -> str:
        header, rows = self.read_csv(csv_path)
        extra_info = self.append_aggregation_info(pd.read_csv(csv_path))
        return ('\n'.join([self.row_2_text(row, header) for row in rows])) + '\n' + extra_info


if __name__ == '__main__':
    t2t = T2T()
    print(t2t.table_2_text("./data/test.csv"))
