import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv("/home/jovyan/work/causal-events/data/subtask1/dev_subtask1.csv")
    print(data.head())
    print(data.info())

