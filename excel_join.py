import pandas as pd
import pathlib

with pd.ExcelWriter('join_fold.xlsx') as writer:
    for filename in pathlib.Path('./data/sample_commit').glob('*.csv'):
        df = pd.read_csv(filename)
        df.to_excel(writer, sheet_name=filename.stem, index=False)
