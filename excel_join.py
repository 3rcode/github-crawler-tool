import pandas as pd
import pathlib
import os
from settings import ROOT_DIR
FOLD = 30
for i in range(7, 8):
    input_path = os.path.join(ROOT_DIR, "data", "sample_commit", f"group_{i + 1}")
    output_path = os.path.join(ROOT_DIR, "data", "sample_commit", f"group_{i + 1}.xlsx")
    try:
        with pd.ExcelWriter(output_path) as writer:
            for filename in pathlib.Path(input_path).glob('*.csv'): 
                df = pd.read_csv(filename)
                print(len(df))
                df.to_excel(writer, sheet_name=filename.stem, index=False)
    except Exception as e:  
        print(e)
