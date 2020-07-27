# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import os

# Read recipe inputs
articles = dataiku.Folder('C9H1g8IY')
articles_info = articles.get_info()
articles_path = articles.get_path()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = pd.DataFrame(columns=['text','filename'])
for filename in articles.list_paths_in_partition():
    file_label = filename.split('/')[1]
    filepath = os.path.join(articles_path, file_label)
    with open(filepath,'r') as f:
        content = f.read()
    df = df.append({'text':content,'filename':file_label}, ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
rax_text = dataiku.Dataset("raw_text_test")
rax_text.write_with_schema(df)