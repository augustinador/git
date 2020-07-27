import dataiku
from dataiku.customrecipe import *
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import matplotlib 
import wordcloud
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read recipe inputs
sentiment_pos = dataiku.Dataset("label_predictions")
df = sentiment_pos.get_dataframe()

folder = dataiku.Folder("AY01UOIe")
wordcloud_path = folder.get_path()

## wordcloud creation
wc = wordcloud.WordCloud(
        background_color='white',
        max_font_size=40, 
        scale=3
).generate(str(df["text"]))
fig = plt.figure()
plt.axis('off')
plt.imshow(wc)
plt.show()

## save the wordcloud to the output folder
path_fig = wordcloud_path + '/' + "wordcloud"
plt.savefig(path_fig)

## export the wordcloud as a static insight
import dataiku.insights
id = "wordcloud_pos"
dataiku.insights.save_figure(id)
