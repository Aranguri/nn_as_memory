import pandas as pd
import numpy as np
df = pd.read_csv('/home/aranguri/Downloads/documents_utf8_filtered_20pageviews.csv',sep=',', header = None, nrows=10000)
data = df.loc[df[1].str.contains('mathematics') & (df[1].str.contains('theory') | df[1].str.contains('set')) & ~df[1].str.contains('mathematician') & ~df[1].str.contains('born') & ~df[1].str.contains('university')]
np.save('data', np.array(data.to_records()))

#** Assume we have the data ** 
