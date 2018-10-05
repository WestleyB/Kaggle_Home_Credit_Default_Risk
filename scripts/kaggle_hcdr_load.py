import os
import sys
import numpy as np
import pandas as pd


path_sources = '../data/'


print("------------------------------------------------------------------------------------------------------")
print("\n=======> READ FILES in {} \n".format(str(path_sources)))
for file_src in os.listdir(path_sources):
	
	print("------------------------------------------------------------------------------------------------------")
	print("	[" + file_src + "]")
	
	if '~' not in file_src and '.csv' in file_src.lower():
		path = os.path.join(path_sources, file_src)
		dt = pd.read_csv(path)
		print("	[Columns] : " + str(len(dt.columns)))
		print("	[Shape] : " + str(dt.shape))
