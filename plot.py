import matplotlib.pyplot as plt
import numpy as np

precision = [0.,         0.59375,    0.5682959,  0.98432699, 0.20942408, 0.55965909, 0.5035545,  0.        ]
recall = [0.,         0.51428571, 0.92238422, 0.96533252, 0.37037037, 0.36146789, 0.15816896, 0.        ]
classes = ["bicycle", "bus", "car", "motor", "pedestrian", "tricycle", "truck", "van"]
X = np.arange(len(classes))
plt.bar(X, precision, width=0.25)
plt.bar(X+0.25, recall, width=0.25)
#plt.
plt.xticks(np.arange(len(classes)), labels=classes, rotation=45)
plt.legend(labels=['Precision', 'Recall'], loc='upper right')
plt.ylabel('Score')
plt.grid(True, linestyle='--')
plt.title('Precision and Recall for each class')
plt.savefig('./assets/precision_recall.pdf')
plt.show()
# import pandas as pd

# df = pd.read_csv('logs/scores_table.txt')
# print(df)
# df.columns = ['acc', 'ap', 'f1', 'ovo', 'ovr']
# print(df.iloc[df['ap'].idxmin()])
# print(df.iloc[df['ap'].idxmin()].mean())

# # print(df)
# # print('Debug')