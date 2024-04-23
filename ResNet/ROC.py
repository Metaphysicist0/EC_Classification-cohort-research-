import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 读取两个CSV文件
df1 = pd.read_csv(r"G:\EC\classification\ResNet\my_model.csv")
# df2 = pd.read_csv(r"C:\Users\Administrator\Desktop\ROC\resnet18.csv")
# df3 = pd.read_csv(r"C:\Users\Administrator\Desktop\ROC\resnet181.csv")
# df4 = pd.read_csv(r"C:\Users\Administrator\Desktop\ROC\VIT.csv")
# df5 = pd.read_csv(r"C:\Users\Administrator\Desktop\ROC\VIT1.csv")
# 提取predictions和labels列的值
predictions1 = df1['predictions']
labels1 = df1['labels']

'''
predictions2 = df2['predictions']
labels2 = df2['labels']

predictions3 = df3['predictions']
labels3 = df3['labels']

predictions4 = df4['predictions']
labels4 = df4['labels']


predictions5 = df5['predictions']
labels5 = df5['labels']
'''
# 计算ROC曲线数据
fpr1, tpr1, _ = roc_curve(labels1, predictions1)
roc_auc1 = auc(fpr1, tpr1)

# fpr2, tpr2, _ = roc_curve(labels2, predictions2)
# roc_auc2 = auc(fpr2, tpr2)
#
# fpr3, tpr3, _ = roc_curve(labels3, predictions3)
# roc_auc3 = auc(fpr3, tpr3)
#
# fpr4, tpr4, _ = roc_curve(labels4, predictions4)
# roc_auc4 = auc(fpr4, tpr4)
#
# fpr5, tpr5, _ = roc_curve(labels5, predictions5)
# roc_auc5 = auc(fpr5, tpr5)
# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, color='black', lw=2, label='ResNet(AUC = %0.4f)' % roc_auc1)
# plt.plot(fpr2, tpr2, color='blue', lw=2, label='CBAM+ResNet(AUC = %0.4f)' % roc_auc2)
# plt.plot(fpr3, tpr3, color='orange', lw=2, label='EfficientNet-B0(AUC = %0.4f)' % roc_auc3)
# plt.plot(fpr4, tpr4, color='green', lw=2, label='ViT(AUC = %0.4f)' % roc_auc4)
# plt.plot(fpr5, tpr5, color='red', lw=2, label='TransMed(AUC = %0.4f)' % roc_auc5)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('ROC curves', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.show()
