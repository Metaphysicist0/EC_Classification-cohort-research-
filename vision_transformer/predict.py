import os
import json
import os

import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import torch
import sys
from torchvision import transforms
import matplotlib.pyplot as plt
from MyDataSet import MyDataSet
import os
import json
from itertools import cycle
import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import label_binarize
from scipy import interp

import sys
from torchvision import transforms
import matplotlib.pyplot as plt
from MyDataSet import MyDataSet

from test_split import read_split_data
import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import torch
import sys
from torchvision import transforms
import matplotlib.pyplot as plt
from MyDataSet import MyDataSet
# from vit_model import vit_base_patch32_224 as create_model
# from vit_model import vit_large_patch32_224_in21k as create_model
from vit_model import vit_base_patch16_224 as create_model
from test_split import read_split_data
test_data_path = r"G:\EC\ECDataset\test"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    test_images_path, test_images_label = read_split_data(test_data_path)
    batch_size = 64
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 实例化训练数据集
    test_dataset = MyDataSet(images_path=test_images_path, images_class=test_images_label, transform=data_transform)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               pin_memory = True,
                                               num_workers = nw,
                                               collate_fn = test_dataset.collate_fn)


    # create model
    net = create_model(num_classes=4).to(device)

    # load model weights
    model_weight_path = r"G:\\EC\\classification\\ResNet\\weight\\ViT.pth"  # 训练后权重的路径
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net = net.cuda()
    net.eval()
    with torch.no_grad():
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        sample_num = 0
        preds = []
        all_labels = []
        probsList = []
        pro = []
        test_loader = tqdm(test_loader, file=sys.stdout)
        for step, data in enumerate(test_loader):
            images, labels = data
            sample_num += images.shape[0]
            pred = net(images.to(device))
            _, y_hat = torch.max(pred, 1)
            probs = pred[:, 1]
            preds.extend(y_hat.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probsList.extend(probs.cpu().numpy())
            pro.extend(pred.cpu().numpy())
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            test_loader.desc = "acc: {:.3f}".format(accu_num.item() / sample_num)
            print(accu_num.item() / sample_num)
        coarse_confusion_matrix = confusion_matrix(preds, all_labels, normalize="true")
        plot_cfm1(coarse_confusion_matrix, r".\confusion_matrix", 'vit')
        results = pd.DataFrame({'predictions': probsList, 'labels': all_labels})
        results.to_csv('my_model.csv', index=False)

        y_score = pro
        y_label = all_labels

        y_label = np.array(y_label)
        y_score = np.array(y_score)
        y_label = label_binarize(y_label, classes=[0, 1, 2, 3])

        n_classes = 4

        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # macro（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2
        plt.figure(figsize=(8, 6))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (AUC = {0:0.4f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (AUC = {0:0.4f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (AUC = {1:0.4f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC curves of ViT-base-16', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()

        return accu_num.item() / sample_num


def plot_cfm1(cf_matrix, save_path, experiment_name):
    # Set the font family for Chinese characters (SimSun) and numbers (Euclid)
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
    matplotlib.rc('font', **font)

    class_names = ['MMRd', 'NSMP', 'P53abn', 'POLEmut']


    # cf_matrix = confusion_matrix(real, pred, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=class_names)

    # d = {'vmin':0,'vmax':0.8}
    disp.plot(cmap='Blues',  values_format='.2f')
    plt.tight_layout()

    # Set custom labels for the bottom and left of the confusion matrix plot
    disp.ax_.set_xlabel('Predicted Image-based Molecular Class of EC', fontsize=14)
    disp.ax_.set_ylabel('True Molecular Class of EC', fontsize=14)
    disp.ax_.set_title('ViT Confusion Matrix (test)', fontsize=14)

    plt.savefig(os.path.join(save_path))
    plt.show()
    plt.close()

    specificity = cf_matrix.diagonal() / cf_matrix.sum(axis=1)
    sensitivity = cf_matrix.diagonal() / cf_matrix.sum(axis=0)

    print(f'Specificity: {specificity}')
    print(f'Sensitivity: {sensitivity}')


if __name__ == '__main__' :
    main()