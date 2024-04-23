import os
import json

import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import torch
import sys
from torchvision import transforms
import matplotlib.pyplot as plt
from my_dataset import MyDataSet
from model import efficientnetv2_s as create_model
from test_split import read_split_data

test_data_path = r"G:\EC\ECDataset\test"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    test_images_path, test_images_label = read_split_data(test_data_path)
    batch_size = 4
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = {
        "test": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    test_dataset = MyDataSet(images_path=test_images_path, images_class=test_images_label, transform=data_transform["test"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=test_dataset.collate_fn)


    # create model
    net = create_model(num_classes=4).to(device)

    # load model weights
    model_weight_path = r"G:\EC\classification\EffecientNet\weights\model-0.pth" # 训练后权重的路径
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net = net.cuda()
    net.eval()
    with torch.no_grad():
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        sample_num = 0
        preds = []
        all_labels = []
        probsList = []
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
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            test_loader.desc = "acc: {:.3f}".format(accu_num.item() / sample_num)
            print(accu_num.item() / sample_num)
        coarse_confusion_matrix = confusion_matrix(preds, all_labels, normalize="true")
        plot_cfm1(coarse_confusion_matrix, r".\efficientnet", 'efficientnet')
        results = pd.DataFrame({'predictions': probsList, 'labels': all_labels})
        results.to_csv('my_model.csv', index=False)
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
    disp.ax_.set_title('EfficientNet Confusion Matrix (test)', fontsize=14)

    plt.savefig(os.path.join(save_path))
    plt.show()
    plt.close()

    specificity = cf_matrix.diagonal()/cf_matrix.sum(axis=1)
    sensitivity = cf_matrix.diagonal()/cf_matrix.sum(axis=0)

    print(f'Specificity: {specificity}')
    print(f'Sensitivity: {sensitivity}')


if __name__ == '__main__':
    main()
