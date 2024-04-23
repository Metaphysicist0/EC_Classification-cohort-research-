from torch.utils.tensorboard import SummaryWriter
from vit_model import vit_large_patch32_224_in21k as create_model
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), r"G:\EC\ECDataset"))  # get data root path
    image_path = os.path.join(data_root)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)



    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    model_weight = 1

    model = create_model(num_classes=4, has_logits=False).to(device)
    if model_weight == 1:
        model_weight_path = r"G:\OneDrive\Desktop\weight\VITL2500.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path, map_location=device)
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))
    for name, para in model.named_parameters():
         # 除head, pre_logits外，其他权重全部冻结
        if "head" not in name and "pre_logits" not in name:
            para.requires_grad_(False)
        else:
            print("training {}".format(name))
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    epochs = 2000
    best_acc = 0.0
    save_path = 'G:\\EC\\classification\\ResNet\\weight\\ViT.pth'
    train_steps = len(train_loader)
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,loss)
            average_train_loss = running_loss / val_num
        train_losses.append(average_train_loss)
        # validate
        # validate
        model.eval()
        val_loss = 0.0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                outputs = model(val_images)
                loss = loss_function(outputs, val_labels)
                val_loss += loss.item()

                _, predict_y = torch.max(outputs, dim=1)
                acc += torch.sum(predict_y == val_labels.data)

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
        # Calculate average validation loss
        val_losses.append(val_loss / len(validate_loader))

        # Calculate accuracy
        val_accurate = acc / val_num
        val_accuracies.append(val_accurate)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
    epochs_list = list(range(1, epochs + 1))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)

    val_accuracies_np = [tensor.cpu().numpy() for tensor in val_accuracies]
    plt.plot(epochs_list, val_accuracies_np, label='Validation Accuracy', color='green')

    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print('Finished Training')

if __name__ == '__main__':
    main()