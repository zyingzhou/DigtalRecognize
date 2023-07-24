from torchvision import datasets
import torch
from torch.utils.data import DataLoader
import PIL as plt
from DigtalImageDataSet import CustomImageDataset
from DigtalRecNet import LeNetPytorch
from DigtalRecNet import NeuralNetwork
from torch import nn


def train_model(model, total_epoch):
    learning_rate = 0.001
    data_dir = './dataset/mnist/train'
    label_path = './dataset/mnist/train/label.txt'
    training_data = CustomImageDataset(data_dir, label_path)
    train_dataloader = DataLoader(training_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    size = len(train_dataloader)
    loss_fn = nn.CrossEntropyLoss()
    # Display image and label.
    for epoch in range(total_epoch):
        for batch_id, data in enumerate(train_dataloader):
            print('batch_id', batch_id)
            # print(data[0].shape)
            # print(data[0].dtype)
            pred = model(data[0])
            # print(pred.shape)
            # print(pred.dtype)
            # print(f'data[1]{data[1]}')
            # print(data[1].shape)
            # print(data[1].dtype)
            # Initialize the loss function
            # gt = torch.unsqueeze(data[1], 1)
            # print(gt.shape)
            loss = loss_fn(pred, data[1].long())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_id % 100 == 0:
                loss, current = loss.item(), (batch_id + 1) * len(data[0])
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                torch.save(model, f'model_{batch_id}.pth')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load_image()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(device)
    model = LeNetPytorch().to(device)
    print(model)
    # model = NeuralNetwork().to(device)
    # print(model)
    X = torch.rand(1, 1, 28, 28, device=device)
    print(X.dtype)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    # train_model(model, 20)

    model_trained = torch.load('./models/model_900.pth')

    data_dir = './dataset/mnist/val'
    label_path = './dataset/mnist/val/label.txt'
    training_data = CustomImageDataset(data_dir, label_path)
    image, label = training_data.__getitem__(11)
    image = torch.unsqueeze(image, 0)
    pred_dig = model_trained(image)
    print(pred_dig)
    pred_dig_pro = nn.Softmax(dim=1)(pred_dig)
    result = pred_dig_pro.argmax(1)
    print(f"预测结果：{result}")
    print(label)


