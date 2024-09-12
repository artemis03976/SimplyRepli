import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from global_utilis import save_and_load
from global_utilis.early_stopping import EarlyStopping
from Models.ImageClassification.utilis import load_data


def train(config, model):
    # get train and val loader
    train_loader, val_loader = load_data.get_train_val_loader(config)

    # pre-defined loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    num_epochs = config.epochs

    # initialize early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        # main train step
        train_loss, train_accuracy = train_step(model, config, train_info, criterion, optimizer)
        # main val step
        val_loss, val_accuracy = validation(model, config, val_loader, criterion)

        print(
            '\nEpoch [{}/{}], Average Loss: {:.4f}, Average Accuracy: {:.4f}'
            .format(epoch + 1, num_epochs, train_loss, train_accuracy)
        )
        print(
            'Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'
            .format(val_loss, val_accuracy)
        )

        # check early stopping condition
        early_stopping(val_accuracy)
        if early_stopping.early_stop:
            print("Need Early Stopping")
            break

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    # switch mode
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    # recounting for mean accuracy
    num_samples = 0

    for batch_idx, (image, label) in enumerate(train_info):
        image = image.to(config.device)
        label = label.to(config.device)

        prediction = model(image)

        # special branch for Inception network
        if config.network == 'inception_v3':
            loss_main = criterion(prediction['main'], label)
            loss_aux = criterion(prediction['aux'], label)
            loss = loss_main + 0.4 * loss_aux
        elif config.network == 'googlenet':
            loss_main = criterion(prediction['main'], label)
            loss_aux1 = criterion(prediction['aux_1'], label)
            loss_aux2 = criterion(prediction['aux_2'], label)
            loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
        else:
            loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # calculate accuracy
        if config.network in ['inception_v3', 'googlenet']:
            prediction = prediction['main']
        acc = torch.sum(torch.eq(prediction.argmax(dim=1), label))
        total_accuracy += acc.item()
        num_samples += image.shape[0]

        # set progress bar info
        train_info.set_postfix(Loss=loss.item(), Acc=acc.item() / image.shape[0])

    return total_loss / len(train_info), total_accuracy / num_samples


def validation(model, config, val_loader, criterion):
    # switch mode
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    # recounting for mean accuracy
    num_samples = 0

    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(config.device)
            label = label.to(config.device)

            prediction = model(image)
            # special branch for Inception network
            if config.network in ['inception_v3', 'googlenet']:
                prediction = prediction['main']

            total_loss += criterion(prediction, label).item()
            # calculate accuracy
            total_accuracy += torch.sum(torch.eq(prediction.argmax(dim=1), label)).item()
            num_samples += image.shape[0]

    return total_loss / len(val_loader), total_accuracy / num_samples
