import torch
from Models.TraditionalCNN.utilis import load_data


def inference(config, model):
    model.eval()

    test_loader = load_data.get_test_loader(config)

    total_accuracy = 0.0
    num_samples = 0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image = image.to(config.device)
            label = label.to(config.device)

            prediction = model(image)
            # special branch for Inception network
            if config.network in ['inception_v3', 'googlenet']:
                prediction = prediction[0]

            total_accuracy += torch.sum(torch.eq(prediction.argmax(dim=1), label)).item()
            num_samples += image.shape[0]

    print(f"Accuracy: {total_accuracy / num_samples}")
