import torch


def load_model(weights_path=None):
    """
    Loads MobileNetV2 pre-trained on ImageNet from PyTorch's cloud.
    Modifies last layers to fit our pose regression problem.
    """
    # Base model is MobileNetV2 from PyTorch's hub
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    # We modify the classifier of MobileNetV2 with a custom regressor
    in_features = list(model.classifier.children())[-1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=in_features,
            out_features=2048,
            bias=True
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=2048,
            out_features=7,
            bias=True
        )
    )
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    return model
