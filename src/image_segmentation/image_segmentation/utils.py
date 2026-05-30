import torch 
import torchvision

def load_model(model="deeplab_v3", device="cuda"):
    if model == "deeplab_v3":
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).to(device)
        model.eval()
    elif model == "lraspp":
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True).to(device)
        model.eval()
    else:
        raise ValueError(f"Model {model} not supported.")
    return model