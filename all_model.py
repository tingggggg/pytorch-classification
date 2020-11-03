from models import *

def get_model(model_name, num_classes=10):
    if "vgg" in model_name.lower():
        model_type = model_name.lower().replace("vgg", "")
        try:
            return VGG(model_type, num_classes=num_classes)
        except:
            print(f"{model_type} not in vgg builder")
            print("input 11, 13, 16 or 19")
            exit(0)
    
    elif "resnet" in model_name.lower():
        model_type = model_name.lower().replace("resnet", "")
        try:
            return RESNET(model_type, num_classes=num_classes)
        except:
            print(f"{model_type} not in ResNet builder")
            print("input 18, 34, 50, 101 or 152")
            exit(0)

    elif "resnext" in model_name.lower():
        return ResNeXt29_32x4d(num_classes=num_classes)

    elif "mobilenet" in model_name.lower():
        return MobileNet(num_classes=num_classes)

    elif "efficientnet" in model_name.lower():
        model_type = model_name.lower().replace("efficientnet", "")
        return Efficiennet(model_type=model_type, num_classes=num_classes)

    elif "ghostnet" in model_name.lower():
        return Ghostnet(num_classes=num_classes)

    elif "shufflenetv1" in model_name.lower():
        return ShuffleNetG2(num_classes=num_classes)

    elif "shufflenetv2" in model_name.lower():
        return ShuffleNetV2(num_classes=num_classes)

