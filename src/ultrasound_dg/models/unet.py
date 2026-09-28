import segmentation_models_pytorch as smp


def create_unet() -> smp.Unet:
    """Create a binary U-Net with an ImageNet-pretrained ResNet34 encoder."""

    return smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
    )
