from Models.models import build_unet


if __name__ == "__main__":
    input_shape = (512, 512, 3)
    NUM_CLASSES = 18
    model = build_unet(input_shape, num_classes=NUM_CLASSES)
    model.summary()