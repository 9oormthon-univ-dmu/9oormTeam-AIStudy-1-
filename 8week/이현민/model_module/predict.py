
import argparse

import torch
import torchvision

import build_model


parser = argparse.ArgumentParser()

parser.add_argument("--image_path",
                    help="path of the image to predict on")

parser.add_argument("--model_path",
                    default="model_module/models/CNNAugment_TinyVGG_modular.pth",
                    type=str,
                    help="path of the saved model-parameters (.pth)")


args = parser.parse_args()


class_names = ["pizza", "steak", "sushi"]

device = "cuda" if torch.cuda.is_available() else "cpu"


IMG_PATH = args.image_path
print("[INFO] Predicting on {}".format(IMG_PATH))


def load_model(file_path=args.model_path):
    
    model = build_model.CNNAugment_TinyVGG(num_channels=3, 
                                             num_filters=32, 
                                             num_classes=3).to(device)

    print("[INFO] Loading model-parameters from: {}".format(file_path))

    model.load_state_dict(torch.load(file_path))

    return model


def predict_on_image(image_path=IMG_PATH, file_path=args.model_path):

    model = load_model(file_path)

    img_tensor = torchvision.io.read_image(IMG_PATH)
    
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(dim=0)
    
    transform = torchvision.transforms.Resize(size=(64, 64)) # Don't need 'transforms.Compose'
    img_tensor_transformed = transform(img_tensor) 

    
    model.eval()
    
    with torch.inference_mode():
        
        pred_logits = model(img_tensor_transformed.to(device)) 
        pred_prob = pred_logits.softmax(dim=1)
        pred_prob_top = pred_prob.max(dim=1)[0].item()
        pred_label = pred_prob.argmax(dim=1)
        pred_label_class = ["pizza", "steak", "sushi"][pred_label]

        print("[INFO] Predicted class: {}, Prediction probability: {:.3f}".format(pred_label_class,
                                                                                  pred_prob_top))
        
if __name__ == "__main__":
    
    predict_on_image()
