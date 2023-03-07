import torch
import torchvision
import PIL
import model

def main():

    # load settings
    ics = model.ImageClassifierSettings(model_file="neuronalnet.dat")

    # load image
    image = PIL.Image.open("../trainingdata/test/fish/upcoming2.jpg")
    image = ics.image_transformer(image)
    image = torch.unsqueeze(image, 0)
    print()

    # evaluation
    ics.model.eval()
    labels = ["cat", "fish"]
    prediction = torch.nn.functional.softmax(ics.model(image), dim=1)
    print(prediction)
    prediction = prediction.argmax()
    print(prediction)
    print(labels[prediction])

if __name__=="__main__":
    main()
