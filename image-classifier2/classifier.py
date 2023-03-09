import torch
import torchvision
import PIL
import model

def main():

    # load settings
    ics = model.ImageClassifierSettings(model_file="neuronalnet.dat")

    # load image
    image = PIL.Image.open("../trainingdata/test/fish/246232021_a806f30fbc.jpg")
    image = ics.image_transformer(image).unsqueeze(0)

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
