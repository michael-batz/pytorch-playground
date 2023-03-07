import torch
import torchvision
import model

def main():

    # load the image classifier settings
    ics = model.ImageClassifierSettings()

    # definition of training, validation and test data
    train_data = torchvision.datasets.ImageFolder(root="../trainingdata/train", transform=ics.image_transformer)
    val_data = torchvision.datasets.ImageFolder(root="../trainingdata/val", transform=ics.image_transformer)
    test_data = torchvision.datasets.ImageFolder(root="../trainingdata/test", transform=ics.image_transformer)

    # creating data loader
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=ics.batch_size)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=ics.batch_size)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=ics.batch_size)

    # start with training
    for epoch in range(ics.epochs):
        training_loss = 0.0
        valid_loss = 0.0
        # training
        ics.model.train()
        for batch in train_data_loader:
            ics.optimizer.zero_grad()
            input_data = batch[0]
            target_data = batch[1]
            # move data to defined device
            input_data = input_data.to(ics.device)
            target_data = target_data.to(ics.device)
            # put input_data into the neuronal network
            output = ics.model(input_data)
            # calculate difference between output of the model and target
            # with a loss function
            loss = ics.loss_function(output, target_data)
            loss.backward()
            ics.optimizer.step()
            training_loss += loss.data.item() * input_data.size(0)
        training_loss = training_loss / len(train_data_loader.dataset)

        # validiation
        num_correct = 0
        num_examples = 0
        ics.model.eval()
        for batch in val_data_loader:
            input_data = batch[0]
            target_data = batch[1]
            input_data = input_data.to(ics.device)
            target_data = target_data.to(ics.device)
            output = ics.model(input_data)
            loss = ics.loss_function(output, target_data)
            valid_loss += loss.data.item() * input_data.size(0)
            correct = torch.eq(torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)[1], target_data)
            num_correct = num_correct + torch.sum(correct).item()
            num_examples = num_examples + correct.shape[0]
        valid_loss = valid_loss / len(val_data_loader.dataset)

        # output
        print("Epoch {}, Training Loss {}, Valid Loss {}, accuracy {}".format(epoch, training_loss, valid_loss,
            num_correct / num_examples))

    # save model to file
    torch.save(ics.model, "neuronalnet.dat")

if __name__ == "__main__":
    main()
