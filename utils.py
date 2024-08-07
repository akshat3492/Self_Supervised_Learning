
def save_rotation_model():
    path = "save_model/self_supervised_rotation_model.pth"
    torch.save(net.state_dict(), path)


# Both the self-supervised rotation task and supervised CIFAR10 classification are
# trained with the CrossEntropyLoss, so I can use general training loop code.
def train(net, criterion, optimizer, num_epochs, decay_epochs, init_lr, task):

    best_accuracy = 0.0

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        start_time = time.time()

        net.train()

        for i, (imgs, imgs_rotated, rotation_label, cls_label) in enumerate(trainloader, 0):

            adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs)

            # Set the data to the correct device; Different task will use different inputs and labels
            #
            if task == "rotation":
                inputs = imgs_rotated.to(device)
                labels = rotation_label.to(device)
            else:
                inputs = imgs.to(device)
                # print(inputs.get_device())
                labels = cls_label.to(device)


            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #

            # Get predicted results
            predicted = torch.argmax(outputs, dim = 1)

            # print statistics
            print_freq = 100
            running_loss += loss.item()

            # calc acc
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            if i % print_freq == (print_freq - 1):    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_freq:.3f} acc: {100*running_correct / running_total:.2f} time: {time.time() - start_time:.2f}')
                running_loss, running_correct, running_total = 0.0, 0.0, 0.0
                start_time = time.time()

        # Run the run_test() function after each epoch; Set the model to the evaluation mode.
        
        net.eval()
        accuracy = run_test(net, testloader, criterion, task)

        if task == "rotation" and accuracy > best_accuracy:
            save_rotation_model()
            best_accuracy = accuracy

        # if task == "classification" and accuracy > 60:
        #   print("Early Classification Stoppping")
        #   break

        if task == "rotation" and accuracy > 78:
          print("Early Rotation Stoppping")
          break

    print('Finished Training')


def run_test(net, testloader, criterion, task):
    correct = 0
    total = 0
    avg_test_loss = 0.0
    # Don't need to calculate gradients since its an evaluation step, so use torch.no_grad() to not track gradients and speed up the computation
    with torch.no_grad():
        for images, images_rotated, labels, cls_labels in testloader:
            if task == 'rotation':
              images, labels = images_rotated.to(device), labels.to(device)
            elif task == 'classification':
              images, labels = images.to(device), cls_labels.to(device)
            # Calculate outputs by running images through the network
            # The class with the highest energy is what we choose as prediction
            outputs = net(images.to(device))
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # loss
            avg_test_loss += criterion(outputs, labels)  / len(testloader)
    print('TESTING:')
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
    print(f'Average loss on the 10000 test images: {avg_test_loss:.3f}')

    return 100 * (correct / total)


def adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr