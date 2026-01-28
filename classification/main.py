import splitfolders
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from torchsampler import ImbalancedDatasetSampler


def train_model(model, train_loader, val_loader, criterion, optimizer,
                device, args):

    best_accuracy, best_loss, early_stopping_counter = 0, float('inf'), 0
    for epoch in range(args.num_epochs):
        print(f"epoch {epoch + 1}/{args.num_epochs} start!")
        model.train()
        running_loss,correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        print(f"train accuracy {train_accuracy}")
        avg_train_loss = running_loss / len(train_loader)
        print(f"train loss {avg_train_loss}")

        #validation phase
        model.eval()
        val_loss, correct, total, all_labels, all_preds = 0.0, 0, 0, [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_accuracy = correct / total

        #evaluation metric
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"train accuracy: {train_accuracy} | val accuracy: {val_accuracy}")
        print(f"precision: {precision: 4f} | recall: {recall: 4f} | f1: {f1: 4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_path = f"../Model/classification/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + f"{args.dataset}.pth")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter > args.early_stopping:
            print(f"early stopping trigger, stopping training!")
            break

    return save_path + f"{args.dataset}.pth"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'OG')
    parser.add_argument('--path', type = str, default = '../Data/MData/')
    parser.add_argument('--output', type = str, default = '../ModelData/MData')
    parser.add_argument('--image_size', type = int, default = 256)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--model_name', type = str, default = 'vgg16')
    parser.add_argument('--n_classes', type = int, default = 6)
    parser.add_argument('--early_stopping', type = int, default = 20)
    parser.add_argument('--weight_decay', type = float, default = 1e-4)
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--learning_rate', type = float, default = 1e-4)

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output,args.dataset)):
        os.makedirs(os.path.join(args.output,args.dataset))
    if not os.path.exists(os.path.join(args.path, args.dataset)):
        os.makedirs(os.path.join(args.path, args.dataset))

    splitfolders.ratio(os.path.join(args.path, args.dataset),os.path.join(args.output, args.dataset), seed = 2024, ratio = (.7,.15,.15))

    torch.manual_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # Standard for pre-trained models like VGG16
        transforms.RandomHorizontalFlip(p=0.5),  # Breast tissue is mostly symmetric
        transforms.RandomVerticalFlip(p=0.3),  # Less likely but useful for some mammograms
        transforms.RandomRotation(degrees=10),  # Reduce rotation to ≤10° (15° may distort key structures)
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  #Small translations for position invariance
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  #Reduce blur intensity (0.1 to 0.5)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Dataset (Assuming images are stored as 'train/B' and 'train/M' for classes)
    train_dataset = datasets.ImageFolder(os.path.join(args.output, args.dataset) + '/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.output, args.dataset) + '/val', transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.output, args.dataset) + '/test', transform=test_transform)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False, num_workers = 4)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False, num_workers = 4)

    #model
    if args.model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.n_classes)
    elif args.model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.n_classes)
    elif args.model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
    elif args.model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
    else:
        raise ValueError('Model name not recognized')

    model = model.to(device)

    #loss
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = args.weight_decay)
    else:
        raise ValueError('Optimizer not recognized')

    #best_loss = float('inf')
    #early_stopping_count = 0

    best_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, args)

    #load best_model
    model.load_state_dict(torch.load(best_model))
    model.eval()

    correct, total, all_labels, all_preds= 0, 0, [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).long()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        # Compute test metrics
        test_acc = correct / total
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print("Test Set Results:")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")


if __name__ == '__main__':
    main()











