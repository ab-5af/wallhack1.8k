import torch
import torchvision.models as models
import torch.nn as nn
import datasets as data
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np

# supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train(opt):
    if torch.cuda.is_available():
        device = "cuda:"+opt.device
    else:
        device = "cpu"

    print("Loading Wallhack1.8k dataset...")
    def train_val_test_dataset(dataset):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=False, random_state=42)
        val_idx, test_idx = train_test_split(val_idx, test_size=0.5, shuffle=False, random_state=42)
        return Subset(dataset, train_idx), Subset(dataset, val_idx),Subset(dataset, test_idx),train_idx

    def load_sequence(scenario,antenna, sequence):
        sequenceDataset = data.WallhackDataset(f"{opt.data}/{scenario}/{antenna}/{sequence}.csv", opt=opt)
        train, val, test,train_idx = train_val_test_dataset(sequenceDataset)
        trainLabels = sequenceDataset.c[train_idx]
        return train, val, test, trainLabels

    sequences = ["b1","w1","w2","w3","w4","w5","ww1","ww2","ww3","ww4","ww5"]
    trainSubsets = []
    valSubsets = []
    testSubsets = []

    # perform 8:1:1 split on each sequence and combine them into train, val and test datasets
    for s in sequences:
        train_subset, val_subset, test_subset, l = load_sequence(opt.scenario,opt.antenna, s)
        trainSubsets.append(train_subset)
        valSubsets.append(val_subset)
        testSubsets.append(test_subset)

    # create training, validation and test dataloader
    datasetTrain = torch.utils.data.ConcatDataset(trainSubsets)
    datasetVal = torch.utils.data.ConcatDataset(valSubsets)
    datasetTest = torch.utils.data.ConcatDataset(testSubsets) 
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,batch_size=opt.bs,num_workers=opt.workers,drop_last=True,shuffle=True)
    dataloaderVal = torch.utils.data.DataLoader(datasetVal,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
    dataloaderTest = torch.utils.data.DataLoader(datasetTest,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)

    # ==========================================
    # MODIFICATION 1 : Le Modèle EfficientNetV2
    # ==========================================
    model = models.efficientnet_v2_s(weights=None)
    
    # 1 canal en entrée au lieu de 3 (pour le spectrogramme)
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, original_conv.out_channels, 
                                     kernel_size=original_conv.kernel_size, 
                                     stride=original_conv.stride, 
                                     padding=original_conv.padding, 
                                     bias=False)
    
    # 3 classes en sortie au lieu de 1000 (Vide, Marche, Marche+Gestes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    
    model.to(device)

    # ==========================================
    # MODIFICATION 2 : Optimiseur et Loss
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0

    print("Training...")
    for epoch in tqdm(range(opt.epochs), desc='Epochs', unit='epoch'):
        
        # training loop
        model.train()
        train_correct = 0
        train_total = 0
        for batch in tqdm(dataloaderTrain, desc=f'Epoch {epoch + 1}/{opt.epochs}', unit='batch', leave=False):
            feature_window, c = [x.to(device) for x in batch]
            feature_window = feature_window.float()
            
            optimizer.zero_grad()
            prediction = model(feature_window)
            loss = criterion(prediction, c)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(prediction.data, 1)
            train_total += c.size(0)
            train_correct += (predicted == c).sum().item()

        # validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(dataloaderVal, leave=False):
                feature_window, c = [x.to(device) for x in batch]
                feature_window = feature_window.float()
                prediction = model(feature_window)
                
                _, predicted = torch.max(prediction.data, 1)
                val_total += c.size(0)
                val_correct += (predicted == c).sum().item()

        # Sauvegarde du meilleur modèle
        val_acc = val_correct / val_total
        print(f"\nEpoch {epoch+1} - Train Acc: {train_correct/train_total:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("-> Nouveau meilleur modèle sauvegardé !")

    # test loop
    print("Testing...")
    # On recharge le meilleur modèle pour le test final
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in tqdm(dataloaderTest):
            feature_window, c = [x.to(device) for x in batch]
            feature_window = feature_window.float()
            prediction = model(feature_window)
            
            _, predicted = torch.max(prediction.data, 1)
            test_total += c.size(0)
            test_correct += (predicted == c).sum().item()
            
    print(f"\nPrécision Finale sur le Test (Phase 1 Validée) : {test_correct/test_total:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/wallhack1.8k', help='directory of the wallhack1.8k dataset')
    parser.add_argument('--ws', type=int, default=401, help='feature window size (i.e. the number of WiFi packets)')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--scenario', default='LOS', type=str, help='Type of scenario (LOS or NLOS)')
    parser.add_argument('--antenna', default='BQ', type=str, help='Type of antenna (BQ or PIFA)')
    opt = parser.parse_args()

    train(opt)
    print("Done!")
    torch.cuda.empty_cache()
