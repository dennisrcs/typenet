from typenetdataset import TypeNetDataset
from typenetdataset_test import TypeNetDatasetTest
from feature_extractor import FeatureExtractor
from balanced_batch_sampler import BalancedBatchSampler
from contrastive_loss import ContrastiveLoss
from calculate_score import CalculateScore
import torch
from torch import nn
from torch import optim
import numpy as np
from siamesenn import SiameseNet

EXTRACT_FEATURES = False
IS_TRAINING = True
SEQUENCE_SIZE = 50
SIZE_TRAINING = 68000
SIZE_VAL = 1000
SIZE_TEST = 1000
ENROLLMENT_SEQUENCES = 15
GALLERY_SIZE = 10
DATA_ROOT_DIR = 'data'

save_states = False
batch_size = 512
lr = 0.01
num_epochs = 200
input_size = 5
hidden_size = 128
num_layers = 2
n_classes = 2
n_samples = 256

PARTS_TRAINING = np.arange(0, SIZE_TRAINING)
PARTS_VAL = np.arange(SIZE_TRAINING, SIZE_TRAINING + SIZE_VAL)
PARTS_TEST = np.arange(SIZE_TRAINING + SIZE_VAL, SIZE_TRAINING + SIZE_VAL + SIZE_TEST)

def train(model, device, train_loader, epoch, optimizer):
    model.train()
    contrastive_loss = ContrastiveLoss(margin=1.5)

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output1 = model(data[:,0])
        output2 = model(data[:,1])

        target = target.type(torch.LongTensor).to(device)

        loss = contrastive_loss(output1, output2, target)
        loss.backward()

        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx*batch_size, len(train_loader.dataset), 100. * batch_idx*batch_size / len(train_loader.dataset),
            loss.item()))

def validation(model, device, loader):
    model.eval()

    contrastive_loss = ContrastiveLoss(margin=1.5)

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            output1 = model(data[:,0])
            output2 = model(data[:,1])

            target = target.type(torch.LongTensor).to(device)

            loss = loss + contrastive_loss(output1, output2, target)
        
        print('Validation Loss: {:.6f}'.format(loss))
    
def test(model, device, loader, epoch):
    model.eval()
    calculator = CalculateScore()

    with torch.no_grad():

        counter = 0
        score_sum = 0
        for batch_idx, data in enumerate(loader):
            input_data = torch.vstack(data)
            output = model(input_data)

            cur_score = calculator.calculate(output, ENROLLMENT_SEQUENCES, GALLERY_SIZE, epoch, batch_idx)

            score_sum = score_sum + cur_score
            counter = counter + 1

        final_score = score_sum / counter

        print('EER: {:.3f}'.format(final_score))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if EXTRACT_FEATURES:
        extractor = FeatureExtractor(DATA_ROOT_DIR, SEQUENCE_SIZE)
        extractor.extract_features()

    # training dataset
    training_dataset = TypeNetDataset(DATA_ROOT_DIR, ENROLLMENT_SEQUENCES, PARTS_TRAINING)
    training_batch_sampler = BalancedBatchSampler(training_dataset, n_classes, n_samples)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_sampler=training_batch_sampler)
   
    # # validation dataset
    validation_dataset = TypeNetDataset(DATA_ROOT_DIR, ENROLLMENT_SEQUENCES, PARTS_VAL)
    val_batch_sampler = BalancedBatchSampler(validation_dataset, n_classes, n_samples)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_sampler=val_batch_sampler)

    # test dataset
    test_dataset = TypeNetDatasetTest(DATA_ROOT_DIR, PARTS_TEST)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # initializing model
    model = SiameseNet(input_size, hidden_size, num_layers).to(device)

    # adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # first validation loss
    validation(model, device, val_loader)

    # training
    for epoch in range(num_epochs):
        train(model, device, train_loader, epoch, optimizer)
        validation(model, device, val_loader)
        
        if save_states:
            torch.save(model.state_dict(), 'siamese_typenet_{:03}.pt'.format(epoch))
        
    test(model, device, test_loader, str(epoch))
         
if __name__ == '__main__':
    main()