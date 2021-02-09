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
from random import sample 

EXTRACT_FEATURES = False
IS_TRAINING = True
SEQUENCE_SIZE = 50
SIZE_TRAINING = 68000
SIZE_VAL = 100
SIZE_TEST = 100
ENROLLMENT_SEQUENCES = 15
BATCHS_PER_ITER = 150
GALLERY_SIZE = 10
DATA_ROOT_DIR = 'data'

save_states = True
batch_size = 512
lr = 0.01
num_epochs = 200
input_size = 5
hidden_size = 128
num_layers = 2
n_classes = 2
n_samples = 256

PARTS_TRAINING = np.arange(0, SIZE_TRAINING)
PARTS_VAL = np.arange(SIZE_TRAINING + 5000, SIZE_TRAINING + SIZE_VAL + 5000)
PARTS_TEST = np.arange(SIZE_TRAINING + SIZE_VAL + 1000, SIZE_TRAINING + SIZE_VAL + SIZE_TEST + 1000)

def train(model, device, train_loader, epoch, optimizer):
    model.train()
    contrastive_loss = ContrastiveLoss(margin=1.5)

    counter = 0
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

def validation(model, device, loader, epoch):
    model.eval()

    contrastive_loss = ContrastiveLoss(margin=1.5)

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0
        
        counter = 0
        for batch_idx, (data, target) in enumerate(loader):
            if counter == BATCHS_PER_ITER:
                break

            output1 = model(data[:,0])
            output2 = model(data[:,1])

            target = target.type(torch.LongTensor).to(device)

            loss = loss + contrastive_loss(output1, output2, target)

            counter = counter + 1
        
        print('Validation Loss (epoch={:d}): {:.6f}'.format(epoch, loss))
    
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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if EXTRACT_FEATURES:
        extractor = FeatureExtractor(DATA_ROOT_DIR, SEQUENCE_SIZE)
        extractor.extract_features()

    # training dataset
    training_dataset = TypeNetDataset(DATA_ROOT_DIR, ENROLLMENT_SEQUENCES, PARTS_TRAINING)
    
    # # validation dataset
    validation_dataset = TypeNetDataset(DATA_ROOT_DIR, ENROLLMENT_SEQUENCES, PARTS_VAL)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # test dataset
    # test_dataset = TypeNetDatasetTest(DATA_ROOT_DIR, PARTS_TEST)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # initializing model
    model = SiameseNet(input_size, hidden_size, num_layers).to(device)

    # adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # training
    for epoch in range(num_epochs):
        if save_states:
            torch.save(model.state_dict(), 'states/siamese_typenet_{:03}.pt'.format(epoch))

        if epoch % 5 == 0:
            validation(model, device, val_loader, epoch)

        indices = sample(range(0, len(training_dataset)), batch_size * BATCHS_PER_ITER)
        training_subset = torch.utils.data.Subset(training_dataset, indices)
        train_loader = torch.utils.data.DataLoader(training_subset, batch_size=batch_size, shuffle=True)
        train(model, device, train_loader, epoch, optimizer)

    if save_states:
        torch.save(model.state_dict(), 'states/siamese_typenet_final.pt')

    # test(model, device, test_loader, str(epoch))
         
if __name__ == '__main__':
    main()