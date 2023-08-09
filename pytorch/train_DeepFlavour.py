import torch 
import torch.nn as nn
from pytorch_first_try import training_base
from pytorch_deepjet import *
from pytorch_deepjet_run2 import *
from pytorch_deepjet_transformer import DeepJetTransformer
from pytorch_ranger import Ranger

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

# set to True, if old training should be continued
resume_silently = False
recreate_silently = False

num_epochs = 30

lr_epochs = max(1, int(num_epochs * 0.3))
lr_rate = 0.01 ** (1.0 / lr_epochs)
mil = list(range(num_epochs - lr_epochs, num_epochs))

model = DeepJet_Run2(num_classes = 6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = cross_entropy_one_hot
optimizer = Ranger(model.parameters(), lr = 5e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = mil, gamma = lr_rate)

train=training_base(model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, resume_silently = resume_silently, recreate_silently=recreate_silently)
train.train_data.maxFilesOpen=1

attack = None
att_magnitude = 0.
restrict_impact = -1

model,history = train.trainModel(nepochs=num_epochs+lr_epochs, 
                                 batchsize=4000,
                                 attack=attack,
                                 att_magnitude=att_magnitude,
                                 restrict_impact=restrict_impact)
