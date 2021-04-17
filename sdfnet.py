import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

def get_resnet_global_feature_vector(img_in, model, layer):
    features = torch.zeros([32, 512, 1, 1])
    
    # copy the feature from the model's last pooling layer
    def copy_feature(m, i, o):
        # print("resnet feature size: {}".format(o.data.shape))
        # print(o.data)
        features.copy_(o.data)

    h = layer.register_forward_hook(copy_feature)

    model(img_in)
    
    h.remove()
    # Return the feature vector
    return features
        

class sdfnet(nn.Module):
    def __init__(self):

        super(sdfnet, self).__init__()
        
        # resnet
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

        self.extract_image_features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.MaxPool2d(2),
        )
        
        self.create_global_image_descriptors = nn.AvgPool2d(68)

        self.extract_point_features = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )



    # x represents our data
    def forward(self, image_batch, points_batch):

        # Get image features using resnet, 
        model = self.resnet
        layer = model._modules.get('avgpool')
        model.eval()
        image_features = get_resnet_global_feature_vector(image_batch, model, layer)
        # size [32, 512, 5, 5] from layer4
        # size [32, 512, 1, 1] from avgpool
        print("img_feature from resnet: {}".format(image_features.shape))
        image_features = self.extract_image_features(image_batch)
        print(image_features.shape)
        image_descriptors = self.create_global_image_descriptors(image_features).squeeze()
        print(image_descriptors.shape)

        # Get point features
        point_features = self.extract_point_features(points_batch)
        print(point_features.shape)

        # repeat image_descriptors to be same shape as point_features
        image_descriptors = image_descriptors.repeat(1, points_batch.shape[1]).reshape(points_batch.shape[0], points_batch.shape[1], 64)
        print(image_descriptors.shape)
        # print(image_descriptors[0, 0, :])
        # print(image_descriptors[0, -1, :])
        # print(image_descriptors[1, 0, :])
        # print(image_descriptors[1, -1, :])

        # combine point features and image features
        features = torch.cat((image_descriptors, point_features), -1)
        print(features.shape)

        sdf = self.decoder(features)
        print(sdf.shape)


        return sdf
