

# this cod econtains the three losss fucntions for the cycel GAN model


# the first one is the adversarial loss, which is used to train the generator and discriminator
# the second one is the cycle consistency loss, which is used to ensure that the generated images are consistent with the original images
# the third one is the identity loss, which is used to ensure that the generator does not change the images that are already in the target domain


import torch
import torch.nn as nn


class CycleGANLosses:
    """
    Container for CycleGAN loss functions.
    """

    def __init__(self):
      
        self.adversarial_loss_fn = nn.BCEWithLogitsLoss()
        self.cycle_consistency_loss_fn = nn.L1Loss()
        self.identity_loss_fn = nn.L1Loss()



    def adversarial_loss(self, prediction, isReal):
      
        if isReal:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)

        return self.adversarial_loss_fn(prediction, target)



    def cycle_consistency_loss(self, real_image, reconstructed_image):
        return self.cycle_consistency_loss_fn(reconstructed_image, real_image) * 10.0



    def identity_loss(self, real_image, identity_image):
        return self.identity_loss_fn(identity_image, real_image) * 5.0
          
    





     
