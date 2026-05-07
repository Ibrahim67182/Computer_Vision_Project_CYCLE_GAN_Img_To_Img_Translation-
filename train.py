
# the main training loop of the cycle GAN model

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os 
from tqdm import tqdm 

import dataset
from models import generator , discriminator 
import losses

import random
import traceback
import warnings, glob


# -------------------hyperparameters---------------------- 

DATASET = "monet2photo"  # need to change just name of this param to train same model on different dataset
BATCH_SIZE = 1
LEARNING_RATE = 0.0002     
BETA1          = 0.5             # Adam optimizer momentum
BETA2          = 0.999           # Adam optimizer momentum
NUM_EPOCHS     = 400               # can adjust epochs accordingly from 200 (ideal) - 300 (horse2zebra 300 , monet 400 , summer 250)
DECAY_EPOCH    = 200               # epoch to start lr decay (100)
IMAGE_SIZE     = 256
NUM_WORKERS    = 4               # parallel data loading workers
LAMBDA_CYCLE   = 10.0            # cycle consistency loss weight
LAMBDA_IDENTITY= 0.5            # identity loss weight  (0.5 horse2zebra , 0.5 monet2photo and 1.0 summer2winter)



# ----------------PATHS----------------------

DATASET_ROOT = "datasets"  # root directory for datasets
CHECKPOINT_DIR = "checkpoints"  # directory to save model checkpoints
SAMPLE_DIR = "samples"  # directory to save generated sample images during training



# -------------------device----------------------

device  = "cuda" if torch.cuda.is_available() else "cpu"



#-------------------------------replay buffer class definition-------------------------

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []

    def push(self, images):
        returned_images = []

        for image in images.detach():
            image = image.unsqueeze(0)  # [C,H,W] → [1,C,H,W]

            if len(self.buffer) < self.max_size:
                # Buffer not full yet — just store and return the new image
                self.buffer.append(image)
                returned_images.append(image)

            else:
                # Buffer is full — 50/50 chance
                if random.random() > 0.5:
                    # Return a random old image from buffer, replace it with new one
                    idx = random.randint(0, self.max_size - 1)
                    old_image = self.buffer[idx].clone()
                    self.buffer[idx] = image
                    returned_images.append(old_image)
                else:
                    # Return the new image as-is, don't store it
                    returned_images.append(image)

        return torch.cat(returned_images, dim=0)  # stack back into a batch




if __name__ == "__main__":


    # ----------------------------models creation-------------------------------------

    # creating generators instances 

    G_AB = generator.Generator().to(device)         # generator that translates images from domain A to domain B
    G_BA = generator.Generator().to(device)         # generator that translates images from domain B to domain A

    # creating discriminators instances

    D_A = discriminator.Discriminator().to(device)   # discriminator that distinguishes real and fake images in domain A
    D_B = discriminator.Discriminator().to(device)   # discriminator that distinguishes real and fake images in domain B


    # ------INTIALIZE WEIGHTS OF MODELS-------

    generator.initialize_weights(G_AB)
    generator.initialize_weights(G_BA)

    discriminator.initialize_weights(D_A)
    discriminator.initialize_weights(D_B)

    # -------------------creating optimizers----------------------

    g_optimizer = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2))

    d_optimizer = optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2))


    #---------------------learning rate schedulers----------------------

    # scheduler for generators
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, 
        lr_lambda=lambda epoch: 1.0 - max(0, epoch - DECAY_EPOCH) / (NUM_EPOCHS - DECAY_EPOCH))


    # scheduler for discriminators — same lambda, different optimizer
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer,
        lr_lambda=lambda epoch: 1.0 - max(0, epoch - DECAY_EPOCH) / (NUM_EPOCHS - DECAY_EPOCH))




    # creating fake buffer instances for both domain to store atleast 50 generated images from each domain to be used in discriminator training step to stabilize training and prevent overfitting of discriminators on recent generator outputs

    fake_A_buffer = ReplayBuffer(max_size=50)
    fake_B_buffer = ReplayBuffer(max_size=50)




    # ───────────────── RESUME FROM CHECKPOINT if it exists ─────────────────────────────────────

    def find_latest_checkpoint(checkpoint_dir, dataset):
    
        """Scans checkpoint folder and returns the latest epoch number, or 0 if none found."""
        folder = os.path.join(checkpoint_dir, dataset)
        if not os.path.exists(folder):
            return 0
        # G_AB checkpoints are named G_AB_epoch_025.pth — find the highest epoch
        import glob
        files = glob.glob(os.path.join(folder, "G_AB_epoch_*.pth"))
        if not files:
            return 0
        epochs = []
        for f in files:
            try:
                epoch_num = int(os.path.basename(f).split("_epoch_")[1].replace(".pth", ""))
                epochs.append(epoch_num)
            except:
                continue
        return max(epochs) if epochs else 0




    start_epoch = 0
    latest_epoch = find_latest_checkpoint(CHECKPOINT_DIR, DATASET)



    try:
        if latest_epoch > 0:
            checkpoint_folder = os.path.join(CHECKPOINT_DIR, DATASET)
            print(f"\n[RESUME] Found checkpoint at epoch {latest_epoch}. Resuming from epoch {latest_epoch + 1}...\n")

            G_AB.load_state_dict(torch.load(os.path.join(checkpoint_folder, f"G_AB_epoch_{latest_epoch:03d}.pth"), map_location=device))
            G_BA.load_state_dict(torch.load(os.path.join(checkpoint_folder, f"G_BA_epoch_{latest_epoch:03d}.pth"), map_location=device))
            D_A.load_state_dict(torch.load(os.path.join(checkpoint_folder,  f"D_A_epoch_{latest_epoch:03d}.pth"),  map_location=device))
            D_B.load_state_dict(torch.load(os.path.join(checkpoint_folder,  f"D_B_epoch_{latest_epoch:03d}.pth"),  map_location=device))

            start_epoch = latest_epoch
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_epoch):
                    g_scheduler.step()
                    d_scheduler.step()
        else:
            print("\n[FRESH START] No checkpoints found. Starting from epoch 1...\n")

    except Exception as e:
        print(f"\n[ERROR] Resume block crashed: {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────



    # -----------------------------------data loaders-------------------------------------


    train_loader = dataset.get_loader(dataset=DATASET, mode="train", root=DATASET_ROOT, batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    test_loader = dataset.get_loader(dataset=DATASET, mode="test", root=DATASET_ROOT, batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE, num_workers=NUM_WORKERS, shuffle=False)



    #-------------- deifning loss functions object -------------------------------------

    criterion = losses.CycleGANLosses() 



    # -------------------------------------------training loop--------------------------------------------


    for epoch in range( start_epoch , NUM_EPOCHS):
        
        # progress bar for the current epoch 
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=True)

        for batch in loop: 
            
            # ── 1. Get real images ────────────────────────────────────────
        
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)


            # 2. GENERATOR STEP
        
            g_optimizer.zero_grad()

            # ── Forward pass ──────────────────────────────────────────────
            
            fake_B = G_AB(real_A)          # A → B
            fake_A = G_BA(real_B)          # B → A


            # ── Adversarial loss ──────────────────────────────────────────
            
            D_B_fake = D_B(fake_B)         # discriminator judges fake_B
            D_A_fake = D_A(fake_A)         # discriminator judges fake_A

            loss_adv_AB = criterion.adversarial_loss(D_B_fake, isReal=True)   # G wants D to think fake is real
            loss_adv_BA = criterion.adversarial_loss(D_A_fake, isReal=True)


            # ── Cycle consistency loss ─────────────────────────────────────
            

            reconstructed_A = G_BA(fake_B)     # A → fake_B → reconstructed_A
            reconstructed_B = G_AB(fake_A)     # B → fake_A → reconstructed_B

            loss_cycle_A = criterion.cycle_consistency_loss(real_A, reconstructed_A)
            loss_cycle_B = criterion.cycle_consistency_loss(real_B, reconstructed_B)



            # ── Identity loss ─────────────────────────────────────────────


            identity_A = G_BA(real_A)      # G_BA on real_A should return real_A unchanged
            identity_B = G_AB(real_B)      # G_AB on real_B should return real_B unchanged

            loss_identity_A = criterion.identity_loss(real_A, identity_A)
            loss_identity_B = criterion.identity_loss(real_B, identity_B)



            # ── Total generator loss ──────────────────────────────────────
            
            loss_G = (loss_adv_AB + loss_adv_BA) + LAMBDA_CYCLE * (loss_cycle_A + loss_cycle_B) + LAMBDA_IDENTITY * (loss_identity_A + loss_identity_B)

            loss_G.backward()
        
            g_optimizer.step()

    #---------------------------------------------------------------------------
        
            # 3. DISCRIMINATOR STEP
        
        
            d_optimizer.zero_grad()

        
            # ── D_A loss ──────────────────────────────────────────────────
        
        
            fake_A_buffered = fake_A_buffer.push(fake_A)

            D_A_real = D_A(real_A)
            D_A_fake = D_A(fake_A_buffered.detach())

            loss_D_A_real = criterion.adversarial_loss(D_A_real, isReal=True)
            loss_D_A_fake = criterion.adversarial_loss(D_A_fake, isReal=False)
            
            loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2

            
            # ── D_B loss ──────────────────────────────────────────────────
            
            fake_B_buffered = fake_B_buffer.push(fake_B)

            D_B_real = D_B(real_B)
            D_B_fake = D_B(fake_B_buffered.detach())

            loss_D_B_real = criterion.adversarial_loss(D_B_real, isReal=True)
            loss_D_B_fake = criterion.adversarial_loss(D_B_fake, isReal=False)
        
            loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2

        
            # ── Total discriminator loss ───────────────────────────────────
            
            loss_D = (loss_D_A + loss_D_B) / 2

            loss_D.backward()
            d_optimizer.step()


            # ── tqdm progress bar update ───────────────────────────────────
        
            loop.set_postfix(
                loss_G=f"{loss_G.item():.4f}",
                loss_D=f"{loss_D.item():.4f}",
                loss_cycle=f"{(loss_cycle_A + loss_cycle_B).item():.4f}"
            )

        
        
        
        
        # -------------------LOGGING, SCHEDULING, SAMPLES, CHECKPOINTS after each epoch -------------------
        
        
        
        # ── 1. Print epoch losses ─────────────────────────────────────
        
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Loss_G: {loss_G.item():.4f} | "
            f"Loss_D: {loss_D.item():.4f} | "
            f"Cycle: {(loss_cycle_A + loss_cycle_B).item():.4f} | "
            f"Identity: {(loss_identity_A + loss_identity_B).item():.4f}")

        
        # ── 2. Step LR schedulers ─────────────────────────────────────
        
        g_scheduler.step()
        d_scheduler.step()

        
        # ── 3. Every 5 epochs — save samples and checkpoints ─────────
        
        if (epoch + 1) % 5 == 0:

            
            # ── Save sample images ────────────────────────────────────
            
            G_AB.eval()
            G_BA.eval()

            with torch.no_grad():
            
                sample_batch = next(iter(test_loader))
                sample_A = sample_batch["A"].to(device)
                sample_B = sample_batch["B"].to(device)

                sample_fake_B = G_AB(sample_A)      # A → B
                sample_fake_A = G_BA(sample_B)      # B → A

            
                # Denormalize: [-1,1] → [0,1]
            
                sample_A      = sample_A      * 0.5 + 0.5
                sample_fake_B = sample_fake_B * 0.5 + 0.5
                sample_B      = sample_B      * 0.5 + 0.5
                sample_fake_A = sample_fake_A * 0.5 + 0.5

        
            # Save to samples/horse2zebra/epoch_005/
        
            sample_dir = os.path.join(SAMPLE_DIR, DATASET, f"epoch_{epoch+1:03d}")
        
            os.makedirs(sample_dir, exist_ok=True)

            torchvision.utils.save_image(sample_A,      os.path.join(sample_dir, "real_A.png"))
            torchvision.utils.save_image(sample_fake_B, os.path.join(sample_dir, "fake_B.png"))
            torchvision.utils.save_image(sample_B,      os.path.join(sample_dir, "real_B.png"))
            torchvision.utils.save_image(sample_fake_A, os.path.join(sample_dir, "fake_A.png"))

            print(f"  Samples saved → {sample_dir}")

            G_AB.train()
            G_BA.train()

            # ── Save checkpoints ──────────────────────────────────────
            checkpoint_dir = os.path.join(CHECKPOINT_DIR, DATASET)
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save(G_AB.state_dict(), os.path.join(checkpoint_dir, f"G_AB_epoch_{epoch+1:03d}.pth"))
            torch.save(G_BA.state_dict(), os.path.join(checkpoint_dir, f"G_BA_epoch_{epoch+1:03d}.pth"))
            torch.save(D_A.state_dict(),  os.path.join(checkpoint_dir, f"D_A_epoch_{epoch+1:03d}.pth"))
            torch.save(D_B.state_dict(),  os.path.join(checkpoint_dir, f"D_B_epoch_{epoch+1:03d}.pth"))

            print(f"  Checkpoints saved → {checkpoint_dir}")

    
