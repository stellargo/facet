# %%
import torch
from datetime import datetime
from data.presidentdata import get_dataloaders
from model.beta_vae import BetaVAE
import logging
import wandb
from argparse import ArgumentParser
import numpy as np
from utils.visualize_mediapipe import visualize_landmarks
from torch.nn import functional as F


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = ArgumentParser()
parser.add_argument('--device', type=str, default="0", help="cuda device number")
parser.add_argument('--epochs', type=int, default=500, help="number of epochs to train")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--batch_size', type=int, default=512, help="batch size")
parser.add_argument('--val_split', type=float, default=0.1, help="validation split ratio")
parser.add_argument('--test_split', type=float, default=0.1, help="test split ratio")
parser.add_argument('--train_workers', type=int, default=16, help="number of workers for training data loader")
parser.add_argument('--val_workers', type=int, default=8, help="number of workers for validation data loader")
parser.add_argument('--test_workers', type=int, default=8, help="number of workers for test data loader")
parser.add_argument('--wandb', action='store_true', help="enable wandb logging")
parser.add_argument('--random_rotation', type=bool, default=False, help="enable random rotation")
parser.add_argument('--z_dim', type=int, default=16, help="latent dimension")
parser.add_argument('--den', type=float, default=10000000, help="denominator for kld weight")
parser.add_argument('--timestamp', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
parser.add_argument('--enable_kld', action='store_true', help="enable kld loss")
parser.add_argument('--model_path', type=str, default="./model/trained/beta_vae.pth", help="initialize starting model")
args, _ = parser.parse_known_args()
print(args)

logging.info("Loading model...")
device = torch.device("cuda:" + args.device)
input_dim = 478 * 3
model = BetaVAE(input_dim, args.z_dim).to(device)
if args.model_path != "":
    model = torch.load(args.model_path)

model.to(device)

model.train()
logging.info("Model loaded to device.")


def loss_function(recons, input, mu, log_var, kld_weight, beta=4):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + beta * kld_weight * kld_loss
        return loss, recons_loss, kld_loss



optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train_one_epoch(loader):
    running_loss = 0.
    running_recon_loss = 0.
    running_kld_loss = 0.

    for i, data in enumerate(loader):
        if i % 50 == 0:
            print(f"Batch {i} of {len(loader)}")

        inp, _, _, _, _ = data # [bs, 176, 478, 3]

        if args.transformer:
            inputs = inp.reshape((-1, inp.shape[2], inp.shape[3])) # [bs', 478, 3]
        else:
            inputs = inp.reshape((inp.shape[0], inp.shape[1], -1)) # [bs, 176, 478 * 3]
            inputs = inputs.reshape((-1, inputs.shape[2])) # [bs * 176, 478 * 3]

        # shuffle the inputs
        indices = torch.randperm(inputs.shape[0])
        inputs = inputs[indices]
        inputs = inputs.to(device)

        if i == 0 and args.wandb:
            # sample random from inp
            rand_idx = np.random.randint(0, inp.shape[0])
            inp = inp[rand_idx] # [176, 478, 3]
            inp = inp.reshape((inp.shape[0], -1)) # [176, 478 * 3]
            with torch.no_grad():
                out, _, _ = model(inp.to(device))
            inp = inp.cpu().detach().numpy().reshape((inp.shape[0], 478, 3)) * 3 + 0.5
            out = out.cpu().detach().numpy().reshape((inp.shape[0], 478, 3)) * 3 + 0.5
            inp = visualize_landmarks(np.zeros((480, 480, 3)).astype(np.uint8), inp, invert=True)
            out = visualize_landmarks(np.zeros((480, 480, 3)).astype(np.uint8), out, invert=True)
            inp = np.array(inp)
            out = np.array(out)
            # concatenate the videos vertically
            viz = np.concatenate([inp, out], axis=2)
            viz = viz.transpose((0, 3, 1, 2))
            wandb.log({
                 "videos": wandb.Video(viz, fps=24, format="mp4")
            })

        optimizer.zero_grad()
        outputs, mu, log_var = model(inputs)
        loss, reconstruction_loss, kld_loss = loss_function(outputs, inputs, mu, log_var,
                                                            (args.enable_kld * args.batch_size * 176 * args.z_dim) / (args.den * 478 * 3 * len(loader.dataset)))

        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()
        running_recon_loss += reconstruction_loss.detach().item()
        running_kld_loss += kld_loss.detach().item()

    return running_loss / len(loader), running_recon_loss / len(loader), running_kld_loss / len(loader)

# %%
logging.info("Loading data...")
train_dataloader, _, _ = get_dataloaders(
    directory="./dataset/zoomin/processed-non-normalized-fps-adjusted", 
    csv_file_path="./dataset/zoomin_info.csv",
    test_split=args.test_split, 
    val_split=args.val_split, 
    batch_size=args.batch_size,
    train_workers=args.train_workers, 
    val_workers=args.val_workers, 
    test_workers=args.test_workers,
    random_rotation=args.random_rotation,
    dual_channel=False,
    keep_iris=True,
    mesh_order_shuffle=False,
    blendshapes=False,
    single_class=None,
)
logging.info("Data loaded.")

# %%
best_loss = 1000000000

#######
# Wandb
#######
if args.wandb:
    wandb.login()
    wandb.init(project="VAE", config=args, name="den_{}".format(args.den))
    wandb.watch(model, log_freq=100)
#######

for epoch in range(args.epochs):
    logging.info('EPOCH {}:'.format(epoch))
    avg_loss, avg_recon_loss, avg_kld_loss = train_one_epoch(train_dataloader)

    # log the loss
    if args.wandb:
        wandb.log({
            "train_loss": avg_loss,
            "train_recon_loss": avg_recon_loss,
            "train_kld_loss": avg_kld_loss
        })

    # save the entire model including state 
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = "./model/trained/beta_vae_{}_best_{}.pth".format(args.timestamp, args.den)
        torch.save(model, save_path)

# %%



