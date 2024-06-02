# %%
import torch

# Note: pleace update this to data.presidentdata if using
# the president dataset
from data.talkshowdata import get_dataloaders
import logging
import wandb
from argparse import ArgumentParser
from utils.visualize_mediapipe import visualize_video_v3
from datetime import datetime
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

BETA_VAE_RELEVANT_INDICES = [0,1,2,4,5,6,7,8,9,12,13,15]


# =========================
# Argument parsing
# =========================

parser = ArgumentParser()
parser.add_argument('--csv_path', type=str, default="./dataset/zoomin_info.csv", 
                    help="Path to csv file (change for pres if needed)")
parser.add_argument('--data_path', type=str, default="./dataset/zoomin/processed/",
                    help="Path to data directory (change for pres if needed)")
parser.add_argument('--device', type=str, default="0", help="GPU device")
parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--val_split', type=float, default=0.1, help="Validation split ratio")
parser.add_argument('--test_split', type=float, default=0.1, help="Test split ratio")
parser.add_argument('--test_workers', type=int, default=1, help="Number of test workers")
parser.add_argument('--wandb', action='store_true', help="Use wandb for logging")
parser.add_argument('--dual_channel', type=bool, default=False, 
                    help="Use dual channel (not implemented, keep false)")
parser.add_argument('--mesh_order_shuffle', type=bool, default=False, 
                    help="Shuffle mesh order (not relevant for single channel)")
parser.add_argument('--temporal_width', type=int, default=176, help="Temporal width of input")
parser.add_argument('--z_dim', type=int, default=16, help="Dimension of latent space")
parser.add_argument('--chunks', type=int, default=2, 
                    help="Number of chunks to divide the temporal width into (c)")
parser.add_argument('--vae_path', type=str, default="./model/trained/beta_vae.pth")
parser.add_argument('--variable_alpha', action='store_true', help="Predicted translator when true")
parser.add_argument('--variable_changepoints', action='store_true', help="Var. chunks when true")
parser.add_argument('--timestamp', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
parser.add_argument('--alpha_n', type=int, default=32, help="p for fixed translator set")
parser.add_argument('--temporal_coeff', type=float, default=0.0)
parser.add_argument('--slope', type=float, default=8.3, help="Slope/temperature")
parser.add_argument('--person_id', type=str, default=None, help="Person ID to train on")

args, _ = parser.parse_known_args()
print(args)


# =========================
# Define model architecture
# =========================

class Generator(nn.Module):

    def __init__(self, z_dim, temporal_width, chunks=1, horse=False):
        super(Generator, self).__init__()
        self.chunks = chunks
        self.input_dim = z_dim * temporal_width
        self.z_dim = z_dim
        self.temporal_width = temporal_width

        # alphas
        if args.variable_alpha:
            self.a_linear1 = torch.nn.Linear(input_dim, input_dim // 8)
            self.a_linear2 = torch.nn.Linear(input_dim // 8, input_dim // 64)
            self.a_linear3 = torch.nn.Linear(input_dim // 64, z_dim * 2)
        else:
            self.a_linear_fixed = torch.nn.Linear(args.alpha_n * z_dim * 2, args.alpha_n * z_dim * 2)
            self.a_linear_choose1 = torch.nn.Linear(input_dim, input_dim // 8)
            self.a_linear_choose2 = torch.nn.Linear(input_dim // 8, input_dim // 64)
            self.a_linear_choose3 = torch.nn.Linear(input_dim // 64, args.alpha_n)
            
        # timestamps
        self.t_linear1 = torch.nn.Linear(input_dim, input_dim // 8)
        self.t_linear2 = torch.nn.Linear(input_dim // 8, input_dim // 64)
        self.t_linear3 = torch.nn.Linear(input_dim // 64, chunks - 1)

    
    def forward(self, x): # x : [bs, tw, z_dim]
        d = None
        t = None
        a = None

        if self.chunks > 1:

            if args.variable_changepoints:
                # Predict timestamps
                t = torch.flatten(x, start_dim=1) # [bs, tw * z_dim]
                t = self.t_linear1(t) # [bs, tw * z_dim // 8]
                t = torch.nn.functional.leaky_relu(t)
                t = self.t_linear2(t) # [bs, tw * z_dim // 64]
                t = torch.nn.functional.leaky_relu(t)
                t = self.t_linear3(t) # [bs, chunks - 1]
                t = bounded_output(t, lower=0.0, upper=self.temporal_width - 1) # [bs, chunks - 1]
                t = torch.sort(t, dim=1)[0] # [bs, chunks - 1]

            else:
                # fix the chunks-1 changepoints at equal intervals
                t = torch.linspace(self.temporal_width / self.chunks,
                                    (self.temporal_width / self.chunks) * (self.chunks - 1),
                                    self.chunks - 1).unsqueeze(0).repeat(x.shape[0], 1).to(device) # [bs, chunks - 1]

            indices = torch.linspace(0, args.temporal_width - 1, 
                                     args.temporal_width).unsqueeze(0).repeat(t.shape[0], 1).to(device) # [bs, tw]

            # till the mid of the prev timestamp it will follow prev sigmoid
            # after that it will follow next sigmoid
            w_ends = sigmoid(indices.unsqueeze(1), m=t.unsqueeze(2), k=-args.slope) # [bs, chunks-1, tw]
            w_ends = torch.cat((w_ends, torch.ones(w_ends.shape[0], 1, w_ends.shape[2]).to(device)), dim=1) # [bs, chunks, tw]

            w_starts = sigmoid(indices.unsqueeze(1), m=t.unsqueeze(2), k=args.slope) # [bs, chunks-1, tw]
            w_starts = torch.cat((torch.ones(w_starts.shape[0], 1, w_starts.shape[2]).to(device), w_starts), dim=1) # [bs, chunks, tw]

            w = torch.min(w_ends, w_starts) # [bs, chunks, tw]

            # normalize across chunks
            w = w / torch.sum(w, dim=1, keepdim=True) # [bs, chunks, tw]

            x = x.unsqueeze(1).repeat(1, self.chunks, 1, 1) # [bs, chunks, tw, z_dim]
            y = x + 10.0 # [bs, chunks, tw, z_dim]
            y = y * w.unsqueeze(3) # [bs, chunks, tw, z_dim]
            x = x * w.unsqueeze(3) # [bs, chunks, tw, z_dim]

            # We directly predict the alphas of each chunk
            if args.variable_alpha:
                a = torch.flatten(y, start_dim=2) # [bs, chunks, tw * z_dim]
                a = self.a_linear1(a) # [bs, chunks, tw * z_dim // 8]
                a = torch.nn.functional.leaky_relu(a)
                a = self.a_linear2(a) # [bs, chunks, tw * z_dim // 64]
                a = torch.nn.functional.leaky_relu(a)
                a = self.a_linear3(a) # [bs, chunks, z_dim * 2]
                
            # We let each chunk choose from a fixed set of alphas
            else:
                a = torch.ones(x.shape[0], args.alpha_n * self.z_dim * 2).to(device) # [bs, alpha_n * z_dim * 2]
                a = self.a_linear_fixed(a) # [bs, alpha_n * z_dim * 2]
                a = a.view(a.shape[0], args.alpha_n, self.z_dim * 2) # [bs, alpha_n, z_dim * 2]

                d = torch.flatten(y, start_dim=2) # [bs, chunks, tw * z_dim]
                d = self.a_linear_choose1(d) # [bs, chunks, tw * z_dim // 8]
                d = torch.nn.functional.leaky_relu(d)
                d = self.a_linear_choose2(d) # [bs, chunks, tw * z_dim // 64]
                d = torch.nn.functional.leaky_relu(d)
                d = self.a_linear_choose3(d) # [bs, chunks, alpha_n]

                # use d as weights for a
                a = a.unsqueeze(1).repeat(1, self.chunks, 1, 1) # [bs, chunks, alpha_n, z_dim * 2]
                a = a * torch.nn.functional.softmax(d, dim=2).unsqueeze(3) # [bs, chunks, alpha_n, z_dim * 2]
                # sum over alpha_n
                a = torch.sum(a, dim=2) # [bs, chunks, z_dim * 2]

            a_spread = a.unsqueeze(2).repeat(1, 1, self.temporal_width, 1) # [bs, chunks, tw, z_dim * 2]
            a_spread = a_spread.view(a.shape[0], self.chunks, self.temporal_width, self.z_dim, -1) # [bs, chunks, tw, z_dim, 2]

            # multiply x with first half of a_spread
            x = x * a_spread[:, :, :, :, 0] # [bs, chunks, tw, z_dim]
            # add second half of a_spread
            x = x + a_spread[:, :, :, :, 1] # [bs, chunks, tw, z_dim]
            # sum over chunks
            x = torch.sum(x, dim=1) # [bs, tw, z_dim]
            
        else:
            if args.variable_alpha:
                # predict a_spread
                a = torch.flatten(x, start_dim=1) # [bs, tw * z_dim]
                a = self.a_linear1(a) # [bs, tw * z_dim // 8]
                a = torch.nn.functional.leaky_relu(a)
                a = self.a_linear2(a) # [bs, tw * z_dim // 64]
                a = torch.nn.functional.leaky_relu(a)
                a = self.a_linear3(a) # [bs, z_dim * 2]

            # We let each chunk choose from a fixed set of alphas
            else:
                a = torch.ones(x.shape[0], args.alpha_n * self.z_dim * 2).to(device) # [bs, alpha_n * z_dim * 2]
                a = self.a_linear_fixed(a) # [bs, alpha_n * z_dim * 2]
                a = a.view(a.shape[0], args.alpha_n, self.z_dim * 2) # [bs, alpha_n, z_dim * 2]

                d = torch.flatten(x, start_dim=1) # [bs, tw * z_dim]
                d = self.a_linear_choose1(d) # [bs, tw * z_dim // 8]
                d = torch.nn.functional.leaky_relu(d)
                d = self.a_linear_choose2(d) # [bs, tw * z_dim // 64]
                d = torch.nn.functional.leaky_relu(d)
                d = self.a_linear_choose3(d) # [bs, alpha_n]

                # use d as weights for a
                a = a * torch.nn.functional.softmax(d, dim=1).unsqueeze(2) # [bs, alpha_n, z_dim * 2]
                # sum over alpha_n
                a = torch.sum(a, dim=1) # [bs, z_dim * 2]

            
            a_spread = a.unsqueeze(1).repeat(1, self.temporal_width, 1) # [bs, tw, z_dim * 2]
            a_spread = a_spread.view(a.shape[0], self.temporal_width, self.z_dim, -1) # [bs, tw, z_dim, 2]

            # multiply z with first half of a_spread
            x = x * a_spread[:, :, :, 0]
            # add second half of a_spread
            x = x + a_spread[:, :, :, 1]

        return x, a, t, d
    

class Discriminator(nn.Module):

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim // 8)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(input_dim // 8, input_dim // 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.linear3 = torch.nn.Linear(input_dim // 64, 1)
        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x
    

def sigmoid(x, k=1.0, m=0.0):
    f = k * (x - m)
    return torch.sigmoid(f)


def bounded_output(x, lower, upper):
    scale = upper - lower
    return scale * torch.sigmoid(x) + lower

# this is the entropy loss
class ELoss(nn.Module):
    def __init__(self):
        super(ELoss, self).__init__()

    def forward(self, x):
        b = - F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum() / x.shape[0]
        return b


def data_massager(data):
    # Reshape to create temporal chunks
    data = data.reshape((data.shape[0] * (176 // args.temporal_width), args.temporal_width, -1)) # [bs', tw, 1434]
    data = data[torch.randperm(data.size()[0])] # [bs', tw, 1434]
    data = data.reshape((-1, 1434)) # [bs'', 1434]
    data = data.to(device)

    with torch.no_grad():
        data = vae_model.reparameterize(*vae_model.encode(data))

    return data.reshape((-1, args.temporal_width, args.z_dim)) # [bs', tw, z_dim]

# =========================
# Create models
# =========================

logging.info("Loading model...")
device = torch.device("cuda:" + args.device)
input_dim = args.temporal_width * args.z_dim

# Create generators

if args.dual_channel:
    gen_H_left = Generator(input_dim=input_dim).to(device)
    gen_H_right = Generator(input_dim=input_dim).to(device)
    gen_Z_left = Generator(input_dim=input_dim).to(device)
    gen_Z_right = Generator(input_dim=input_dim).to(device)

    opt_gen = torch.optim.Adam(
        list(gen_Z_left.parameters()) + list(gen_Z_right.parameters()) + \
            list(gen_H_left.parameters()) + list(gen_H_right.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
else:
    print("creating generators")
    gen_H = Generator(z_dim=args.z_dim, temporal_width=args.temporal_width, 
                      chunks=args.chunks, horse=True).to(device)
    gen_Z = Generator(z_dim=args.z_dim, temporal_width=args.temporal_width, 
                      chunks=args.chunks, horse=False).to(device)

    opt_gen = torch.optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

# Create discriminators

print("creating discriminators")
disc_H = Discriminator(input_dim=input_dim * (2 if args.dual_channel else 1)).to(device)
disc_Z = Discriminator(input_dim=input_dim * (2 if args.dual_channel else 1)).to(device)

opt_disc = torch.optim.Adam(
    list(disc_H.parameters()) + list(disc_Z.parameters()),
    lr=args.lr,
    betas=(0.5, 0.999),
)

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
entropy_loss = ELoss()

# Load VAE model

vae_model = torch.load(args.vae_path)
vae_model.to(device)
vae_model.eval()

logging.info("Models loaded to device.")

# %%
# =========================
# Load data
# =========================

# Zebra is 0 and ONline (VC)
# Horse is 1 and OFFline (F2F)

logging.info("Loading data...")

train_dataloader_0, test_dataloader_0, val_dataloader_0 = get_dataloaders(
    directory=args.data_path, 
    csv_file_path=args.csv_path, 
    test_split=args.test_split, 
    val_split=args.val_split, 
    batch_size=args.batch_size,
    train_workers=8, 
    val_workers=8, 
    test_workers=8,
    random_rotation=False,
    dual_channel=args.dual_channel,
    keep_iris=True,
    mesh_order_shuffle=args.mesh_order_shuffle,
    blendshapes=False,
    single_class=0,
    person_id=args.person_id,
)

train_dataloader_1, test_dataloader_1, val_dataloader_1 = get_dataloaders(
    directory=args.data_path, 
    csv_file_path=args.csv_path,
    test_split=args.test_split, 
    val_split=args.val_split, 
    batch_size=args.batch_size,
    train_workers=8, 
    val_workers=8, 
    test_workers=8,
    random_rotation=False,
    dual_channel=args.dual_channel,
    keep_iris=True,
    mesh_order_shuffle=args.mesh_order_shuffle,
    blendshapes=False,
    single_class=1,
    person_id=args.person_id,
)

logging.info("Data loaded.")

# %%
def visualize(fake_horse_t, horse_i_reconstructed, fake_horse_i_reconstructed,
              fake_zebra_t, zebra_i_reconstructed, fake_zebra_i_reconstructed):
    
    true_horse_video, true_zebra_video = visualize_video_v3(zebra_i_reconstructed, fake_horse_i_reconstructed,
                                                            horse_i_reconstructed, fake_zebra_i_reconstructed,
                                                            fake_horse_t, fake_zebra_t,
                                                            args.dual_channel, args.temporal_width)
    wandb.log({
        "[Val] true OFFline, fake ONline": wandb.Video(true_horse_video, fps=25, format="mp4"),
        "[Val] true ONline, fake OFFline": wandb.Video(true_zebra_video, fps=25, format="mp4"),
    })
    

def one_epoch(dataloader_0, dataloader_1, train = True, epoch = 0):
    D_loss_total = 0.
    G_loss_total = 0.
    # cycle_loss_total = 0.
    idx = 0

    total_correct = 0
    total_den = 0
    horse_correct = 0
    horse_den = 0
    zebra_correct = 0
    zebra_den = 0
    fake_horse_ts = []
    fake_zebra_ts = []

    for idx, (zebra, horse) in enumerate(zip(dataloader_0, dataloader_1)):
        if not args.dual_channel:
            zebra, _, _, _, _ = zebra
            horse, _, _, _, _ = horse
            zebra = data_massager(zebra) # [bs', tw, z_dim]
            horse = data_massager(horse) # [bs', tw, z_dim]
        else:
            zebra_left, zebra_right, _, _, _ = zebra
            horse_left, horse_right, _, _, _ = horse
            zebra_left = data_massager(zebra_left)
            zebra_right = data_massager(zebra_right)
            horse_left = data_massager(horse_left)
            horse_right = data_massager(horse_right)
            zebra = torch.cat((zebra_left, zebra_right), dim=1)
            horse = torch.cat((horse_left, horse_right), dim=1)


        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            if not args.dual_channel:
                fake_horse, _, fake_horse_t, fake_horse_d = gen_H(zebra)
                fake_zebra, _, fake_zebra_t, fake_zebra_d = gen_Z(horse)

            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # calculate accuracies
            horse_correct += (D_H_real > 0.5).sum().item() + (D_H_fake < 0.5).sum().item()
            zebra_correct += (D_Z_real > 0.5).sum().item() + (D_Z_fake < 0.5).sum().item()
            total_correct += horse_correct + zebra_correct

            horse_den += D_H_real.shape[0] + D_H_fake.shape[0]
            zebra_den += D_Z_real.shape[0] + D_Z_fake.shape[0]
            total_den += horse_den + zebra_den

            D_loss = (D_H_loss + D_Z_loss) / 2
            D_loss_total += D_loss.item()

        if train:
            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # entropy loss
            if not args.variable_alpha:
                loss_G_H += entropy_loss(fake_horse_d.reshape(-1, args.alpha_n)) * args.temporal_coeff
                loss_G_Z += entropy_loss(fake_zebra_d.reshape(-1, args.alpha_n)) * args.temporal_coeff

            # temporal gap loss
            if args.chunks > 1:
                # add 0s to start and args.temporal_width - 1 to end
                fake_horse_gaps = torch.cat((torch.zeros(fake_horse_t.shape[0], 1)
                                             .to(device), fake_horse_t), dim=1) # [bs, chunks]
                fake_horse_gaps = torch.cat((fake_horse_gaps, 
                                             (args.temporal_width - 1) * torch.ones(fake_horse_t.shape[0], 1)
                                             .to(device)), dim=1) # [bs, chunks + 1]
                fake_zebra_gaps = torch.cat((torch.zeros(fake_zebra_t.shape[0], 1)
                                             .to(device), fake_zebra_t), dim=1) # [bs, chunks]
                fake_zebra_gaps = torch.cat((fake_zebra_gaps, 
                                             (args.temporal_width - 1) * torch.ones(fake_zebra_t.shape[0], 1)
                                             .to(device)), dim=1) # [bs, chunks + 1]

                # calculate gaps between consecutive timestamps
                fake_horse_gaps = fake_horse_gaps[:, 1:] - fake_horse_gaps[:, :-1] # [bs, chunks]
                fake_zebra_gaps = fake_zebra_gaps[:, 1:] - fake_zebra_gaps[:, :-1] # [bs, chunks]

                # maximize the entropy
                loss_G_H += -entropy_loss(fake_horse_gaps) * args.temporal_coeff
                loss_G_Z += -entropy_loss(fake_zebra_gaps) * args.temporal_coeff


            # add all togethor
            G_loss = (
                loss_G_H
                + loss_G_Z
            )
            G_loss_total += G_loss.item()

        if train:
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

        if train and args.wandb and idx == 0:
            wandb.log({
                "[Train] D_loss": D_loss.item(), 
                "[Train] G_loss": G_loss.item(),
                "[Train] Accuracy": total_correct / total_den,
                "[Train] Accuracy (F2F)": horse_correct / horse_den,
                "[Train] Accuracy (VC)": zebra_correct / zebra_den,
            })
            total_correct = 0
            total_den = 0
            horse_correct = 0
            horse_den = 0
            zebra_correct = 0
            zebra_den = 0

        if fake_horse_t is not None:
            fake_horse_t = fake_horse_t.cpu().detach().numpy()
            fake_zebra_t = fake_zebra_t.cpu().detach().numpy()
        else:
            fake_horse_t = np.zeros((args.batch_size, args.chunks - 1))
            fake_zebra_t = np.zeros((args.batch_size, args.chunks - 1))

        if not train and args.wandb:
            fake_horse_ts.append(fake_horse_t.reshape(-1))
            fake_zebra_ts.append(fake_zebra_t.reshape(-1))

        if not train and args.wandb and idx == 0:
            random_idx = torch.randint(0, args.batch_size, (1,)).item()
            zebra_i = zebra[random_idx].unsqueeze(0) # [1, tw, z_dim]
            horse_i = horse[random_idx].unsqueeze(0) # [1, tw, z_dim]
            fake_horse_i = fake_horse[random_idx].unsqueeze(0) # [1, tw, z_dim]
            fake_zebra_i = fake_zebra[random_idx].unsqueeze(0) # [1, tw, z_dim]
            fake_horse_t = fake_horse_t[random_idx] # [chunks - 1]
            fake_zebra_t = fake_zebra_t[random_idx] # [chunks - 1]
            fake_horse_t = [int(x) for x in fake_horse_t]
            fake_zebra_t = [int(x) for x in fake_zebra_t]

            with torch.no_grad():    
                zebra_i_reconstructed = vae_model.decode(zebra_i)[0] # [tw, 1434]
                horse_i_reconstructed = vae_model.decode(horse_i)[0] # [tw, 1434]
                fake_horse_i_reconstructed = vae_model.decode(fake_horse_i)[0] # [tw, 1434]
                fake_zebra_i_reconstructed = vae_model.decode(fake_zebra_i)[0] # [tw, 1434]
                
            visualize(fake_horse_t, horse_i_reconstructed, fake_horse_i_reconstructed,
                      fake_zebra_t, zebra_i_reconstructed, fake_zebra_i_reconstructed)

    if not train and args.wandb:
        fake_horse_ts = np.concatenate(fake_horse_ts)
        fake_zebra_ts = np.concatenate(fake_zebra_ts)
        to_log = {
            # "[Val] fake_horse_t": wandb.Histogram(fake_horse_ts, num_bins=args.temporal_width),
            # "[Val] fake_zebra_t": wandb.Histogram(fake_zebra_ts, num_bins=args.temporal_width),
            "[Val] D_loss": D_loss_total / (idx + 1), 
            "[Val] G_loss": G_loss_total / (idx + 1),
            "[Val] Accuracy": total_correct / total_den,
            "[Val] Accuracy (F2F)": horse_correct / horse_den,
            "[Val] Accuracy (VC)": zebra_correct / zebra_den,
        }
        wandb.log(to_log)
    
    return total_correct / total_den, horse_correct / horse_den, zebra_correct / zebra_den

#######
# Wandb
#######
if args.wandb:
    wandb.login()
    wandb.init(project="FacET", config=args)
#######
    
accs = []
h_accs = []
z_accs = []

for epoch in range(args.epochs):
    
    logging.info('EPOCH {}:'.format(epoch + 1))
    
    # Train loop
    if args.dual_channel:
        gen_H_left.train()
        gen_H_right.train()
        gen_Z_left.train()
        gen_Z_right.train()
    else:
        gen_H.train()
        gen_Z.train()
    disc_H.train()
    disc_Z.train()
    one_epoch(train_dataloader_0, train_dataloader_1, train=True, epoch=epoch)

    # Eval loop
    if args.dual_channel:
        gen_H_left.eval()
        gen_H_right.eval()
        gen_Z_left.eval()
        gen_Z_right.eval()
    else:
        gen_H.eval()
        gen_Z.eval()
    disc_H.eval()
    disc_Z.eval()
    with torch.no_grad():
        acc, h_acc, z_acc = one_epoch(val_dataloader_0, val_dataloader_1, train=False, epoch=epoch)
    accs.append(acc)
    h_accs.append(h_acc)
    z_accs.append(z_acc)

    if args.wandb:
        wandb.log({
            "[Val] Avg accuracy last 100 epochs": np.mean(accs[-100:]),
            "[Val] Avg accuracy (F2F) last 100 epochs": np.mean(h_accs[-100:]),
            "[Val] Avg accuracy (VC) last 100 epochs": np.mean(z_accs[-100:]),
        })

    # Save entire models
    model_path = './model/trained/model_cyclegan_vae_{}'.format(args.timestamp)
    if args.wandb and (epoch % 100 == 0 or epoch == args.epochs - 1):
        if args.dual_channel:
            torch.save(gen_H_left, model_path + '_gen_H_left.pt')
            torch.save(gen_H_right, model_path + '_gen_H_right.pt')
            torch.save(gen_Z_left, model_path + '_gen_Z_left.pt')
            torch.save(gen_Z_right, model_path + '_gen_Z_right.pt')
        else:
            torch.save(gen_H, model_path + '_gen_H.pt')
            torch.save(gen_Z, model_path + '_gen_Z.pt')
        torch.save(disc_H, model_path + '_disc_H.pt')
        torch.save(disc_Z, model_path + '_disc_Z.pt')

