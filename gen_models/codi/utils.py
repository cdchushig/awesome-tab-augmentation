import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def warmup_lr(step, warmup_steps=2000):
    return min(step, warmup_steps) / warmup_steps  # Linear increase during warm-up

def infiniteloop(dataloader):
    while True:
        for _, y in enumerate(dataloader):
            yield y

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'sigmoid':
            ed = st + item[0]
            data_t.append(data[:,st:ed])
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.softmax(data[:, st:ed]))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)

def log_sample_categorical(logits, num_classes):
    full_sample = []
    k=0
    for i in range(len(num_classes)):
        logits_column = logits[:,k:num_classes[i]+k]
        k+=num_classes[i]
        uniform = torch.rand_like(logits_column)
        gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30)
        sample = (gumbel_noise + logits_column).argmax(dim=1)
        col_t =np.zeros(logits_column.shape)
        col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1
        full_sample.append(col_t)
    full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
    log_sample = torch.log(full_sample.float().clamp(min=1e-30))
    return log_sample


def sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, trans, FLAGS):
    x_t_con = x_T_con
    x_t_dis = log_x_T_dis

    for time_step in reversed(range(FLAGS.T)):
        t = x_t_con.new_ones([x_t_con.shape[0], ], dtype=torch.long) * time_step
        mean, log_var = net_sampler.p_mean_variance(x_t=x_t_con, t=t, cond = x_t_dis.to(x_t_con.device), trans=trans)
        if time_step > 0:
            noise = torch.randn_like(x_t_con)
        elif time_step == 0:
            noise = 0
        x_t_minus_1_con = mean + torch.exp(0.5 * log_var) * noise
        x_t_minus_1_con = torch.clip(x_t_minus_1_con, -1., 1.)
        x_t_minus_1_dis = trainer_dis.p_sample(x_t_dis, t, x_t_con)
        x_t_con = x_t_minus_1_con
        x_t_dis = x_t_minus_1_dis

    return x_t_con, x_t_dis

def training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, categories, FLAGS):
    
    t = torch.randint(FLAGS.T, size=(x_0_con.shape[0], ), device=x_0_con.device)
    pt = torch.ones_like(t).float() / FLAGS.T

    #co-evolving training and predict positive samples
    noise = torch.randn_like(x_0_con)
    x_t_con = trainer.make_x_t(x_0_con, t, noise)
    log_x_start = torch.log(x_0_dis.float().clamp(min=1e-30))
    x_t_dis = trainer_dis.q_sample(log_x_start=log_x_start, t=t)
    eps = trainer.model(x_t_con, t, x_t_dis.to(x_t_con.device))
    ps_0_con = trainer.predict_xstart_from_eps(x_t_con, t, eps=eps)
    con_loss = F.mse_loss(eps, noise, reduction='none')
    con_loss = con_loss.mean()
    kl, ps_0_dis = trainer_dis.compute_Lt(log_x_start, x_t_dis, t, x_t_con)
    ps_0_dis = torch.exp(ps_0_dis)
    kl_prior = trainer_dis.kl_prior(log_x_start)
    dis_loss = (kl / pt + kl_prior).mean()

    # negative condition -> predict negative samples
    noise_ns = torch.randn_like(ns_con)
    ns_t_con = trainer.make_x_t(ns_con, t, noise_ns)
    log_ns_start = torch.log(ns_dis.float().clamp(min=1e-30))
    ns_t_dis = trainer_dis.q_sample(log_x_start=log_ns_start, t=t)
    eps_ns = trainer.model(x_t_con, t, ns_t_dis.to(ns_t_dis.device))
    ns_0_con = trainer.predict_xstart_from_eps(x_t_con, t, eps=eps_ns)
    _, ns_0_dis = trainer_dis.compute_Lt(log_x_start, x_t_dis, t, ns_t_con)
    ns_0_dis = torch.exp(ns_0_dis)
    
    # contrastive learning loss
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    triplet_con = triplet_loss(x_0_con, ps_0_con, ns_0_con)
    st=0
    triplet_dis = []
    for item in categories:
        ed = st + item
        ps_dis = F.cross_entropy(ps_0_dis[:, st:ed], torch.argmax(x_0_dis[:, st:ed], dim=-1).long(), reduction='none')
        ns_dis = F.cross_entropy(ns_0_dis[:, st:ed], torch.argmax(x_0_dis[:, st:ed], dim=-1).long(), reduction='none')

        triplet_dis.append(max((ps_dis-ns_dis).mean()+1,0))
        st = ed
    triplet_dis = sum(triplet_dis)/len(triplet_dis)
    return con_loss, triplet_con, dis_loss, triplet_dis

def make_negative_condition(x_0_con, x_0_dis):

    device = x_0_con.device
    x_0_con = x_0_con.detach().cpu().numpy()
    x_0_dis = x_0_dis.detach().cpu().numpy()

    nsc_raw = pd.DataFrame(x_0_con)
    nsd_raw = pd.DataFrame(x_0_dis)
    nsc = np.array(nsc_raw.sample(frac=1, replace = False).reset_index(drop=True))
    nsd = np.array(nsd_raw.sample(frac=1, replace = False).reset_index(drop=True))
    ns_con = nsc
    ns_dis = nsd

    return torch.tensor(ns_con).to(device), torch.tensor(ns_dis).to(device)


def train_model(model_con, model_dis, datalooper_train_con, datalooper_train_dis, trainer, trainer_dis, optim_con, optim_dis, 
                sched_con, sched_dis, device, args, ckpt_dir, categories, train, early_stopping_patience=500):
    
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))    
    best_loss = float('inf')
    
    # Original calculations for steps based on the dataset size
    total_steps_per_epoch = int(train.shape[0] / args.training_batch_size + 1)
    total_steps_both = args.total_epochs_both * total_steps_per_epoch
    print("Total steps: %d" % total_steps_both)
    print("Sample steps: %d" % (args.sample_step * total_steps_per_epoch))
    print(f"Total epochs: {args.total_epochs_both}")
    print(f"Total steps per epoch: {total_steps_per_epoch}")
    
    con_lr_track, dis_lr_track = args.lr_con, args.lr_dis
    patience = 0
    start_time = time.time()

    print(f"Training on device: {device}")
    
    for epoch in range(args.total_epochs_both):
        model_con.train()
        model_dis.train()

        epoch_con_loss = 0.0
        epoch_dis_loss = 0.0

        pbar = tqdm(range(total_steps_per_epoch), desc=f"Epoch {epoch + 1}/{args.total_epochs_both}", leave=False)
        for step in pbar:
            x_0_con = next(datalooper_train_con).to(device).float()
            x_0_dis = next(datalooper_train_dis).to(device)

            ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
            con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(
                x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, categories, args
            )

            # Calculate total loss
            loss_con = con_loss + args.lambda_con * con_loss_ns
            loss_dis = dis_loss + args.lambda_dis * dis_loss_ns
            epoch_con_loss += loss_con.item()
            epoch_dis_loss += loss_dis.item()

            # Optimizer steps for continuous model
            optim_con.zero_grad()
            loss_con.backward()
            torch.nn.utils.clip_grad_norm_(model_con.parameters(), args.grad_clip)
            optim_con.step()
            sched_con.step()

            # Optimizer steps for discrete model
            optim_dis.zero_grad()
            loss_dis.backward()
            torch.nn.utils.clip_grad_value_(trainer_dis.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), args.grad_clip)
            optim_dis.step()
            sched_dis.step()
            
            pbar.set_postfix({"Continuous Loss": loss_con.item(), "Discrete Loss": loss_dis.item()})
        
        # Calculate and log average losses
        avg_con_loss = epoch_con_loss / total_steps_per_epoch
        avg_dis_loss = epoch_dis_loss / total_steps_per_epoch
        writer.add_scalar('Loss/Continuous', avg_con_loss, epoch)
        writer.add_scalar('Loss/Discrete', avg_dis_loss, epoch)

        print(f"Epoch {epoch + 1}/{args.total_epochs_both} | Avg Continuous Loss: {avg_con_loss:.3f} | Avg Discrete Loss: {avg_dis_loss:.3f}")


        if epoch % 10 == 0:
            print(f"Continuous LR: {optim_con.param_groups[0]['lr']:.6f} | Discrete LR: {optim_dis.param_groups[0]['lr']:.6f}")

        # Check if this is the best loss and save model
        total_loss = avg_con_loss + avg_dis_loss
        if total_loss < best_loss:
            best_loss = total_loss
            patience = 0
            torch.save(model_con.state_dict(), f'{ckpt_dir}/model_con.pt')
            torch.save(model_dis.state_dict(), f'{ckpt_dir}/model_dis.pt')
            print(f"Model saved to: {ckpt_dir}")
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best loss: {best_loss:.3f}")
                torch.save(model_con.state_dict(), f'{ckpt_dir}/model_con.pt')
                torch.save(model_dis.state_dict(), f'{ckpt_dir}/model_dis.pt')
                print(f"Model saved to: {ckpt_dir}")
                break

        # Save checkpoints every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            torch.save(model_con.state_dict(), os.path.join(ckpt_dir, f'model_con_{epoch + 1}.pt'))
            torch.save(model_dis.state_dict(), os.path.join(ckpt_dir, f'model_dis_{epoch + 1}.pt'))

    writer.close()
    print('Training completed. Total time:', time.time() - start_time)
