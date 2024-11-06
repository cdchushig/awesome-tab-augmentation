import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from datetime import datetime

def train_epoch(
        dataloader,
        synthesizer,
        diffuser,
        loss_fnc,
        optimizer,
        scheduler,
    ):
    """Training module for single epoch, update model parameters and return losses

    Args:
        dataloader (_type_): torch Dataloader
        synthesizer (_type_): model synthesizer
        diffuser (_type_): diffuser model
        loss_fnc (_type_): loss function
        optimizer (_type_): optimizer
        scheduler (_type_): learning rate scheduler

    Returns:
        dict: losses
    """
    total_losses = []

    # iterate over distinct mini-batches
    for batch_cat, batch_num, batch_y in dataloader:
        # set network in training mode
        synthesizer.train()

        # sample timestamps t
        timesteps = diffuser.sample_timesteps(n=batch_cat.shape[0])

        # get cat embeddings
        batch_cat_emb = synthesizer.embed_categorical(x_cat=batch_cat)
        # concat cat & num
        batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

        # add noise
        batch_noise_t, noise_t = diffuser.add_gauss_noise(x_num=batch_cat_num, t=timesteps)

        # conduct forward encoder/decoder pass
        predicted_noise = synthesizer(x=batch_noise_t, timesteps=timesteps, label=batch_y)

        # compute train loss
        train_losses = loss_fnc(
            input=noise_t,
            target=predicted_noise,
        )

        # reset encoder and decoder gradients
        optimizer.zero_grad()

        # run error back-propagation
        train_losses.backward()

        # optimize encoder and decoder parameters
        optimizer.step()

        # collect rec error losses
        total_losses.append(train_losses.detach().cpu().numpy())

    # average of rec errors
    total_losses_mean = np.mean(np.array(total_losses))

    # update learning rate according to the scheduler
    scheduler.step()

    return {'losses': total_losses_mean}

@torch.no_grad()
def generate_samples(
        synthesizer,
        diffuser,
        encoded_dim,
        last_diff_step,
        n_samples=None, 
        label=None
    ):
    """ Generation of samples. 
        For unconditional sampling use n_samples, for conditional sampling provide label.

    Args:
        synthesizer (_type_): synthesizer model
        diffuser (_type_): diffuzer model
        encoded_dim (int): transformed data dimension 
        last_diff_step (int): total number of diffusion steps
        n_samples (int, optional): number of samples to sample. Defaults to None.
        label (tensor, optional): list of labels for conditional sampling. Defaults to None.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    if (n_samples is None) and (label is None):
        raise Exception("either n_samples or label needs to be given")

    if label is not None:
        n_samples = len(label)
        

    z_norm = torch.randn((n_samples, encoded_dim))

    pbar = tqdm(iterable=reversed(range(0, last_diff_step)))
    for i in pbar:

        pbar.set_description(f"SAMPLING STEP: {i:4d}")

        t = torch.full((n_samples,), i, dtype=torch.long)

        model_out = synthesizer(z_norm.float(), t, label)

        z_norm = diffuser.p_sample_gauss(model_out, z_norm, t)

    return z_norm

def decode_sample(
        sample,
        cat_dim,
        n_cat_emb,
        num_attrs,
        cat_attrs,
        num_scaler,
        vocab_per_attr,
        label_encoder,
        synthesizer
    ):
    """ Decoding function for unscaling numeric attributes and inverse encoding of categorical attributes.
        Used once synthetic data is generated. 

    Args:
        sample (tensor): input samples for decoding
        cat_dim (int): categorical dimension
        n_cat_emb (int): size of categorical embeddings
        num_attrs (list): numeric attributes
        cat_attrs (list): categorical attributes
        num_scaler (_type_): numeric scaler from sklearn
        vocab_per_attr (dict): vocabulary of distinct values in attribute
        label_encoder (_type_): categorical encoder
        synthesizer (_type_): model synthesizer

    Returns:
        pandas DataFrame: decoded dataframe
    """

    # split sample into numeric and categorical parts
    sample = sample.cpu().numpy()
    sample_num = sample[:, cat_dim:]
    sample_cat = sample[:, :cat_dim]

    # denormalize numeric attributes
    z_norm_upscaled = num_scaler.inverse_transform(sample_num)
    z_norm_df = pd.DataFrame(z_norm_upscaled, columns=num_attrs)

    # get embedding lookup matrix
    embedding_lookup = synthesizer.get_embeddings().cpu() 
    # reshape back to batch_size * n_dim_cat * cat_emb_dim
    sample_cat = sample_cat.reshape(-1, len(cat_attrs), n_cat_emb)
    # compute pairwise distances
    distances = torch.cdist(x1=embedding_lookup, x2=torch.Tensor(sample_cat))
    # get the closest distance based on the embeddings that belong to a column category
    z_cat_df = pd.DataFrame(index=range(len(sample_cat)), columns=cat_attrs)
    nearest_dist_df = pd.DataFrame(index=range(len(sample_cat)), columns=cat_attrs)
    for attr_idx, attr_name in enumerate(cat_attrs):
        attr_emb_idx = list(vocab_per_attr[attr_name])
        attr_distances = distances[:, attr_emb_idx, attr_idx]
        # nearest_idx = torch.argmin(attr_distances, dim=1).cpu().numpy()
        nearest_values, nearest_idx = torch.min(attr_distances, dim=1)
        nearest_idx = nearest_idx.cpu().numpy()

        z_cat_df[attr_name] = np.array(attr_emb_idx)[nearest_idx]  # need to map emb indices back to column indices
        nearest_dist_df[attr_name] = nearest_values.cpu().numpy()

    z_cat_df = z_cat_df.apply(label_encoder.inverse_transform)
    sample_decoded = pd.concat([z_cat_df, z_norm_df], axis=1)

    return sample_decoded

def download_and_extract_data(url, extract_path='data'):
    """Downloads and extracts data from a given URL."""
    response = requests.get(url)
    zip_file = ZipFile(BytesIO(response.content))
    zip_file.extractall(extract_path)

def load_and_preprocess_data(data, cat_attrs, num_attrs, target_col):
    """Loads and preprocesses the data, including scaling and encoding."""

    data[cat_attrs] = data[cat_attrs].astype(str)
    label = data[target_col]
    train = data[[*cat_attrs, *num_attrs]]

    num_scaler = QuantileTransformer(output_distribution='normal')
    num_scaler.fit(train[num_attrs])
    train_num_scaled = num_scaler.transform(train[num_attrs])

    vocabulary_classes = np.unique(train[cat_attrs])
    label_encoder = LabelEncoder()
    label_encoder.fit(vocabulary_classes)
    train_cat_scaled = train[cat_attrs].apply(label_encoder.transform)
    vocab_per_attr = {cat_attr: set(train_cat_scaled[cat_attr]) for cat_attr in cat_attrs}

    return train_num_scaled, train_cat_scaled, label, num_scaler, vocab_per_attr, label_encoder

def create_dataloader(train_cat, train_num, label, batch_size):
    """Creates a DataLoader for the training data."""
    train_num_torch = torch.FloatTensor(train_num)
    train_cat_torch = torch.LongTensor(train_cat.values)
    label_torch = torch.LongTensor(label)
    train_set = TensorDataset(train_cat_torch, train_num_torch, label_torch)
    dataloader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    return dataloader, label_torch

def train_model(model, diffuser, dataloader, optimizer, scheduler, loss_fn, epochs, device):
    """Trains the synthesizer model."""
    train_epoch_losses = []
    model.train()
    model = model.to(device)
    pbar = tqdm(iterable=range(epochs), position=0, leave=True)
    for epoch in pbar:
        batch_losses = []
        for batch_cat, batch_num, batch_y in dataloader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            timesteps = diffuser.sample_timesteps(n=batch_cat.shape[0])
            batch_cat_emb = model.embed_categorical(x_cat=batch_cat)
            batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)
            batch_noise_t, noise_t = diffuser.add_gauss_noise(x_num=batch_cat_num, t=timesteps)                        
                                    
            predicted_noise = model(x=batch_noise_t, timesteps=timesteps, label=batch_y)
            batch_loss = loss_fn(input=noise_t, target=predicted_noise)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            batch_losses.append(batch_loss.detach().cpu().numpy())

        batch_losses_mean = np.mean(batch_losses)
        scheduler.step()
        train_epoch_losses.append(batch_losses_mean)
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        pbar.set_description(f'[LOG {now}] epoch: {epoch:04}, train-loss: {batch_losses_mean:.4f}')
    
    # Save the trained models
    mlp_path = f'{diffuser.save_dir}/mlp_model.pt'
    torch.save(model.state_dict(), mlp_path)
    diffuser.save_model()
    print(f"MLP model saved at {mlp_path}")     

    return train_epoch_losses

def plot_training_loss(train_epoch_losses):
    """Plots the training loss over epochs."""
    plt.figure()
    plt.grid(linestyle='dotted')
    plt.plot(np.array(range(1, len(train_epoch_losses) + 1)), train_epoch_losses, label='epoch loss (blue)')
    plt.xlabel('[Training Epoch $e_i$]', fontsize=10)
    plt.ylabel('[MSE Error $\mathcal{L}^{MSE}$]', fontsize=10)
    plt.legend(loc='upper right', numpoints=1, fancybox=True)
    plt.title('FinDiff Training Epochs $e_i$ vs. MSE Error $L^{MSE}$', fontsize=10)
    plt.show()

def generate_samples(model, diffuser, encoded_dim, label_torch, diffusion_steps, device):
    """Generates synthetic samples using the trained model and diffuser."""
    samples = torch.randn((len(label_torch), encoded_dim), device=device)
    pbar = tqdm(iterable=reversed(range(0, diffusion_steps)), position=0, leave=True)
    with torch.no_grad():
        for diffusion_step in pbar:
            now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            pbar.set_description(f'[LOG {now}] Diffusion Step: {diffusion_step:04}')
            timesteps = torch.full((len(label_torch),), diffusion_step, dtype=torch.long, device=device)
            model_out = model(x=samples.float(), timesteps=timesteps, label=label_torch.to(device))
            samples = diffuser.p_sample_gauss(model_out, samples, timesteps)
    return samples.detach().cpu().numpy()

def decode_samples(samples, cat_dim, num_scaler, vocab_per_attr, label_encoder, cat_attrs, num_attrs, synthesizer_model, cat_emb_dim):
    """Decodes generated samples back to their original format."""
    samples_num = samples[:, cat_dim:]
    samples_cat = samples[:, :cat_dim]
    z_norm_upscaled = num_scaler.inverse_transform(samples_num)
    z_norm_df = pd.DataFrame(z_norm_upscaled, columns=num_attrs)
    embedding_lookup = synthesizer_model.get_embeddings().cpu()
    samples_cat = samples_cat.reshape(-1, len(cat_attrs), cat_emb_dim)
    distances = torch.cdist(x1=embedding_lookup, x2=torch.Tensor(samples_cat))
    z_cat_df = pd.DataFrame(index=range(len(samples_cat)), columns=cat_attrs)
    for attr_idx, attr_name in enumerate(cat_attrs):
        attr_emb_idx = list(vocab_per_attr[attr_name])
        attr_distances = distances[:, attr_emb_idx, attr_idx]
        nearest_values, nearest_idx = torch.min(attr_distances, dim=1)
        z_cat_df[attr_name] = np.array(attr_emb_idx)[nearest_idx.cpu().numpy()]
    z_cat_df = z_cat_df.apply(label_encoder.inverse_transform)
    return pd.concat([z_cat_df, z_norm_df], axis=1)

def evaluate_samples(real_data, synthetic_data):
    """Evaluates the quality of the generated synthetic samples."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)
    quality_report = sdv_st.evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    fig = quality_report.get_visualization(property_name='Column Shapes')
    fig.show()
    fig = quality_report.get_visualization(property_name='Column Pair Trends')
    fig.show()
