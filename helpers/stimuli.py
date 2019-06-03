import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras import regularizers
from keras.metrics import categorical_accuracy


with open('data/input/times_ids.pkl', 'rb') as f:
    times_ids = pkl.load(f)
with open('data/input/batches_200.pkl', 'rb') as f:
    batches_200 = pkl.load(f)
with open('data/input/GID_to_num.pkl', 'rb') as f:
    GID_to_num = pkl.load(f)
with open('data/input/xz_array.pkl', 'rb') as f:
    xz_array = pkl.load(f)


def create_data(times_ids=times_ids, batches_200=batches_200,
    GID_to_num=GID_to_num, missing_blobs=0):

    x_data = []
    for batch in batches_200:
        if missing_blobs != 0:
            unique_blobs = [list(x) for x in set(tuple(x)
                for x in times_ids[batch])]
            choice = list(np.random.choice(unique_blobs,
                len(unique_blobs) - missing_blobs, False))
            chosen = [x for x in times_ids[batch] if x in choice]
            batch_gids = np.concatenate(chosen)
        else:
            batch_gids = np.concatenate(times_ids[batch])
        batch_gid_nums = [GID_to_num[key] for key in batch_gids]
        data = np.bincount(batch_gid_nums)
        data = np.concatenate((data, np.zeros(len(
            GID_to_num.keys()) - len(data))))
        x_data.append(data)
    x_data = np.array(x_data)
    x_data = x_data / np.max(x_data)
    return x_data


def get_train_test(x_data, train_ratio=0.8):
    div_ind = int(len(x_data) * train_ratio)
    x_train = x_data[:div_ind]
    x_test = x_data[div_ind:]
    return x_train, x_test


def plot(data, titles=None, fig_title="", plot_name="title.png",
    show=False, save=True, ticksize=None,
    scale_to_01=True, xz_array=xz_array):

    if titles is None:
        titles = [""] * len(data)

    """Plots valued scatter plots."""
    fig, axs = plt.subplots(1, len(data), sharey=True, 
        figsize=(4 * len(data), 4))
    if len(data) == 1:
        axs = [axs]

    if scale_to_01:
        norm = colors.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.get_cmap('plasma')
    rgba = cmap(0)

    fig.suptitle(fig_title)

    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set(adjustable='box-forced', aspect='equal')
        ax.set_facecolor(rgba)

    images = [axs[i].scatter(xz_array[:, 1], xz_array[:, 2],
        c=np.zeros(xz_array.shape[0]), s=3, cmap='plasma')
        for i in range(len(axs))]

    cbar = fig.colorbar(images[0], ax=list(axs), orientation='vertical',
        fraction=.02)
    if ticksize is not None:
        cbar.ax.tick_params(labelsize=ticksize)

    if len(data[0].shape) == 1:
        for i, im in enumerate(images):
            if not scale_to_01:
                norm = colors.Normalize(vmin=np.min(data[i]),
                    vmax=np.max(data[i]))
            im.set_norm(norm)
            im.set_array(data[i].reshape(-1))

        if save:
            fig.savefig('data/incite/{}'.format(plot_name), dpi=300)
    else:
        def update(frame):
            for i, im in enumerate(images):
                if not scale_to_01:
                    norm = colors.Normalize(vmin=np.min(data[i]),
                        vmax=np.max(data[i]))
                    im.set_norm(norm)
                im.set_array(data[i][frame])
            return images

        ani = animation.FuncAnimation(fig, update, frames=data[0].shape[0],
                                      interval=1000, blit=True)
        if save:
            ani.save("data/incite/{}.mp4".format(plot_name), dpi=300)

    if show:
        plt.show()

    plt.close('all')


def build_autoenc(encoding_dim=128, input_shape=(2170,),
    optimizer='adam', loss='mean_squared_error',
    activation='sigmoid', sparsity=0):

    encoding_dim = encoding_dim

    input_img = Input(shape=input_shape)
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(sparsity))(input_img)
    decoded = Dense(input_shape[0], activation=activation)(encoded)

    # Connect autoencoder
    autoencoder = Model(input_img, decoded)
    # Connect encoder
    encoder = Model(input_img, encoded)

    # Connect decoder
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Compile autoencoder
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder, encoder, decoder


def build_readout(encoder=None, output_dim=8, num_hid=0, hidden_dim=1000,
    input_shape=(2170,), sparsity=0, activation='sigmoid', optimizer='adam',
    loss='categorical_crossentropy', metrics=[categorical_accuracy],
    conv=False):

    if encoder:
        input_repr = encoder.layers[-1].get_output_at(0)
    else:
        input_repr = Input(shape=input_shape)

    if conv:
        input_repr_flat = Flatten()(input_repr)
        prev_layer = input_repr_flat
    else:
        prev_layer = input_repr

    hidden_layers = [prev_layer]
    for i in range(num_hid):
        layer = Dense(hidden_dim, activation='relu',
            activity_regularizer=regularizers.l1(sparsity))(prev_layer)
        hidden_layers.append(layer)
        prev_layer = layer

    output_layer = Dense(output_dim, activation=activation)(hidden_layers[-1])

    if encoder:
        readout = Model(encoder.layers[0].get_output_at(0), output_layer)
    else:
        readout = Model(input_repr, output_layer)

    readout.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return readout


def get_ro(ro, y_test_hot, max_pixels=15, num_rounds=5, step=1):
    ls_mean, acs_mean = [], []
    ls_std, acs_std = [], []
    for i in range(0, max_pixels, step):
        if i % 100 == 0:
            print(i)
        ls_i, acs_i = [], []
        for j in range(num_rounds):
            _, data = get_train_test(create_data(missing_blobs=i))
            l, a = ro.evaluate(data, y_test_hot, batch_size=32, verbose=2)
            ls_i.append(l)
            acs_i.append(a)
        ls_mean.append(np.mean(ls_i))
        acs_mean.append(np.mean(acs_i))
        ls_std.append(np.std(ls_i))
        acs_std.append(np.std(acs_i))
    return ls_mean, acs_mean, ls_std, acs_std


def get_ro_repr(ro_repr, encoder, y_test_hot,
    max_pixels=15, num_rounds=5, step=1):
    ls_mean, acs_mean = [], []
    ls_std, acs_std = [], []
    for i in range(0, max_pixels, step):
        ls_i, acs_i = [], []
        for j in range(num_rounds):
            _, data = get_train_test(create_data(missing_blobs=i))
            #ro_repr.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy])
            l, a = ro_repr.evaluate(encoder.predict(data),
                                    y_test_hot, batch_size=1, verbose=2)
            ls_i.append(l)
            acs_i.append(a)
        ls_mean.append(np.mean(ls_i))
        acs_mean.append(np.mean(acs_i))
        ls_std.append(np.std(ls_i))
        acs_std.append(np.std(acs_i))
    return ls_mean, acs_mean, ls_std, acs_std
