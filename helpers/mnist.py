
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.metrics import categorical_accuracy
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten


def display_model(encoder, autoencoder, x_test, repr_shape=(40, 25)):
    """Display input, reconstruction and representations."""
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display representations
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(encoded_imgs[i].reshape(repr_shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def get_r2(autoencoder, x_test, title, x_train=None, figname=None):
    if x_train is None:
        x_train = np.copy(x_test)
    x_test = x_test.reshape(-1, 784)
    x_test_means = np.mean(x_test, axis=0)
    x_pred = autoencoder.predict(x_train)
    x_pred = x_pred.reshape(-1, 784)
    ss_tot_mat = (x_test - x_test_means)**2
    ss_res_mat = (x_pred - x_test)**2
    ss_tot_vec = np.mean(ss_tot_mat, axis=0)
    ss_res_vec = np.mean(ss_res_mat, axis=0)
    r2_vec = 1 - np.divide(ss_res_vec, ss_tot_vec)
    r2_vec_pos = r2_vec[r2_vec > 0]
    r2_vec_posorzero = np.where(r2_vec > 0, r2_vec, 0)

    if r2_vec_pos.size == 0:
        pos_part = 0
    else:
        pos_part = np.divide(len(r2_vec_pos), len(r2_vec_posorzero))

    with plt.style.context('ggplot'):
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        fig.suptitle('{}: $R^2$ Positive Values'.format(title), fontsize=14)
        axs[0].set_title('Histogram', fontsize=12)
        axs[1].set_title('2D Distribution', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.subplots_adjust(wspace=-0.25)
        axs[0].hist(r2_vec_pos, bins=50, color=cm.get_cmap('viridis')(0.72), alpha=0.7)
        im = axs[1].imshow(r2_vec_posorzero.reshape(28, 28), norm=colors.Normalize(vmin=0, vmax=1))
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        axs[0].text(0.05, 0.80, '$\mu_{pos}$' + ' = {:.3f}'.format(np.mean(r2_vec_pos)) + \
                    '\n$N_{pos}/N_{tot}$' + ' = {:.3f}'.format(pos_part),
                    transform=axs[0].transAxes, bbox=props)
        axs[0].set_xlim(-0.03, 1)
        fig.colorbar(im)
        if figname:
            fig.savefig(figname, dpi=300)


def build_autoenc(encoding_dim=1000, input_shape=(784,),
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


def build_deep_autoenc(encoding_dim=10, input_shape=(784,), num_hid=0,
    hidden_dims=None, loss='mean_squared_error', sparsity=0):

    if hidden_dims is None:
        hidden_dims = np.linspace(encoding_dim, input_shape,
                                  num_hid + 2).astype(int)[::-1][1:]

    input_img = Input(shape=input_shape)

    prev_layer = input_img
    for hidden_dim in hidden_dims:
        layer = Dense(hidden_dim, activation='relu',
                      activity_regularizer=regularizers.l1(sparsity))(prev_layer)
        prev_layer = layer

    # this model maps an input to its encoded representation
    deep_encoder = Model(input_img, prev_layer)

    for hidden_dim in np.linspace(encoding_dim,
                                  input_shape, num_hid + 2).astype(int)[1:-1]:
        layer = Dense(hidden_dim, activation='relu',
                      activity_regularizer=regularizers.l1(sparsity))(prev_layer)
        prev_layer = layer

    decoded = Dense(input_shape[0], activation='sigmoid',
                      activity_regularizer=regularizers.l1(sparsity))(prev_layer)

    deep_autoencoder = Model(input_img, decoded)
    deep_autoencoder.compile(optimizer='adam', loss=loss)

    return deep_autoencoder, deep_encoder


def build_conv_autoenc():
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    conv_encoder = Model(input_img, encoded)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    conv_autoencoder = Model(input_img, decoded)

    conv_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return conv_autoencoder, conv_encoder


def build_denois_autoenc():
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    denois_encoder = Model(input_img, encoded)

    # at this point the representation is (7, 7, 32)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    denois_autoencoder = Model(input_img, decoded)
    denois_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return denois_autoencoder, denois_encoder


def build_readout(encoder=None, output_dim=10, num_hid=0, hidden_dim=1000,
    input_shape=(784,), sparsity=0, activation='sigmoid', optimizer='adam',
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


def create_error_data(x_data, missing_pixels):
    if missing_pixels == 0:
        return x_data
    else:
        x_data_copy = np.copy(x_data)
        for i in range(len(x_data_copy)):
            for j in np.random.choice(x_data_copy.shape[1],
                                      missing_pixels, False):
                x_data_copy[i, j] = 0.0
        x_data_copy = x_data_copy / np.max(x_data)
        return x_data_copy


def get_ro(ro, x_test, y_test_hot, max_pixels=15, num_rounds=1, step=1, conv=False):
    ls_mean, acs_mean = [], []
    if num_rounds > 1:
        ls_std, acs_std = [], []
    for i in range(0, max_pixels, step):
        if i % 100 == 0:
            print(i)
        ls_i, acs_i = [], []
        for j in range(num_rounds):
            data = create_error_data(x_test, missing_pixels=i)
            if conv:
                data = np.reshape(data, (len(data), 28, 28, 1))
            l, a = ro.evaluate(data, y_test_hot, batch_size=32, verbose=2)
            ls_i.append(l)
            acs_i.append(a)
        ls_mean.append(np.mean(ls_i))
        acs_mean.append(np.mean(acs_i))
        if num_rounds > 1:
            ls_std.append(np.std(ls_i))
            acs_std.append(np.std(acs_i))
    if num_rounds > 1:
        return ls_mean, acs_mean, ls_std, acs_std
    else:
        return ls_mean, acs_mean


def get_ro_repr(ro_repr, encoder, x_test, y_test_hot,
    max_pixels=15, num_rounds=1, step=1, conv=False):
    ls_mean, acs_mean = [], []
    if num_rounds > 1:
        ls_std, acs_std = [], []
    for i in range(0, max_pixels, step):
        ls_i, acs_i = [], []
        if i % 100 == 0:
            print(i)
        for j in range(num_rounds):
            data = create_error_data(x_test, missing_pixels=i)
            if conv:
                data = np.reshape(data, (len(data), 28, 28, 1))
            l, a = ro_repr.evaluate(encoder.predict(data),
                                    y_test_hot, batch_size=32, verbose=2)
            ls_i.append(l)
            acs_i.append(a)
        ls_mean.append(np.mean(ls_i))
        acs_mean.append(np.mean(acs_i))
        if num_rounds > 1:
            ls_std.append(np.std(ls_i))
            acs_std.append(np.std(acs_i))
    if num_rounds > 1:
        return ls_mean, acs_mean, ls_std, acs_std
    else:
        return ls_mean, acs_mean
