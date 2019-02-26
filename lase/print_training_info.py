import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    directory = "../experiments/mini-email_EU/" \
                "agg-sage_combine-concat_lr-0.0100_dr-0.000_eta-0.002_sample-uniform_intv-1_d0-064_d1-064_d2-048_d3-032_s1-025_s2-025_s3-000" \
                "/"
    filename = directory + "training_info.json"
    with open(filename, 'r') as fp:
        info = eval(fp.read())
        t = np.array(list(info['train_losses'].keys()))
        train_losses = np.array(list(info['train_losses'].values()))
        train_f1_mics = np.array(list(info['train_f1_mics'].values()))
        val_losses = np.array(list(info['val_losses'].values()))
        val_f1_mics = np.array(list(info['val_f1_mics'].values()))

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, train_losses, t, val_losses)
    axs[1].plot(t, train_f1_mics, t, val_f1_mics)

    plt.savefig(directory + 'training_info2.png', format='png', dpi=100)
