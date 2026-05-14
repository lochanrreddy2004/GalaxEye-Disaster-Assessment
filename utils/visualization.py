import matplotlib.pyplot as plt


def visualize(pre, post, gt, pred, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(pre)
    axes[0].set_title('Pre')

    axes[1].imshow(post)
    axes[1].set_title('Post')

    axes[2].imshow(gt, cmap='gray')
    axes[2].set_title('Ground Truth')

    axes[3].imshow(pred, cmap='gray')
    axes[3].set_title('Prediction')

    for ax in axes:
        ax.axis('off')

    if save_path:
        plt.savefig(save_path)

    plt.close()