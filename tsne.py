import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
def plot_tsne(model, dataloader ):
    fig = plt.figure()
    dr = 1
    dc = 1
    ii = 0
    resnet = model.resnet







    Data = []
    Labels = []
    count = 0
    for img, target in dataloader:

        representation = resnet(img)
        if count == 0:
            Data = representation
            Labels = target
        else:
            Data = torch.cat((Data, representation), 0)
            Labels = torch.cat((Labels, target))

        count+=1




        # === TSNE visualisation ==========================================================================================

    Labels = np.asarray(Labels)
    Data = np.asarray(Data.detach().numpy())

    labels = np.unique(Labels)
    print(f"Data.shape: {Data.shape}, \t Labels.shape: {Labels.shape}, \t unique labels: {labels}")

    tsne_model = TSNE(n_components=2, random_state=0)
    Data_embeded = tsne_model.fit_transform(Data)

    ii =1
    plt.subplot(dr, dc, ii)
    for i in labels:
        plt.plot(Data_embeded[Labels == i, 0], Data_embeded[Labels == i, 1], ".", markersize=5)

    plt.legend(['non-flare', 'flare'])
    plt.tick_params(axis='both', labelsize=12)
    plt.title("t-SNE")
    plt.grid()

    plt.savefig('./tsne.pdf'  )