import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import tss, hss, write_yaml
import yaml

def test(test_dataloader, model,  criterion, device, model_path):
    model.eval()
    test_loss = 0.0
    y_pred = []
    y_true=[]

    for image, target in test_dataloader:
        image, target = image.to(device), target.to(device)
        y = model(image)
        cur_y_pred = np.argmax(y.detach().numpy(), axis=1)

        y_pred = list(y_pred) + list(cur_y_pred)
        y_true = list(y_true) + list(target.detach().numpy())
        loss = criterion(y, target)
        test_loss += loss.detach().cpu().item() / len(test_dataloader)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tss_metric = tss(tp, fp, tn, fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    hss_metric = hss(tp, fp, tn, fn)

    print("tss: {:.4f}".format(tss_metric))

    metrics = {"tss": float(tss_metric), "hss": float(hss_metric), "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall)}

    with open(model_path / 'metrics2.yaml', 'w') as outfile:
        yaml.dump(metrics, outfile, default_flow_style=False)

    print('test_loss: {}'.format(test_loss))


