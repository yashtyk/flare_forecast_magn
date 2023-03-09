import numpy as np
import yaml
y = './model1/metrics2.yaml'

with open("./model1/metrics2.yaml", "r") as stream:

        k1 = yaml.safe_load(stream)

with open("./model3/metrics2.yaml", "r") as stream:
    k2 = yaml.safe_load(stream)

with open("./model6/metrics2.yaml", "r") as stream:
    k3 = yaml.safe_load(stream)

mean_tss = np.mean((k1['tss'], k2['tss'], k3['tss']))
mean_hss = np.mean((k1['hss'], k2['hss'], k3['hss']))
mean_acc = np.mean((k1['accuracy'], k2['accuracy'], k3['accuracy']))
mean_pr = np.mean((k1['precision'], k2['precision'], k3['precision']))
mean_rec = np.mean((k1['recall'], k2['recall'], k3['recall']))

std_tss = np.std((k1['tss'], k2['tss'], k3['tss']))
std_hss = np.std((k1['hss'], k2['hss'], k3['hss']))
std_acc = np.std((k1['accuracy'], k2['accuracy'], k3['accuracy']))
std_pr = np.std((k1['precision'], k2['precision'], k3['precision']))
std_rec = np.std((k1['recall'], k2['recall'], k3['recall']))

print('tss: {} ({}), hss: {} ({}), acc: {} ({}), pr: {} ({}), rec: {} ({}) '.format(mean_tss, std_tss, mean_hss, std_hss, mean_acc, std_acc, mean_pr, std_pr, mean_rec, std_rec))






