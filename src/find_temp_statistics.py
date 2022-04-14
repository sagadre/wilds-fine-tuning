import json
import numpy as np

train_data = {}
with open('/local/crv/sagadre/repos/wilds-fine-tuning/src/data/curr/train.json', 'r') as f:
    train_data = json.load(f)

temps = []

for p in train_data:
    temps.append(p['temp'])

m = np.mean(temps)
s = np.std(temps)

print((np.min(temps)-m)/s)
print((np.max(temps)-m)/s)