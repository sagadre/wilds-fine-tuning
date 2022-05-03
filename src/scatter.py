import matplotlib.pyplot as plt
import matplotlib.font_manager
import os
import json
import numpy as np
import matplotlib as mpl
import random

mpl.rcParams['figure.dpi'] = 300

plt.rcParams.update({
    "text.usetex": True,
})

result_path = '/local/crv/sagadre/repos/wilds-fine-tuning/src/results'

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(12.5, 4.5)

grid_min = 200
grid_max = 325

# axes[0].grid()
# axes[0].set_title('Train')
# axes[0].set_xlim(left=grid_min, right=grid_max)
# axes[0].set_ylim(bottom=grid_min, top=grid_max)
# axes[0].set_xlabel("Pred Tempreture (K)")
# axes[0].set_ylabel("GT Tempreature (K)")

space_time = None
space = None
time = None

density_view = True

with open(f'{result_path}/space_time.json', 'r') as f:
    space_time = json.load(f)

with open(f'{result_path}/space.json', 'r') as f:
    space = json.load(f)

with open(f'{result_path}/time.json', 'r') as f:
    time = json.load(f)

space_time_temps = [e['gt_temp'] for e in space_time]
space_temps = [e['gt_temp'] for e in space]
time_temps = [e['gt_temp'] for e in time]

# fixed bin size
# bins = np.arange(200, 310, 2) # fixed bin size

# plt.xlim([200, 310])

# a = plt.hist(space_temps, bins=bins, alpha=0.5)
# print(a)
# plt.title('Random Gaussian data (fixed bin size)')
# plt.xlabel('variable X (bin size = 5)')
# plt.ylabel('count')
# plt.savefig('tmp3.png')
# exit(0)

results = [space_time, space, time]

ticks = np.arange(200, 325, 25)

extrapolation_time = 0.9411764740943909
interpolation_time = 0.4117647111415863

axes[0].grid()
axes[0].set_title('Space and Time Unseen')
axes[0].set_xlim(left=grid_min, right=grid_max)
axes[0].set_ylim(bottom=grid_min, top=grid_max)
axes[0].set_ylabel("Pred Tempreture (K)")
axes[0].set_xlabel("GT Tempreature (K)")
axes[0].set_xticks(ticks)
axes[0].set_yticks(ticks)

axes[1].grid()
axes[1].set_title('Space Unseen')
axes[1].set_xlim(left=grid_min, right=grid_max)
axes[1].set_ylim(bottom=grid_min, top=grid_max)
axes[1].set_ylabel("Pred Tempreture (K)")
axes[1].set_xlabel("GT Tempreature (K)")
axes[1].set_xticks(ticks)
axes[1].set_yticks(ticks)

axes[2].grid()
axes[2].set_title('Time Unseen')
axes[2].set_xlim(left=grid_min, right=grid_max)
axes[2].set_ylim(bottom=grid_min, top=grid_max)
axes[2].set_ylabel("Pred Tempreture (K)")
axes[2].set_xlabel("GT Tempreature (K)")
axes[2].set_xticks(ticks)
axes[2].set_yticks(ticks)

colors = ['tab:red', 'tab:blue', 'tab:green']

interpolation_color = 'tab:orange'
extrapolation_color = 'tab:purple'
other_color = 'tab:grey'

lims = [
    np.min([axes[0].get_xlim(), axes[0].get_ylim()]),  # min of both axes
    np.max([axes[0].get_xlim(), axes[0].get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
axes[0].plot(lims, lims, '--', alpha=0.75, zorder=-1, color='black', label='$y=x$')
axes[1].plot(lims, lims, '--', alpha=0.75, zorder=-1, color='black', label='$y=x$')
axes[2].plot(lims, lims, '--', alpha=0.75, zorder=-1, color='black', label='$y=x$')

for i in range(3):
    random.shuffle(results[i])
    c = 0
    for dp in results[i][:1000]:
        color = colors[i]

        if not density_view:
            if dp['time'] == interpolation_time:
                color = interpolation_color
                # print('yooooo')
            elif dp['time'] == extrapolation_time:
                color = extrapolation_color
                # continue
            else:
                color = other_color
                # continue

        axes[i].scatter(dp['gt_temp'], dp['pred_temp'], color=color, edgecolors=(0, 0, 0, 1))#, label=name, color=color, marker=marker, s=80, edgecolors=(0, 0, 0, 1), zorder=10)

plt.savefig('tmp.png')