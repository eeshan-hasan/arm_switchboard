import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

colors=sns.color_palette("colorblind")

def make_full_plot(M):
    best_params=M.best_params
    full_data = M.model_data
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    full_data = full_data.set_index('trial_no')
    n_sub = len(full_data['subject_ID'].unique())
    plt.figure(figsize=[14,3*n_sub])

    i=1
    for subject_ID in full_data['subject_ID'].unique():
        subj_data= full_data[full_data['subject_ID']==subject_ID]
        task = subj_data['task_structure'].iloc[0]

        subj_data_1 = subj_data[subj_data['phase']==1]
        subj_data_2 = subj_data[subj_data['phase']==2]
        phase_change=subj_data_2.index[0]

        ax=plt.subplot(n_sub,4,i)
        subj_data_1['acc'].rolling(window=45,center=True,win_type='triang').mean().plot(ax=ax,c=colors[0],label='Data')
        subj_data_1['model_prob_correct'].rolling(window=45,center=True,win_type='triang').mean().plot(ax=ax,c=colors[1],ls='--',lw=2,label='Model')
        subj_data_2['acc'].rolling(window=45,center=True,win_type='triang').mean().plot(ax=ax,c=colors[0],label='_nolegend_')
        subj_data_2['model_prob_correct'].rolling(window=45,center=True,win_type='triang').mean().plot(ax=ax,c=colors[1],ls='--',lw=2,label='_nolegend_')
        plt.axvline(phase_change,ls=':',c='black',label='Phase Change')
        plt.title(f'Subject ID: {subject_ID} Task: {task}')
        plt.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(title=f'Subject {subject_ID}  {task}', xlabel='Trial', ylabel='Accuracy', ylim=(0, 1))
        plt.ylim([0,1])

        
        ax=plt.subplot(n_sub,4,i+1)
        plt.plot(subj_data_1[subj_data_1['trial_type']=='Transfer'][['model_prob_correct']].rolling(window=15,center=True,win_type='triang').mean(),ls='--',lw=2,c=colors[2])
        plt.plot(subj_data_1[subj_data_1['trial_type']=='Interference'][['model_prob_correct']].rolling(window=15,center=True,win_type='triang').mean(),ls='--',lw=2,c=colors[3])

        plt.plot(subj_data_1[subj_data_1['trial_type']=='Transfer'][['acc']].rolling(window=15,center=True,win_type='triang').mean('acc'),ls='-',lw=2,c=colors[2])
        plt.plot(subj_data_1[subj_data_1['trial_type']=='Interference'][['acc']].rolling(window=15,center=True,win_type='triang').mean('acc'),ls='-',lw=2,c=colors[3])
        
        plt.plot(subj_data_2[subj_data_2['trial_type']=='Transfer'][['model_prob_correct']].rolling(window=15,center=True,win_type='triang').mean(),ls='--',lw=2,c=colors[2])
        plt.plot(subj_data_2[subj_data_2['trial_type']=='Interference'][['model_prob_correct']].rolling(window=15,center=True,win_type='triang').mean(),ls='--',lw=2,c=colors[3])

        plt.plot(subj_data_2[subj_data_2['trial_type']=='Transfer'][['acc']].rolling(window=15,center=True,win_type='triang').mean('acc'),ls='-',lw=2,c=colors[2])
        plt.plot(subj_data_2[subj_data_2['trial_type']=='Interference'][['acc']].rolling(window=15,center=True,win_type='triang').mean('acc'),ls='-',lw=2,c=colors[3])

        plt.axvline(phase_change,ls=':',c='black')

        ax.spines[['top', 'right']].set_visible(False)
        ax.set(title=f'Subject {subject_ID}  {task}', xlabel='Trial', ylabel='Accuracy', ylim=(0, 1))
        plt.ylim([0,1])
        legend = [
            Line2D([0],[0], c=colors[2], lw=2, label='Transfer'),
            Line2D([0],[0], c=colors[3], lw=2, label='Interference'),
            Line2D([0],[0], c='black', lw=2, label='Data'),
            Line2D([0],[0], c='black', lw=2, ls='--', label='Model')
        ]
        ax.legend(handles=legend, frameon=False, ncol=2)

        
        ax=plt.subplot(n_sub,4,i+2)
        sigmoid(subj_data[['alpha_1','alpha_2']]).plot(ax=ax,color=[colors[3],colors[4]])
        plt.axvline(phase_change,ls=':',c='black')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(title=f'Subject {subject_ID}  {task}', xlabel='Trial', ylabel=r'Attention $\sigma(\alpha)$', ylim=(0, 1))
        plt.ylim([0,1])
        legend = [
            Line2D([0],[0], c=colors[3], lw=2, label='Alpha Dim. 1'),
            Line2D([0],[0], c=colors[4], lw=2, label='Alpha Dim. 2')
        ]
        ax.legend(handles=legend, frameon=False, ncol=1)


        ax = plt.subplot(n_sub, 4, i+3)

        params = best_params[subject_ID]
        lines = []
        for k, v in params.items():
            name = k.replace('_', ' ').title()
            v = np.asarray(v)
            value = f'{v.item():.2f}' if v.ndim == 0 else ', '.join(f'{x:.2f}' for x in v.ravel())
            lines.append(f'{name}:  {value}')

        ax.text(0, .95, 'Best-fit parameters', transform=ax.transAxes,
                ha='left', va='top', fontsize=11, fontweight='bold')

        ax.text(0, .82, '\n'.join(lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=10, linespacing=1.2)

        ax.axis('off')

        

        i+=4
        plt.tight_layout()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.signal.windows import triang

colors = sns.color_palette("colorblind")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_bootstrap_ci(ax, data, column='acc', window=45, color='black',
                      label=None, ls='-', lw=2, alpha=.15, n_boot=1000, seed=42):

    y = data[column].astype(float)
    mean = y.rolling(window=window, center=True, win_type='triang').mean()

    weights = triang(window)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)

    lower = np.full(len(y), np.nan)
    upper = np.full(len(y), np.nan)

    half = window // 2

    for j in range(half, len(y) - half):
        vals = y.iloc[j-half:j+half+1].to_numpy()

        if len(vals) != window or np.isnan(vals).any():
            continue

        samples = rng.choice(vals, size=(n_boot, window), replace=True, p=weights)
        boot_means = samples.mean(axis=1)

        lower[j], upper[j] = np.percentile(boot_means, [2.5, 97.5])

    ax.plot(data.index, mean, c=color, ls=ls, lw=lw, label=label)
    ax.fill_between(data.index, lower, upper, color=color, alpha=alpha, linewidth=0)


def plot_model(ax, data, window, color, label=None):
    mean = data['model_prob_correct'].rolling(window=window, center=True, win_type='triang').mean()
    ax.plot(data.index, mean, c=color, ls='--', lw=2, label=label)


def make_full_plot_ci(M):

    best_params = M.best_params
    full_data = M.model_data.copy().set_index('trial_no')

    subject_ids = full_data['subject_ID'].unique()
    n_sub = len(subject_ids)

    plt.figure(figsize=[20, 3*n_sub])

    i = 1

    for subject_ID in subject_ids:

        subj_data = full_data[full_data['subject_ID'] == subject_ID].copy()
        task = subj_data['task_structure'].iloc[0]

        subj_data_1 = subj_data[subj_data['phase'] == 1].copy()
        subj_data_2 = subj_data[subj_data['phase'] == 2].copy()
        phase_change = subj_data_2.index[0]

        # Overall accuracy
        ax = plt.subplot(n_sub, 4, i)

        plot_bootstrap_ci(ax, subj_data_1, 'acc', 45, colors[0], 'Data')
        plot_bootstrap_ci(ax, subj_data_2, 'acc', 45, colors[0], '_nolegend_')

        plot_model(ax, subj_data_1, 45, colors[1], 'Model')
        plot_model(ax, subj_data_2, 45, colors[1], '_nolegend_')

        ax.axvline(phase_change, ls=':', c='black', label='Phase Change')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(title=f'Subject {subject_ID}  {task}', xlabel='Trial',
               ylabel='Accuracy', ylim=(0, 1))
        ax.legend(frameon=False)


        # Transfer / Interference
        ax = plt.subplot(n_sub, 4, i+1)

        for phase_data in [subj_data_1, subj_data_2]:

            transfer = phase_data[phase_data['trial_type'] == 'Transfer']
            interference = phase_data[phase_data['trial_type'] == 'Interference']

            plot_bootstrap_ci(ax, transfer, 'acc', 15, colors[2])
            plot_bootstrap_ci(ax, interference, 'acc', 15, colors[3])

            plot_model(ax, transfer, 15, colors[2])
            plot_model(ax, interference, 15, colors[3])

        ax.axvline(phase_change, ls=':', c='black')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(title=f'Subject {subject_ID}  {task}', xlabel='Trial',
               ylabel='Accuracy', ylim=(0, 1))

        legend = [
            Line2D([0], [0], c=colors[2], lw=2, label='Transfer'),
            Line2D([0], [0], c=colors[3], lw=2, label='Interference'),
            Line2D([0], [0], c='black', lw=2, label='Data'),
            Line2D([0], [0], c='black', lw=2, ls='--', label='Model')
        ]

        ax.legend(handles=legend, frameon=False, ncol=2)


        # Attention
        ax = plt.subplot(n_sub, 4, i+2)

        sigmoid(subj_data[['alpha_1', 'alpha_2']]).plot(
            ax=ax, color=[colors[3], colors[4]]
        )

        ax.axvline(phase_change, ls=':', c='black')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(title=f'Subject {subject_ID}  {task}', xlabel='Trial',
               ylabel=r'Attention $\sigma(\alpha)$', ylim=(0, 1))

        legend = [
            Line2D([0], [0], c=colors[3], lw=2, label='Alpha Dim. 1'),
            Line2D([0], [0], c=colors[4], lw=2, label='Alpha Dim. 2')
        ]

        ax.legend(handles=legend, frameon=False)


        # Parameters
        ax = plt.subplot(n_sub, 4, i+3)

        lines = []

        for k, v in best_params[subject_ID].items():
            name = k.replace('_', ' ').title()
            v = np.asarray(v)
            value = f'{v.item():.2f}' if v.ndim == 0 else ', '.join(f'{x:.2f}' for x in v.ravel())
            lines.append(f'{name}:  {value}')

        ax.text(0, .95, 'Best-fit parameters', transform=ax.transAxes,
                ha='left', va='top', fontsize=11, fontweight='bold')

        ax.text(0, .82, '\n'.join(lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=10, linespacing=1.2)

        ax.axis('off')

        i += 4

    plt.tight_layout()