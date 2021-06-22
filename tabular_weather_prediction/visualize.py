import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def visualize_temperatures(df_list, df_names, plot_type='histogram'):
    '''
    Compare the distribution of temperature labels in datasets

    Supported visualization formats:
     - histogram
     - violin
    '''
    temperatures = np.concatenate([df['fact_temperature'] for df in df_list])
    names = np.concatenate([[name]*len(df) for name, df in zip(df_names, df_list)])

    df_to_plot = pd.DataFrame(data={'Temperature': temperatures, 'Dataset Name':names})

    if plot_type == 'histogram':
        sns.histplot(
            data=df_to_plot, x="Temperature", hue="Dataset Name",
            element="step", stat="density", common_norm=False)
    elif plot_type == 'violin':
        sns.violinplot(x='Temperature', y='Dataset Name',
                              data=df_to_plot, scale='width', palette='Set3')
    else:
        raise ValueError("Unsupported plot_type")
    
    plt.show()
    plt.clf()

