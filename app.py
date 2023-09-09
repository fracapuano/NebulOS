from search.ga import GeneticSearch
from commons.hw_nats_fast_interface import HW_NATS_FastInterface
from commons.utils import DEVICES, DATASETS
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

DAYS_7 = 604800

@st.cache_data(ttl=DAYS_7)  
def load_lookup_table():
    """Load recap table of NebulOS metrics and cache it.
    """
    df_nebuloss = pd.read_csv('data/df_nebuloss.csv').rename(columns = {'test_accuracy' : 'validation_accuracy'})
    
    return df_nebuloss

@st.cache_data(ttl=DAYS_7)  
def subset_dataframe(df_nebuloss, dataset):
    """Subset df_nebuloss based on the right dataset.
    """    
    return df_nebuloss[df_nebuloss['dataset'] == dataset]

@st.cache_data(ttl=DAYS_7)  
def compute_quantiles(df_nebuloss_dataset):
    """Turn the values of df_nebuloss (of a certain dataset) into the corresponding quantiles, computed along the columns
    """
    # compute quantiles
    quantiles = df_nebuloss_dataset.drop(columns = ['idx']).rank(pct = True)
    # re-attach the original indices
    quantiles['idx'] = df_nebuloss_dataset['idx']

    return quantiles

# Streamlit app
def main():
    # load the lookup table of NebulOS metrics
    df_nebuloss = load_lookup_table()
    # add a title
    st.sidebar.title("🚀 NebulOS 🌿")

    # dropdown menu for dataset selection
    dataset = st.sidebar.selectbox("Select Dataset", DATASETS)

    # dropdown menu for device selection
    device = st.sidebar.selectbox("Select Device", DEVICES)

    # slider for performance weight selection
    performance_weight = st.sidebar.slider(
        "Select trade-off between PERFORMANCE WEIGHT and HARDWARE WEIGHT.\nHigher values will give larger weight to the performance.", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    # hardware weight (complementary to performance weight)
    hardware_weight = 1.0 - performance_weight

    # subset the dataframe for the current daset and device
    df_nebuloss_dataset = subset_dataframe(df_nebuloss, dataset)

    # create a single figure
    fig = plt.figure(figsize=(15, 6))

    # SCATTER PLOT
    # plot the architecture space for this combination of device and dataset
    plt.subplot(1, 2, 1)
    plt.scatter(
        df_nebuloss_dataset.loc[df_nebuloss['dataset'] == dataset, f'{device}_energy'], 
        df_nebuloss_dataset.loc[df_nebuloss['dataset'] == dataset, 'validation_accuracy'], 
        c = 'lightgray'
    )
    # find the best architecture -> this will be plotted as a blue star on top of the other architectures
    best_arch_idx = 9930
    plt.scatter(df_nebuloss_dataset.loc[df_nebuloss_dataset['idx'] == best_arch_idx, f'{device}_energy'], 
                df_nebuloss_dataset.loc[df_nebuloss_dataset['idx'] == best_arch_idx, 'validation_accuracy'], 
                c = 'blue', marker = '*', label = 'Best TF-Architecture', s = 100)

    # trigger the search
    searchspace_interface = HW_NATS_FastInterface(device=device, dataset=dataset)
    search = GeneticSearch(
        searchspace=searchspace_interface, 
        fitness_weights=np.array([performance_weight, hardware_weight])
        )
    # this perform the actual architecture search
    results = search.solve()

    # plot the best architecture for this convex combination of weights -> this will be plotted as a red dot
    arch_idx = searchspace_interface.architecture_to_index["/".join(results[0].genotype)]
    plt.scatter(df_nebuloss_dataset.loc[df_nebuloss_dataset['idx'] == arch_idx, f'{device}_energy'], 
                df_nebuloss_dataset.loc[df_nebuloss_dataset['idx'] == arch_idx, 'validation_accuracy'], 
                c = 'red', label = 'NebulOS Architecture')
    plt.xlabel(f'{device.upper()} energy')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Validation Accuracy vs. {device.upper()} Energy - {dataset}')
    plt.legend()

    # RADAR CHART
    # find the quantiles dataframe
    quantiles = compute_quantiles(df_nebuloss_dataset)
    # specify the variables you wish to appear in the chart
    categories = ['flops', 'params', 'validation_accuracy', f'{device}_energy', f'{device}_latency']
    # before, we found the best architecture
    # now, we want to plot the corresponding row as a radar chart, where each value corresponds to the quantile of that column
    row_to_plot = quantiles.loc[quantiles['idx'] == arch_idx, categories]
    # number of variables
    N = len(categories)            
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # initialise the spider plot
    ax = plt.subplot(122, polar=True)
    
    # draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([.2, .4, .6, .8, 1], ["0.2", "0.4", "0.6", "0.8", "1"], color="grey", size=7)
    plt.ylim(0,1)
    
    # we are going to plot the first line of the data frame.
    # we need to repeat the first value to close the circular graph:
    values=row_to_plot.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label = 'NebulOS Architecture', c = 'red')
    # fill area
    ax.fill(angles, values, 'r', alpha=0.1)
    # add title
    plt.title(f'Metrics for best architecture - {dataset}x{device.upper()}')
    # plot the radar plot for the best architecture
    best_row_to_plot = quantiles.loc[quantiles['idx'] == best_arch_idx, categories]
    best_values=best_row_to_plot.values.flatten().tolist()
    best_values += best_values[:1]
    ax.plot(angles, best_values, linewidth=1, linestyle='solid', label = 'Best TF-Architecture', c = 'blue')
    # Fill area
    ax.fill(angles, best_values, 'b', alpha=0.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.1))
    st.pyplot(fig)

if __name__ == "__main__":
    main()