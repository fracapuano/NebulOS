from src.search.ga import GeneticSearch
from src.hw_nats_fast_interface import HW_NATS_FastInterface
from src.utils import DEVICES, DATASETS
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import OrderedDict
st.set_page_config(layout="wide")

TIME_TO_SCORE_EACH_ARCHITECTURE=0.15
DAYS_7 = 604800
NEBULOS_COLOR = '#FF6961'
TF_COLOR = '#A7C7E7'

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
    # mapping the devices pseudo-symbols to actual names
    device_mapping_dict = {
        "edgegpu": "NVIDIA Jetson nano",
        "eyeriss": "Eyeriss",
        "fpga": "FPGA",
    }

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

    # best architecture index
    best_arch_idx = 9930

    # Trigger the search and plot NebulOS Architecture
    searchspace_interface = HW_NATS_FastInterface(device=device, dataset=dataset)
    search = GeneticSearch(
        searchspace=searchspace_interface,
        fitness_weights=np.array([performance_weight, hardware_weight])
    )

    results = search.solve(return_trajectory=True)

    arch_idx = searchspace_interface.architecture_to_index["/".join(results[0].genotype)]

    # Create scatter plot
    scatter_trace1 = go.Scatter(
        x=df_nebuloss_dataset.loc[df_nebuloss['dataset'] == dataset, f'{device}_energy'],
        y=df_nebuloss_dataset.loc[df_nebuloss['dataset'] == dataset, 'validation_accuracy'],
        mode='markers',
        marker=dict(color='#D3D3D3', size=5),
        name='Architectures in the search space'
    )

    # Scatter plot for best architecture
    scatter_trace2 = go.Scatter(
        x=df_nebuloss_dataset.loc[df_nebuloss_dataset['idx'] == best_arch_idx, f'{device}_energy'],
        y=df_nebuloss_dataset.loc[df_nebuloss_dataset['idx'] == best_arch_idx, 'validation_accuracy'],
        mode='markers',
        marker=dict(color=TF_COLOR, symbol='circle-dot', size=12),
        name='Best TF-Architecture'
    )

    scatter_trace3 = go.Scatter(
        x=df_nebuloss_dataset.loc[df_nebuloss_dataset['idx'] == arch_idx, f'{device}_energy'],
        y=df_nebuloss_dataset.loc[df_nebuloss_dataset['idx'] == arch_idx, 'validation_accuracy'],
        mode='markers',
        marker=dict(color=NEBULOS_COLOR, symbol='circle-dot', size=12),
        name='NebulOS Architecture'
    )
    scatter_layout = go.Layout(
        title=f'Validation Accuracy vs. {device_mapping_dict[device]} Energy Consumption',
        xaxis=dict(title=f'{device.upper()} Energy'),
        yaxis=dict(title='Validation Accuracy'),
        showlegend=True
    )
    scatter_fig = go.Figure(data=[scatter_trace1, scatter_trace2, scatter_trace3], layout=scatter_layout)

    # Extracting quantile values
    metrics_considered = OrderedDict()
    # these are the metrics that we want to plot
    metrics_considered["flops"] = "FLOPS", 
    metrics_considered["params"] = "Num. Params", 
    metrics_considered["validation_accuracy"] = "Accuracy",
    metrics_considered[f"{device}_energy"] = f"{device_mapping_dict[device]} - Energy Consumption",
    metrics_considered[f"{device}_latency"] = f"{device_mapping_dict[device]} - Latency"


    # this retrieves the optimal row
    best_row_to_plot = df_nebuloss_dataset.loc[
        df_nebuloss_dataset['idx'] == best_arch_idx, 
        list(metrics_considered.keys())
    ].values

    # this retrieves the row that has been found by the NAS search
    row_to_plot = df_nebuloss_dataset.loc[
        df_nebuloss_dataset['idx'] == arch_idx, 
        list(metrics_considered.keys())
    ].values

    row_to_plot = row_to_plot/best_row_to_plot
    best_row_to_plot = best_row_to_plot/best_row_to_plot

    best_row_to_plot = best_row_to_plot.flatten().tolist()
    row_to_plot = row_to_plot.flatten().tolist()

    # Bar chart for NebulOS Architecture
    bar_trace1 = go.Bar(
        x=list(metrics_considered.keys()),
        y=row_to_plot,
        name='NebulOS Architecture',
        marker=dict(color=NEBULOS_COLOR)
    )
    # Bar chart for Best TF-Architecture
    bar_trace2 = go.Bar(
        x=list(metrics_considered.keys()),
        y=best_row_to_plot,
        name='Best TF-Architecture Found',
        marker=dict(color=TF_COLOR)
    )
    # Layout configuration
    bar_layout = go.Layout(
        title=f'Hardware-Agnostic Architecture (blue) vs. NebulOS (red)',
        yaxis=dict(title="(%)Hardware-Agnostic Architecture Value"),
        barmode='group'
    )

    # Combining traces with the layout
    bar_fig = go.Figure(data=[bar_trace2, bar_trace1], layout=bar_layout)

    # Create two columns in Streamlit to show data near each other
    col1, col2 = st.columns(2)

    # Display scatter plot in the first column
    with col1:
        st.plotly_chart(scatter_fig)

    # Display bar chart in the second column
    with col2:
        st.plotly_chart(bar_fig)

    best_architecture = df_nebuloss_dataset.loc[
        df_nebuloss_dataset['idx'] == best_arch_idx, 
        list(metrics_considered.keys())
    ]

    best_architecture_string = searchspace_interface[best_arch_idx]["architecture_string"]

    found_architecture = df_nebuloss_dataset.loc[
        df_nebuloss_dataset['idx'] == arch_idx, 
        list(metrics_considered.keys())
    ]

    message = \
    f"""
        <h4>NebulOS Search Process: Outcome</h4>
        <p>
        This search took ~{results[-1]*TIME_TO_SCORE_EACH_ARCHITECTURE} seconds (scoring {results[-1]} architectures using ~{TIME_TO_SCORE_EACH_ARCHITECTURE} seconds each)
        </p>
        The architecture found for <b>{device_mapping_dict[device]}</b> is: <b>{searchspace_interface[arch_idx]["architecture_string"]}</b><br>
        The optimal (hardware-agnostic) architecture in the searchspace is <b>{best_architecture_string}</b>
        </p>
        <p>
        You can find the recap, in terms of the percentage of the Training-Free metric found in the table to your right 👉
        </p>
    """

    # Sample data - replace these with your actual ratio values
    data = {
        "Metric": ["FLOPS", "Number of Parameters", "Validation Accuracy", "Energy Consumption", "Latency"],
        "NebulOS vs. Hardware Agnostic Network": ["{:.2g}%".format(val) for val in row_to_plot]
    }
    
    col1, _, col2 = st.columns([2,1,2])
    recap_df = pd.DataFrame(data).sort_values(by="Metric").set_index("Metric")
    
    with col1:
        st.write(message, unsafe_allow_html=True)
    
    with col2:
        st.dataframe(recap_df)

if __name__ == "__main__":
    main()