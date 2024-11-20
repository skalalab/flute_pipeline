# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:58:44 2024

@author: Wenxuan Zhao

ui for the pipeline 
"""
from pipeline import Pipeline
import flute_pipeline_visualizer as visualizer

import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path

@st.cache_data
def show_intensity(sdt_filename):
    fig, sdt_data = visualizer.visualize_sdt(sdt_filename)
    return fig, sdt_data

@st.cache_data
def show_mask(mask_filename, title="mask", label=-1):
    fig = visualizer.visualize_tiff(mask_filename, title, label)
    return fig


@st.cache_data
def calculate_phaser_coords(image, _pipeline):
    coords = []
    for cell in image["cells"]:
        coords.append(_pipeline.get_cell_phasor(cell, image["IRF_decay"]))
    return coords

def fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)  # Reset the buffer's pointer to the start
    return buf

def show_intermediate_steps(sdt_filename, mask_filename, irf_filename, shifted_irf):
    
    st.write("Reading in the sdt and mask")

    # display intensity image and cell image side by side 
    fig_intensity, sdt_data = show_intensity(sdt_filename)
    buf_intensity = fig_to_buf(fig_intensity)
    # show all cells 
    fig_mask = show_mask(mask_filename, label=-1)
    buf_mask = fig_to_buf(fig_mask)
     # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.image(buf_intensity, use_container_width=False, width=400)

    with col2:
        st.image(buf_mask, use_container_width=False, width=400)
            
    st.write("Reading in the irf and shift it")
    with open(irf_filename) as irf:
        irf_values = [int(line) for line in irf if line.strip()]
    fig = visualizer.plot_irfs_against_sdt(np.array(irf_values), shifted_irf, sdt_data)
    st.pyplot(fig)

def show_coord(coords, image):
    df = pd.DataFrame(columns=["Cell Label", "G coordinate", "S coordinate"])

    for i, cell_label in enumerate(image["values"]):   
        cell_coord = coords[i]
        df.loc[len(df)] = {"Cell Label": cell_label, "G coordinate": cell_coord[0], "S coordinate": cell_coord[1]}
    st.dataframe(df) 

def phasor_plot(coords, image, callback, sdt_filename, mask_filename):
    f = 0.080 # laser repetition rate in [GHz]
    title = "Interactive Phasor Plot"

    # Create the figure
    fig = go.Figure()

    # Set axis limits
    fig.update_layout(
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 0.55]), 
        showlegend=False # Hide the legend
    )

    # Plot the curve
    u = np.arange(0, 100, 0.01)
    x_curve = 1 / (1 + u**2)
    y_curve = u / (1 + u**2)

    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        line=dict(color='black'),
        name='Curve', 
        hoverinfo='skip' # Hide the hover info for this trace
    ))

    # Calculate and plot specific points
    wt = 2 * np.pi * f * np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    x_points = 1 / (1 + wt**2)
    y_points = wt / (1 + wt**2)

    fig.add_trace(go.Scatter(
        x=x_points,
        y=y_points,
        mode='markers',
        marker=dict(size=8, color='black'),
        name='Lifetime Markers', 
        hoverinfo='skip' # Hide the hover info for this trace
    ))

    # Annotate the points
    lifetime_labels = ['0.5 ns', '1 ns', '2 ns', '3 ns', '4 ns', '5 ns']
    labels = len(lifetime_labels)
    label_coords = list(zip(x_points - 0.02, y_points + 0.03))[:labels]

    for i in range(labels):
        fig.add_annotation(
            x=label_coords[i][0],
            y=label_coords[i][1],
            text=lifetime_labels[i],
            showarrow=False,
            font=dict(size=10),
            xanchor='left'
        )

    # Add titles and axis labels
    fig.update_layout(
        title=title,
        xaxis_title='g',
        yaxis_title='s',
        font=dict(size=15),
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15, family='Arial', color='black')),
        yaxis=dict(title_font=dict(size=15, family='Arial', color='black'))
    )

    # Add text inside the plot
    fig.add_annotation(
        x=0.8,
        y=0.5,
        text=f"{f * 1000} MHz",
        showarrow=False,
        font=dict(size=15, color='black'),
        xanchor='left'
    )

    alpha = 1
    colors = ['blue']  # Extend colors if more classes

    
    g = [point[0] for point in coords]
    s = [point[1] for point in coords]
    labels = image["values"]

    fig.add_trace(go.Scatter(
        x=g,
        y=s,
        customdata=labels,
        mode='markers',
        marker=dict(size=5, color=colors[0 % len(colors)]),
        opacity=alpha,
        hovertemplate='<br>g: %{x:.2f}<br>s: %{y:.2f}<br>cell label: %{customdata}<extra></extra>',
    ))

    # Display the figure and capture click events
    selected_points = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=600,
        override_width='100%'
    )

    # Process clicked points
    if selected_points:
        point = selected_points[0]
        cell_label = labels[point["pointIndex"]]
        cell_label = int(cell_label)
        phasor_plot_callback(cell_label, sdt_filename, mask_filename)
    else:
        st.write("Click on a single cell coordinate to display the intensity and mask image for that cell.")

def phasor_plot_callback(cell_label, sdt_filename, mask_filename):
    # display intensity image and cell image side by side 
    fig_intensity, _ = show_intensity(sdt_filename)
    buf_intensity = fig_to_buf(fig_intensity)
    st.write(f"Cell label: {cell_label}")
    fig_mask = show_mask(mask_filename, label=cell_label)
    buf_mask = fig_to_buf(fig_mask)
     # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.image(buf_intensity, use_container_width=False, width=400, caption="Intensity")

    with col2:
        st.image(buf_mask, use_container_width=False, width=400, caption="Mask")

    return

def main():
    st.set_page_config(page_title="Pipeline Showcase", layout="wide")
    # run pipeline
    pipeline = Pipeline()   
    # Create two columns
    col1, col2 = st.columns([0.5,1])

    with col1:
        st.header("Inputs")
        
        # File uploaders for sdt, mask, and irf files
        sdt_file = st.file_uploader("Upload sdt file", type=['sdt'])
        mask_file = st.file_uploader("Upload mask file", type=['tiff'])
        irf_file = st.file_uploader("Upload irf file", type=['txt'])
    
        # Switch to show intermediate steps
        show_intermediate = st.checkbox("Show Intermediate Steps")

    with col2:
        st.header("Results")

        # Check if all files are uploaded
        if sdt_file and mask_file and irf_file:
            images = []
            image = pipeline.mask_image(Path(sdt_file.name),Path(irf_file.name), Path(mask_file.name), flute_mode=False)
            if show_intermediate:
                # Display intermediate steps
                show_intermediate_steps(sdt_file.name, mask_file.name, irf_file.name, image["IRF_decay"])
            else:
                coords = calculate_phaser_coords(image, pipeline)
                show_phasor_coord = st.checkbox("Show Single-cell phasor coordinates")
                if show_phasor_coord:
                    show_coord(coords, image)
                else:
                    phasor_plot(coords, image, phasor_plot_callback, sdt_file.name, mask_file.name)
        else:
            st.warning("Please upload all required files.")

if __name__ == "__main__":
    main()