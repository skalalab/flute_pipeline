# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:58:44 2024

@author: Wenxuan Zhao

ui for the pipeline 
"""
from pipeline import Pipeline
import flute_pipeline_visualizer as visualizer

import streamlit as st
import numpy as np
from io import BytesIO
from pathlib import Path

@st.cache_data
def show_intensity(sdt_file):
    fig, sdt_data = visualizer.visualize_sdt(sdt_file.name)
    return fig, sdt_data

def fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)  # Reset the buffer's pointer to the start
    return buf

def show_intermediate_steps(sdt_file, irf_file, shifted_irf):
    # Your function to process and display intermediate steps
    st.write("Reading in the sdt and mask")
    fig, sdt_data = show_intensity(sdt_file)
    buf = fig_to_buf(fig)
    st.image(buf, use_container_width=False, width=400)
    
    st.write("Reading in the irf and shift it")
    with open(irf_file.name) as irf:
        irf_values = [int(line) for line in irf if line.strip()]
    fig = visualizer.plot_irfs_against_sdt(np.array(irf_values), shifted_irf, sdt_data)
    st.pyplot(fig)


def generate_plots(image):
    # Your function to generate and display plots
    st.write("Generating plots...")
    # Include your plotting logic here
    # For example:
    # plot = your_plotting_function(sdt_file, mask_file, irf_file)
    # st.pyplot(plot)

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
                show_intermediate_steps(sdt_file, irf_file, image["IRF_decay"])
            else:
                # Display final plots
                generate_plots(image)
        else:
            st.warning("Please upload all required files.")

if __name__ == "__main__":
    main()