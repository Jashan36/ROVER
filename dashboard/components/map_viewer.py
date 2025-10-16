"""
Map Viewer Component
Terrain map and navigation visualization
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Optional


def render_map_viewer(processed_data):
    """
    Render terrain map viewer
    
    Args:
        processed_data: ProcessedData object
    """
    st.markdown("### 🗺️ Terrain Map")
    
    if processed_data is None:
        st.info("⏳ Waiting for terrain data...")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Direction", "Terrain Composition", "Heat Map"])
    
    with tab1:
        _render_direction_indicator(processed_data)
    
    with tab2:
        _render_terrain_composition(processed_data)
    
    with tab3:
        _render_traversability_heatmap(processed_data)


def _render_direction_indicator(processed_data):
    """Render recommended direction compass"""
    from dashboard.utils.visualizations import create_direction_rose
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Direction compass
        fig = create_direction_rose(
            processed_data.current_direction,
            title="Recommended Direction"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Direction details
        st.markdown("#### Navigation Info")
        
        st.metric(
            "Best Direction",
            f"{processed_data.current_direction:.1f}°",
            help="Optimal heading for safe navigation"
        )
        
        # Safety indicator
        if processed_data.current_traversability > 0.7:
            safety_status = "🟢 Safe"
            safety_color = "green"
        elif processed_data.current_traversability > 0.4:
            safety_status = "🟡 Caution"
            safety_color = "orange"
        else:
            safety_status = "🔴 Unsafe"
            safety_color = "red"
        
        st.markdown(f"**Status:** :{safety_color}[{safety_status}]")
        
        st.progress(processed_data.current_traversability)
        st.caption(f"Traversability: {processed_data.current_traversability:.1%}")


def _render_terrain_composition(processed_data):
    """Render terrain composition pie chart"""
    from dashboard.utils.visualizations import create_terrain_pie_chart
    
    if not processed_data.terrain_composition:
        st.warning("No terrain data available")
        return
    
    fig = create_terrain_pie_chart(
        processed_data.terrain_composition,
        title="Terrain Type Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    with st.expander("📊 Detailed Breakdown"):
        for terrain, ratio in sorted(
            processed_data.terrain_composition.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            st.metric(
                terrain.replace('_', ' ').title(),
                f"{ratio:.1%}"
            )


def _render_traversability_heatmap(processed_data):
    """Render traversability heat map"""
    st.markdown("#### Traversability Heat Map")
    
    # Create synthetic heatmap from history
    if len(processed_data.traversability_history) > 10:
        # Reshape history into grid for visualization
        history = np.array(processed_data.traversability_history[-100:])
        
        # Create 10x10 grid
        grid_size = 10
        if len(history) >= grid_size * grid_size:
            heatmap = history[-grid_size*grid_size:].reshape(grid_size, grid_size)
        else:
            # Pad with current value
            padded = np.full(grid_size * grid_size, processed_data.current_traversability)
            padded[-len(history):] = history
            heatmap = padded.reshape(grid_size, grid_size)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            colorscale='RdYlGn',
            zmid=0.5,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Safety")
        ))
        
        fig.update_layout(
            title="Recent Traversability Pattern",
            template='plotly_dark',
            height=400,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Collecting data for heat map... (need 10+ samples)")


def render_map_controls():
    """Render map control panel"""
    with st.expander("🎛️ Map Controls", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            map_center = st.selectbox(
                "Center On",
                ["Rover", "Target", "Hazards"],
                help="What to center the map on"
            )
        
        with col2:
            map_zoom = st.slider(
                "Zoom Level",
                min_value=1,
                max_value=10,
                value=5,
                help="Map zoom level"
            )
        
        show_trajectory = st.checkbox(
            "Show Trajectory",
            value=True,
            help="Display planned path"
        )
        
        show_hazards = st.checkbox(
            "Show Hazards",
            value=True,
            help="Display detected hazards"
        )
        
        return {
            'center': map_center,
            'zoom': map_zoom,
            'show_trajectory': show_trajectory,
            'show_hazards': show_hazards
        }