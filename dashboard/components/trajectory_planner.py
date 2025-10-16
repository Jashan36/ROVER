"""
Trajectory Planner Component
Path visualization and planning
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Optional, List, Tuple


def render_trajectory_planner(processed_data):
    """
    Render trajectory planning view
    
    Args:
        processed_data: ProcessedData object
    """
    st.markdown("### 🛣️ Trajectory Planning")
    
    if processed_data is None:
        st.info("⏳ Waiting for navigation data...")
        return
    
    # Trajectory visualization
    _render_trajectory_plot(processed_data)
    
    # Waypoint management
    _render_waypoint_manager()


def _render_trajectory_plot(processed_data):
    """Render trajectory plot"""
    st.markdown("#### Planned Path")
    
    # Create simple trajectory based on direction history
    fig = go.Figure()
    
    # Current position (origin)
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        name='Current Position',
        marker=dict(size=15, color='green', symbol='circle')
    ))
    
    # Projected path based on best direction
    direction_rad = np.radians(processed_data.current_direction)
    path_length = 10  # meters
    
    path_x = [0, path_length * np.sin(direction_rad)]
    path_y = [0, path_length * np.cos(direction_rad)]
    
    fig.add_trace(go.Scatter(
        x=path_x,
        y=path_y,
        mode='lines+markers',
        name='Recommended Path',
        line=dict(color='cyan', width=3, dash='dash'),
        marker=dict(size=10)
    ))
    
    # Simulated waypoints
    waypoints_x = [path_length * 0.5 * np.sin(direction_rad),
                   path_length * np.sin(direction_rad)]
    waypoints_y = [path_length * 0.5 * np.cos(direction_rad),
                   path_length * np.cos(direction_rad)]
    
    fig.add_trace(go.Scatter(
        x=waypoints_x,
        y=waypoints_y,
        mode='markers',
        name='Waypoints',
        marker=dict(size=12, color='yellow', symbol='diamond')
    ))
    
    # Add safety zones
    if processed_data.current_hazards > 0:
        # Simulated hazard positions
        hazard_angles = np.random.uniform(-45, 45, processed_data.current_hazards)
        hazard_distances = np.random.uniform(2, 8, processed_data.current_hazards)
        
        first_hazard = True
        for angle, dist in zip(hazard_angles, hazard_distances):
            angle_rad = np.radians(angle + processed_data.current_direction)
            hx = dist * np.sin(angle_rad)
            hy = dist * np.cos(angle_rad)
            
            fig.add_trace(go.Scatter(
                x=[hx],
                y=[hy],
                mode='markers',
                name='Hazard' if first_hazard else None,
                marker=dict(size=20, color='red', symbol='x'),
                showlegend=bool(first_hazard)
            ))
            first_hazard = False
    
    fig.update_layout(
        title="Navigation View (Top-Down)",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        template='plotly_dark',
        height=500,
        hovermode='closest',
        showlegend=True,
        xaxis=dict(scaleanchor='y', scaleratio=1),
        yaxis=dict(scaleanchor='x', scaleratio=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_waypoint_manager():
    """Render waypoint management interface"""
    st.markdown("#### Waypoint Management")
    
    with st.expander("➕ Add Waypoint", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            wp_x = st.number_input("X Position (m)", value=0.0, step=0.5)
        
        with col2:
            wp_y = st.number_input("Y Position (m)", value=5.0, step=0.5)
        
        wp_name = st.text_input("Waypoint Name", value="WP_01")
        
        if st.button("Add Waypoint"):
            st.success(f"✅ Added waypoint '{wp_name}' at ({wp_x}, {wp_y})")
    
    # Waypoint list
    st.markdown("#### Active Waypoints")
    
    # Simulated waypoint list
    waypoints = [
        {"name": "WP_01", "x": 5.0, "y": 5.0, "status": "Pending"},
        {"name": "WP_02", "x": 10.0, "y": 0.0, "status": "Pending"}
    ]
    
    for wp in waypoints:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.write(f"**{wp['name']}**")
        
        with col2:
            st.write(f"({wp['x']:.1f}, {wp['y']:.1f})")
        
        with col3:
            st.write(wp['status'])
        
        with col4:
            if st.button("Remove", key=f"remove_{wp['name']}"):
                st.info(f"Removed {wp['name']}")


def render_trajectory_controls():
    """Render trajectory planning controls"""
    with st.expander("🎮 Planning Controls", expanded=False):
        st.markdown("#### Path Planning")
        
        planning_mode = st.selectbox(
            "Planning Mode",
            ["Autonomous", "Assisted", "Manual"],
            help="Select path planning mode"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_speed = st.slider(
                "Max Speed (m/s)",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        with col2:
            safety_margin = st.slider(
                "Safety Margin (m)",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5
            )
        
        avoid_hazards = st.checkbox(
            "Avoid Detected Hazards",
            value=True
        )
        
        optimize_path = st.checkbox(
            "Optimize Path Length",
            value=True
        )
        
        return {
            'mode': planning_mode,
            'max_speed': max_speed,
            'safety_margin': safety_margin,
            'avoid_hazards': avoid_hazards,
            'optimize_path': optimize_path
        }
