"""
Visualization Utilities
Plotting functions for dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_terrain_plot(
    time_history: List[float],
    traversability_history: List[float],
    title: str = "Traversability Over Time"
) -> go.Figure:
    """
    Create traversability time series plot
    
    Args:
        time_history: List of timestamps
        traversability_history: List of traversability values
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Convert timestamps to relative time
    if time_history:
        start_time = time_history[0]
        relative_time = [t - start_time for t in time_history]
    else:
        relative_time = []
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=relative_time,
        y=traversability_history,
        mode='lines',
        name='Traversability',
        line=dict(color='#00CC96', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.2)'
    ))
    
    # Add safety threshold line
    if relative_time:
        fig.add_hline(
            y=0.7,
            line_dash="dash",
            line_color="green",
            annotation_text="Safe threshold"
        )
        
        fig.add_hline(
            y=0.3,
            line_dash="dash",
            line_color="orange",
            annotation_text="Caution threshold"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Traversability Score",
        yaxis_range=[0, 1],
        hovermode='x unified',
        template='plotly_dark',
        height=300
    )
    
    return fig


def create_hazard_plot(
    time_history: List[float],
    hazard_history: List[int],
    title: str = "Hazards Detected Over Time"
) -> go.Figure:
    """
    Create hazard detection time series plot
    
    Args:
        time_history: List of timestamps
        hazard_history: List of hazard counts
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if time_history:
        start_time = time_history[0]
        relative_time = [t - start_time for t in time_history]
    else:
        relative_time = []
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=relative_time,
        y=hazard_history,
        mode='lines+markers',
        name='Hazards',
        line=dict(color='#EF553B', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Number of Hazards",
        hovermode='x unified',
        template='plotly_dark',
        height=300
    )
    
    return fig


def create_performance_plot(
    time_history: List[float],
    fps_history: List[float],
    title: str = "System Performance"
) -> go.Figure:
    """
    Create performance metrics plot
    
    Args:
        time_history: List of timestamps
        fps_history: List of FPS values
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if time_history:
        start_time = time_history[0]
        relative_time = [t - start_time for t in time_history]
    else:
        relative_time = []
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=relative_time,
        y=fps_history,
        mode='lines',
        name='FPS',
        line=dict(color='#AB63FA', width=2)
    ))
    
    # Target FPS line
    if relative_time:
        fig.add_hline(
            y=5.0,
            line_dash="dash",
            line_color="green",
            annotation_text="Target FPS"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Frames Per Second",
        hovermode='x unified',
        template='plotly_dark',
        height=300
    )
    
    return fig


def create_direction_rose(
    direction_deg: float,
    title: str = "Recommended Direction"
) -> go.Figure:
    """
    Create directional indicator (compass rose)
    
    Args:
        direction_deg: Direction in degrees (0=north/forward)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Convert to radians for plotting
    direction_rad = np.radians(direction_deg)
    
    # Create arrow
    arrow_length = 0.8
    arrow_x = [0, arrow_length * np.sin(direction_rad)]
    arrow_y = [0, arrow_length * np.cos(direction_rad)]
    
    fig = go.Figure()
    
    # Add compass circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatterpolar(
        r=[1]*100,
        theta=np.degrees(theta),
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ))
    
    # Add cardinal directions
    cardinal_angles = [0, 90, 180, 270]
    cardinal_labels = ['N', 'E', 'S', 'W']
    
    for angle, label in zip(cardinal_angles, cardinal_labels):
        fig.add_trace(go.Scatterpolar(
            r=[1.1],
            theta=[angle],
            mode='text',
            text=[label],
            textfont=dict(size=14, color='white'),
            showlegend=False
        ))
    
    # Add direction arrow
    fig.add_trace(go.Scatterpolar(
        r=[0, arrow_length],
        theta=[direction_deg, direction_deg],
        mode='lines',
        line=dict(color='#00CC96', width=4),
        showlegend=False
    ))
    
    # Add arrow head
    fig.add_trace(go.Scatterpolar(
        r=[arrow_length],
        theta=[direction_deg],
        mode='markers',
        marker=dict(
            size=15,
            color='#00CC96',
            symbol='triangle-up'
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[0, 1.2]
            ),
            angularaxis=dict(
                direction='clockwise',
                rotation=90
            )
        ),
        showlegend=False,
        title=title,
        template='plotly_dark',
        height=350,
        width=350
    )
    
    return fig


def create_terrain_pie_chart(
    terrain_composition: Dict[str, float],
    title: str = "Terrain Composition"
) -> go.Figure:
    """
    Create terrain composition pie chart
    
    Args:
        terrain_composition: Dictionary of terrain types and ratios
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if not terrain_composition:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=300
        )
        return fig
    
    labels = list(terrain_composition.keys())
    values = list(terrain_composition.values())
    
    # Colors matching terrain classes
    colors = {
        'soil': '#8B4513',      # Brown
        'bedrock': '#808080',   # Gray
        'sand': '#FFE4B5',      # Tan
        'big_rock': '#696969',  # Dark gray
        'background': '#000000'  # Black
    }
    
    color_list = [colors.get(label, '#CCCCCC') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=color_list),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=300,
        showlegend=True
    )
    
    return fig


def create_hazard_bar_chart(
    hazard_breakdown: Dict[str, int],
    title: str = "Hazard Breakdown"
) -> go.Figure:
    """
    Create hazard breakdown bar chart
    
    Args:
        hazard_breakdown: Dictionary of hazard types and counts
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if not hazard_breakdown:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=300
        )
        return fig
    
    hazard_types = list(hazard_breakdown.keys())
    counts = list(hazard_breakdown.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=hazard_types,
            y=counts,
            marker_color='#EF553B'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Hazard Type",
        yaxis_title="Total Detected",
        template='plotly_dark',
        height=300
    )
    
    return fig


# Testing
if __name__ == '__main__':
    import plotly.io as pio
    
    # Test data
    time_hist = list(range(50))
    trav_hist = [0.7 + 0.2 * np.sin(t/5) for t in time_hist]
    hazard_hist = [int(5 + 3 * np.sin(t/10)) for t in time_hist]
    fps_hist = [5 + np.random.rand() for _ in time_hist]
    
    # Create plots
    trav_fig = create_terrain_plot(time_hist, trav_hist)
    hazard_fig = create_hazard_plot(time_hist, hazard_hist)
    perf_fig = create_performance_plot(time_hist, fps_hist)
    dir_fig = create_direction_rose(45.0)
    
    terrain_comp = {
        'soil': 0.5,
        'bedrock': 0.3,
        'sand': 0.15,
        'big_rock': 0.05
    }
    pie_fig = create_terrain_pie_chart(terrain_comp)
    
    hazard_break = {
        'large_rock': 15,
        'sand_trap': 8,
        'uncertain': 5
    }
    bar_fig = create_hazard_bar_chart(hazard_break)
    
    print("Visualizations created successfully!")
    print("Figures can be displayed with fig.show()")
