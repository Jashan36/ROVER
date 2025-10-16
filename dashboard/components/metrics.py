"""
Metrics Component
Performance metrics and statistics
"""

import streamlit as st
import numpy as np
from typing import Optional


def render_metrics(processed_data, connector_stats):
    """
    Render performance metrics
    
    Args:
        processed_data: ProcessedData object
        connector_stats: Connector statistics
    """
    st.markdown("### 📊 System Metrics")
    
    if processed_data is None:
        st.info("⏳ Waiting for data...")
        return
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Traversability
        current_traversability = 0.0
        try:
            current_traversability = float(processed_data.current_traversability)
        except (TypeError, ValueError):
            pass

        delta = None
        if len(processed_data.traversability_history) > 1:
            try:
                previous = float(processed_data.traversability_history[-2])
                delta = current_traversability - previous
            except (TypeError, ValueError):
                delta = None
        
        st.metric(
            "Traversability",
            f"{current_traversability:.1%}",
            delta=f"{delta:.1%}" if delta is not None else None,
            help="Current terrain safety score"
        )
    
    with col2:
        # Hazards
        try:
            current_hazards = int(processed_data.current_hazards)
        except (TypeError, ValueError):
            current_hazards = 0

        delta = None
        if len(processed_data.hazard_history) > 1:
            try:
                previous = int(processed_data.hazard_history[-2])
                delta = current_hazards - previous
            except (TypeError, ValueError):
                delta = None
        
        st.metric(
            "Hazards Detected",
            current_hazards,
            delta=delta if delta is not None else None,
            delta_color="inverse",
            help="Number of detected hazards"
        )
    
    with col3:
        # FPS
        try:
            current_fps = float(processed_data.current_fps)
        except (TypeError, ValueError):
            current_fps = 0.0

        st.metric(
            "System FPS",
            f"{current_fps:.1f}",
            help="Processing frame rate"
        )
    
    with col4:
        # Connection status
        if connector_stats and connector_stats.get('connected'):
            st.metric(
                "Connection",
                "🟢 Active",
                help="ROS2 connection status"
            )
        else:
            st.metric(
                "Connection",
                "🔴 Offline",
                help="ROS2 connection status"
            )
    
    # Detailed metrics
    _render_detailed_metrics(processed_data, connector_stats)


def _render_detailed_metrics(processed_data, connector_stats):
    """Render detailed metrics"""
    with st.expander("📈 Detailed Metrics", expanded=False):
        # Performance statistics
        st.markdown("#### Performance Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Traversability", f"{processed_data.avg_traversability:.1%}")
            st.metric("Max Hazards", processed_data.max_hazards)
        
        with col2:
            st.metric("Avg FPS", f"{processed_data.avg_fps:.1f}")
            
            if connector_stats:
                st.metric("Data Points", len(processed_data.traversability_history))
        
        # Connection statistics
        if connector_stats:
            st.markdown("#### Connection Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Messages Received",
                    connector_stats.get('messages_received', 0)
                )
            
            with col2:
                st.metric(
                    "Queue Size",
                    connector_stats.get('queue_size', 0)
                )
            
            with col3:
                time_since = connector_stats.get('time_since_last_message', 0)
                st.metric(
                    "Last Message",
                    f"{time_since:.1f}s ago"
                )


def render_performance_charts(processed_data):
    """Render performance time series charts"""
    if processed_data is None or not processed_data.time_history:
        return
    
    st.markdown("### 📈 Performance History")
    
    from dashboard.utils.visualizations import (
        create_terrain_plot,
        create_hazard_plot,
        create_performance_plot
    )
    
    # Traversability over time
    fig_trav = create_terrain_plot(
        processed_data.time_history,
        processed_data.traversability_history,
        title="Traversability Over Time"
    )
    st.plotly_chart(fig_trav, use_container_width=True)
    
    # Hazards over time
    fig_hazard = create_hazard_plot(
        processed_data.time_history,
        processed_data.hazard_history,
        title="Hazards Detected Over Time"
    )
    st.plotly_chart(fig_hazard, use_container_width=True)
    
    # FPS over time
    fig_perf = create_performance_plot(
        processed_data.time_history,
        processed_data.fps_history,
        title="System Performance (FPS)"
    )
    st.plotly_chart(fig_perf, use_container_width=True)


def render_system_health():
    """Render system health indicators"""
    st.markdown("### 🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**AI Pipeline**")
        st.success("✅ Operational")
    
    with col2:
        st.markdown("**ROS2 Bridge**")
        st.success("✅ Connected")
    
    with col3:
        st.markdown("**Navigation**")
        st.success("✅ Ready")
