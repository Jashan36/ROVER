"""
Hazard Panel Component
Hazard warnings and details
"""

import streamlit as st
from typing import Optional


def render_hazard_panel(processed_data):
    """
    Render hazard warning panel
    
    Args:
        processed_data: ProcessedData object
    """
    st.markdown("### ⚠️ Hazard Detection")
    
    if processed_data is None:
        st.info("⏳ Waiting for hazard data...")
        return
    
    # Summary
    _render_hazard_summary(processed_data)
    
    # Detailed breakdown
    _render_hazard_breakdown(processed_data)
    
    # Alert log
    _render_alert_log(processed_data)


def _render_hazard_summary(processed_data):
    """Render hazard summary"""
    # Color-coded alert based on hazard count
    if processed_data.current_hazards == 0:
        st.success("✅ **No hazards detected** - Path is clear")
    elif processed_data.current_hazards <= 2:
        st.warning(f"⚠️ **{processed_data.current_hazards} hazards detected** - Proceed with caution")
    else:
        st.error(f"🚨 **{processed_data.current_hazards} hazards detected** - High risk area")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Hazards",
            processed_data.current_hazards
        )
    
    with col2:
        st.metric(
            "Peak Hazards",
            processed_data.max_hazards
        )
    
    with col3:
        total_hazards = sum(processed_data.hazard_breakdown.values()) if processed_data.hazard_breakdown else 0
        st.metric(
            "Total Detected",
            total_hazards
        )


def _render_hazard_breakdown(processed_data):
    """Render hazard type breakdown"""
    if not processed_data.hazard_breakdown:
        return
    
    st.markdown("#### Hazard Breakdown")
    
    from dashboard.utils.visualizations import create_hazard_bar_chart
    
    fig = create_hazard_bar_chart(
        processed_data.hazard_breakdown,
        title="Cumulative Hazard Detection"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed list
    with st.expander("📋 Detailed List"):
        for hazard_type, count in sorted(
            processed_data.hazard_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Hazard icon based on type
                icon = _get_hazard_icon(hazard_type)
                st.write(f"{icon} **{hazard_type.replace('_', ' ').title()}**")
            
            with col2:
                st.write(f"**{count}**")


def _render_alert_log(processed_data):
    """Render recent alert log"""
    st.markdown("#### Recent Alerts")
    
    # Simulated alert log (in production, this would come from actual data)
    with st.expander("📜 Alert History"):
        if processed_data.current_hazards > 0:
            st.markdown("""
            - 🔴 **12:34:56** - Large rock detected at 3.2m
            - 🟡 **12:34:52** - Sand trap identified ahead
            - 🟡 **12:34:48** - Low confidence region detected
            """)
        else:
            st.info("No recent alerts")


def _get_hazard_icon(hazard_type: str) -> str:
    """Get emoji icon for hazard type"""
    icons = {
        'large_rock': '🪨',
        'rock_cluster': '🏔️',
        'sand_trap': '🏖️',
        'steep_slope': '⛰️',
        'uncertain_terrain': '❓',
        'shadow_region': '🌑'
    }
    return icons.get(hazard_type, '⚠️')


def render_hazard_settings():
    """Render hazard detection settings"""
    with st.expander("⚙️ Detection Settings", expanded=False):
        st.markdown("#### Sensitivity")
        
        sensitivity = st.select_slider(
            "Detection Sensitivity",
            options=["Low", "Medium", "High"],
            value="Medium",
            help="Adjust hazard detection sensitivity"
        )
        
        st.markdown("#### Hazard Types")
        
        detect_rocks = st.checkbox("Detect Rocks", value=True)
        detect_sand = st.checkbox("Detect Sand Traps", value=True)
        detect_slopes = st.checkbox("Detect Steep Slopes", value=True)
        detect_uncertain = st.checkbox("Detect Uncertain Terrain", value=True)
        
        return {
            'sensitivity': sensitivity,
            'detect_rocks': detect_rocks,
            'detect_sand': detect_sand,
            'detect_slopes': detect_slopes,
            'detect_uncertain': detect_uncertain
        }