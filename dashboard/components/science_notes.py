"""
Science Notes Component
Auto-generated scientific insights and observations
"""

import streamlit as st
from datetime import datetime
from typing import Optional


def render_science_notes(processed_data):
    """
    Render auto-generated science notes
    
    Args:
        processed_data: ProcessedData object
    """
    st.markdown("### 📝 Science Notes")
    
    if processed_data is None:
        st.info("⏳ Generating observations...")
        return
    
    # Generate automatic observations
    observations = _generate_observations(processed_data)
    
    # Display in tabs
    tab1, tab2, tab3 = st.tabs(["Current", "Geology", "Navigation"])
    
    with tab1:
        _render_current_observations(observations)
    
    with tab2:
        _render_geology_notes(processed_data)
    
    with tab3:
        _render_navigation_notes(processed_data)


def _generate_observations(processed_data):
    """Generate automatic observations from data"""
    observations = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Traversability observation
    if processed_data.current_traversability > 0.8:
        observations.append({
            'time': timestamp,
            'type': 'terrain',
            'priority': 'low',
            'text': f"Excellent traversability ({processed_data.current_traversability:.1%}). Terrain appears stable and safe for navigation."
        })
    elif processed_data.current_traversability < 0.4:
        observations.append({
            'time': timestamp,
            'type': 'terrain',
            'priority': 'high',
            'text': f"Low traversability ({processed_data.current_traversability:.1%}). Exercise extreme caution."
        })
    
    # Hazard observation
    if processed_data.current_hazards > 3:
        observations.append({
            'time': timestamp,
            'type': 'hazard',
            'priority': 'high',
            'text': f"Multiple hazards detected ({processed_data.current_hazards}). Recommend alternate route."
        })
    
    # Terrain composition observation
    if processed_data.terrain_composition:
        dominant_terrain = max(
            processed_data.terrain_composition.items(),
            key=lambda x: x[1]
        )
        
        observations.append({
            'time': timestamp,
            'type': 'geology',
            'priority': 'medium',
            'text': f"Terrain dominated by {dominant_terrain[0]} ({dominant_terrain[1]:.1%}). Typical of local geological conditions."
        })
    
    return observations


def _render_current_observations(observations):
    """Render current observations"""
    st.markdown("#### Current Observations")
    
    if not observations:
        st.info("No observations at this time")
        return
    
    for obs in observations:
        # Color code by priority
        if obs['priority'] == 'high':
            st.error(f"**🔴 {obs['time']}** - {obs['text']}")
        elif obs['priority'] == 'medium':
            st.warning(f"**🟡 {obs['time']}** - {obs['text']}")
        else:
            st.info(f"**🟢 {obs['time']}** - {obs['text']}")


def _render_geology_notes(processed_data):
    """Render geological analysis"""
    st.markdown("#### Geological Analysis")
    
    if not processed_data.terrain_composition:
        st.info("Insufficient data for geological analysis")
        return
    
    # Analyze terrain composition
    composition = processed_data.terrain_composition
    
    st.markdown("**Terrain Classification:**")
    
    # Soil analysis
    if 'soil' in composition and composition['soil'] > 0.5:
        st.write("- **Regolith Cover:** Extensive (>50%)")
        st.write("  - Indicates weathered surface material")
        st.write("  - Good for sample collection")
    
    # Bedrock analysis
    if 'bedrock' in composition and composition['bedrock'] > 0.3:
        st.write("- **Bedrock Exposure:** Significant")
        st.write("  - Ancient crustal material visible")
        st.write("  - Potential for stratigraphic analysis")
    
    # Sand analysis
    if 'sand' in composition and composition['sand'] > 0.2:
        st.write("- **Sandy Deposits:** Present")
        st.write("  - Aeolian processes active")
        st.write("  - Caution: mobility risk")
    
    # Rock analysis
    if 'big_rock' in composition and composition['big_rock'] > 0.1:
        st.write("- **Rock Population:** High density")
        st.write("  - Impact ejecta or erosional remnants")
        st.write("  - Navigation challenge")


def _render_navigation_notes(processed_data):
    """Render navigation recommendations"""
    st.markdown("#### Navigation Recommendations")
    
    # Direction recommendation
    st.write(f"**Recommended Heading:** {processed_data.current_direction:.1f}°")
    
    # Safety assessment
    if processed_data.current_traversability > 0.7:
        st.success("✅ **Assessment:** Safe to proceed")
        st.write("- Clear path ahead")
        st.write("- Minimal obstacles detected")
        st.write("- Standard navigation protocols apply")
    elif processed_data.current_traversability > 0.4:
        st.warning("⚠️ **Assessment:** Proceed with caution")
        st.write("- Moderate terrain challenges")
        st.write("- Enhanced monitoring recommended")
        st.write("- Reduced speed advisable")
    else:
        st.error("🚨 **Assessment:** High risk conditions")
        st.write("- Significant hazards present")
        st.write("- Alternative route strongly recommended")
        st.write("- Consider manual navigation")
    
    # Specific recommendations
    st.markdown("**Specific Actions:**")
    
    if processed_data.current_hazards > 0:
        st.write(f"1. {processed_data.current_hazards} hazard(s) require navigation around")
    
    st.write("2. Maintain continuous terrain monitoring")
    st.write("3. Update path planning every 5 meters")


def render_note_controls():
    """Render science note controls"""
    with st.expander("✍️ Note Controls", expanded=False):
        st.markdown("#### Manual Observation")
        
        obs_type = st.selectbox(
            "Observation Type",
            ["Terrain", "Geology", "Navigation", "Other"]
        )
        
        obs_text = st.text_area(
            "Add Manual Observation",
            placeholder="Enter your observation here..."
        )
        
        if st.button("Save Observation"):
            if obs_text:
                st.success("✅ Observation saved")
            else:
                st.warning("Please enter an observation")
        
        st.markdown("#### Export")
        
        if st.button("Export All Notes"):
            st.info("Notes exported to notes.txt")