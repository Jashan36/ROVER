"""
Camera View Component
Live camera feed with terrain overlay
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from dashboard.utils.camera_stream import fetch_ip_camera_frame


def render_camera_view(
    terrain_data,
    col_width=None,
    camera_stream_url: Optional[str] = None,
    raw_frame: Optional[np.ndarray] = None,
):
    """
    Render live camera view with optional terrain overlays.

    Args:
        terrain_data: TerrainData object
        col_width: Column width (for layout)
        camera_stream_url: Optional camera stream URL provided by the user
        raw_frame: Optional pre-fetched RGB frame for direct camera mode
    """
    st.markdown("### Live Camera Feed")

    if camera_stream_url:
        st.markdown(f"**Current IP Stream:** `{camera_stream_url}`")

    if terrain_data is None or terrain_data.overlay_image is None:
        frame = raw_frame
        if frame is None and camera_stream_url:
            frame = fetch_ip_camera_frame(camera_stream_url)

        if frame is not None:
            st.image(frame, caption="Live IP Camera Feed", use_container_width=True)
        else:
            if camera_stream_url:
                st.error("Failed to retrieve frame from the IP camera stream. Check the URL or network connection.")
            else:
                st.info("Waiting for camera data...")
            st.image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                caption="No camera feed",
                use_container_width=True,
            )
        return

    if camera_stream_url and st.session_state.get("camera_connected"):
        frame = raw_frame if raw_frame is not None else fetch_ip_camera_frame(camera_stream_url)
        if frame is not None:
            st.image(frame, caption="Live IP Camera Feed", use_container_width=True)
            st.markdown("---")
        else:
            st.error("Failed to retrieve frame from the IP camera stream. Check the URL or network connection.")
            st.markdown("---")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overlay", "Traversability", "Hazards"])

    with tab1:
        if terrain_data.overlay_image is not None:
            st.image(
                terrain_data.overlay_image,
                caption=f"Terrain Overlay - {terrain_data.timestamp:.2f}s",
                use_container_width=True,
            )
        else:
            st.warning("No overlay image available")

    with tab2:
        if terrain_data.traversability_image is not None:
            st.image(
                terrain_data.traversability_image,
                caption="Traversability Analysis",
                use_container_width=True,
            )
        else:
            st.warning("No traversability image available")

    with tab3:
        if terrain_data.hazard_image is not None:
            st.image(
                terrain_data.hazard_image,
                caption=f"Hazard Detection - {terrain_data.num_hazards} hazards",
                use_container_width=True,
            )
        else:
            st.warning("No hazard image available")

    # Image metadata
    with st.expander("Image Info"):
        col1, col2, col3 = st.columns(3)

        with col1:
            if terrain_data.overlay_image is not None:
                height, width = terrain_data.overlay_image.shape[0], terrain_data.overlay_image.shape[1]
                st.metric("Resolution", f"{width}x{height}")

        with col2:
            st.metric("Processing Time", f"{terrain_data.inference_time_ms:.1f} ms")

        with col3:
            st.metric("FPS", f"{terrain_data.fps:.1f}")


def render_camera_controls():
    """Render camera control panel."""
    st.markdown("### Camera Controls")

    with st.expander("Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            overlay_alpha = st.slider(
                "Overlay Transparency",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Adjust terrain overlay transparency",
            )

        with col2:
            update_rate = st.slider(
                "Update Rate (Hz)",
                min_value=1,
                max_value=10,
                value=5,
                help="Camera feed update frequency",
            )

        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=False,
            help="Display AI confidence levels",
        )

        show_grid = st.checkbox(
            "Show Grid Overlay",
            value=False,
            help="Display reference grid",
        )

        return {
            "overlay_alpha": overlay_alpha,
            "update_rate": update_rate,
            "show_confidence": show_confidence,
            "show_grid": show_grid,
        }
