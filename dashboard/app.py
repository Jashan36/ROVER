"""
Mars Rover AI Dashboard
Real-time web interface for terrain analysis and navigation monitoring

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import sys
from pathlib import Path
import time
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utilities
from dashboard.utils.ros_connector import ROSConnector
from dashboard.utils.data_processor import DataProcessor
from dashboard.utils.direct_camera import DirectCameraProcessor
from dashboard.utils.camera_stream import fetch_ip_camera_frame

# Import components
from dashboard.components import camera_view
from dashboard.components import map_viewer
from dashboard.components import metrics
from dashboard.components import hazard_panel
from dashboard.components import science_notes
from dashboard.components import trajectory_planner

# Page configuration
st.set_page_config(
    page_title="Mars Rover AI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styling"""
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'connector' not in st.session_state:
        st.session_state.connector = None
    
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "ROS/WebSocket"
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DataProcessor(history_size=100)
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = 0.0
    
    if 'paused' not in st.session_state:
        st.session_state.paused = False
    
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'camera_stream_url' not in st.session_state:
        st.session_state.camera_stream_url = ""
    
    if 'camera_connected' not in st.session_state:
        st.session_state.camera_connected = False

    if 'direct_camera_processor' not in st.session_state:
        st.session_state.direct_camera_processor = None

    if 'latest_camera_frame' not in st.session_state:
        st.session_state.latest_camera_frame = None

    if 'direct_camera_error' not in st.session_state:
        st.session_state.direct_camera_error = None

    if 'camera_control_settings' not in st.session_state:
        st.session_state.camera_control_settings = {}

init_session_state()


# Sidebar controls
def render_sidebar():
    """Render sidebar with controls and settings"""
    with st.sidebar:
        logo_path = Path(__file__).parent / "assets" / "favicon.png"
        if logo_path.exists():
            st.image(str(logo_path), width=100)
        
        st.title("🚀 Mars Rover AI")
        st.markdown("---")

        st.subheader("📡 Data Source")
        st.session_state.data_source = st.radio(
            "Select camera source",
            options=["ROS/WebSocket", "Direct IP Camera"],
            index=0 if st.session_state.data_source == "ROS/WebSocket" else 1,
            help="Choose to stream processed ROS data or a direct IP camera feed."
        )
        st.markdown("---")
        
        # Connection settings
        if st.session_state.data_source == "ROS/WebSocket":
            st.subheader("🔌 Connection")
            
            websocket_uri = st.text_input(
                "WebSocket URI",
                value="ws://localhost:8765",
                help="WebSocket server address"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Connect", use_container_width=True):
                    connect_to_ros(websocket_uri)
            
            with col2:
                if st.button("Disconnect", use_container_width=True):
                    disconnect_from_ros()
            
            # Connection status
            if st.session_state.connector and st.session_state.connector.is_connected():
                st.success("✅ Connected")
            else:
                st.error("🔴 Disconnected")
            
            st.markdown("---")
        else:
            if st.session_state.connector:
                disconnect_from_ros()

            st.subheader("📷 IP Camera Connection")
            default_url = st.session_state.camera_stream_url if st.session_state.camera_stream_url else ""
            pending_url = st.text_input(
                "Camera Stream URL",
                value=default_url,
                placeholder="http://<ip-address>:<port>/video",
                help="Enter the MJPEG/HTTP stream from your phone or camera."
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Connect Camera", use_container_width=True):
                    if pending_url.strip():
                        st.session_state.camera_stream_url = pending_url.strip()
                        st.session_state.camera_connected = True
                        st.session_state.direct_camera_processor = None
                        st.session_state.direct_camera_error = None
                        st.session_state.latest_camera_frame = None
                        st.success("Camera stream configured. Check the Camera tab.")
                    else:
                        st.error("Please enter a valid camera stream URL before connecting.")
            with col2:
                if st.button("Disconnect Camera", use_container_width=True):
                    st.session_state.camera_connected = False
                    st.session_state.camera_stream_url = ""
                    st.session_state.direct_camera_processor = None
                    st.session_state.direct_camera_error = None
                    st.session_state.latest_camera_frame = None
                    st.session_state.current_data = None
                    st.session_state.processed_data = None
                    st.info("Camera stream cleared.")

            if st.session_state.camera_connected and st.session_state.camera_stream_url:
                st.success(f"Connected to {st.session_state.camera_stream_url}")
            else:
                st.warning("Camera not connected.")

            st.markdown("---")
        
        # Controls
        st.subheader("⚙️ Controls")
        
        # Pause/Resume
        if st.session_state.paused:
            if st.button("▶️ Resume", use_container_width=True):
                st.session_state.paused = False
                st.rerun()
        else:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.paused = True
                st.rerun()
        
        # Reset data
        if st.button("🔄 Reset Data", use_container_width=True):
            st.session_state.processor.reset()
            st.success("Data reset!")
        
        # Refresh rate
        refresh_rate = st.slider(
            "Refresh Rate (Hz)",
            min_value=1,
            max_value=10,
            value=5,
            help="Dashboard update frequency"
        )
        
        st.session_state.refresh_interval = 1.0 / refresh_rate
        
        st.markdown("---")
        
        # System info
        st.subheader("ℹ️ System Info")
        
        if st.session_state.connector:
            stats = st.session_state.connector.get_statistics()
            
            st.metric("Messages", stats.get('messages_received', 0))
            st.metric("Queue Size", stats.get('queue_size', 0))
            
            time_since = stats.get('time_since_last_message', 0)
            st.metric("Last Update", f"{time_since:.1f}s ago")
        
        if st.session_state.processor:
            proc_stats = st.session_state.processor.get_statistics()
            st.metric("Data Points", proc_stats.get('data_points', 0))
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Mars Rover AI Dashboard**
            
            Real-time terrain analysis and navigation monitoring for autonomous Mars rovers.
            
            **Features:**
            - Live camera feed with terrain overlay
            - Hazard detection and warnings
            - Traversability analysis
            - Performance metrics
            - Scientific observations
            
            **Version:** 1.0.0
            """)


def connect_to_ros(uri: str):
    """Connect to ROS WebSocket bridge"""
    try:
        if st.session_state.connector:
            st.session_state.connector.stop()
        
        st.session_state.connector = ROSConnector(websocket_uri=uri)
        st.session_state.connector.start()
        
        st.success(f"Connecting to {uri}...")
        time.sleep(1)
        
    except Exception as e:
        st.error(f"Connection failed: {e}")
        logger.error(f"Connection error: {e}")


def disconnect_from_ros():
    """Disconnect from ROS"""
    if st.session_state.connector:
        st.session_state.connector.stop()
        st.session_state.connector = None
        st.info("Disconnected")


def update_data():
    """Update data based on the selected source."""
    if st.session_state.paused:
        return False

    if st.session_state.data_source == "Direct IP Camera":
        return update_data_from_direct_camera()

    return update_data_from_ros()


def update_data_from_ros():
    """Fetch and process data from the ROS/WebSocket connector."""
    if not st.session_state.connector:
        return False

    terrain_data = st.session_state.connector.get_latest_data()

    if terrain_data and terrain_data.is_valid:
        st.session_state.current_data = terrain_data
        st.session_state.processed_data = st.session_state.processor.process(terrain_data)
        st.session_state.last_update = time.time()
        return True

    return False


def update_data_from_direct_camera():
    """Fetch a frame from the direct camera feed and run local inference."""
    if not st.session_state.camera_connected or not st.session_state.camera_stream_url:
        st.session_state.latest_camera_frame = None
        return False

    frame = fetch_ip_camera_frame(st.session_state.camera_stream_url)
    if frame is None:
        st.session_state.latest_camera_frame = None
        st.session_state.current_data = None
        st.session_state.processed_data = None
        st.session_state.direct_camera_error = "Failed to retrieve frame from the camera stream."
        return False

    st.session_state.latest_camera_frame = frame

    processor = st.session_state.direct_camera_processor
    if processor is None:
        processor = DirectCameraProcessor()
        st.session_state.direct_camera_processor = processor

    overlay_alpha = st.session_state.camera_control_settings.get("overlay_alpha")
    if overlay_alpha is not None:
        processor.set_overlay_alpha(overlay_alpha)

    try:
        terrain_data = processor.process_frame(frame)
    except RuntimeError as exc:
        st.session_state.direct_camera_error = str(exc)
        st.session_state.current_data = None
        st.session_state.processed_data = None
        logger.error("Direct camera pipeline unavailable: %s", exc)
        return False
    except Exception as exc:
        st.session_state.direct_camera_error = f"Camera processing failed: {exc}"
        st.session_state.current_data = None
        st.session_state.processed_data = None
        logger.error("Direct camera processing error: %s", exc, exc_info=True)
        return False

    st.session_state.direct_camera_error = None
    st.session_state.current_data = terrain_data
    st.session_state.processed_data = st.session_state.processor.process(terrain_data)
    st.session_state.last_update = time.time()

    return True


def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.title("🚀 Mars Rover Dashboard")
    
    with col2:
        if st.session_state.current_data:
            # Status indicator
            age = time.time() - st.session_state.last_update
            if age < 2.0:
                st.success(f"🟢 LIVE - Updated {age:.1f}s ago")
            elif age < 5.0:
                st.warning(f"🟡 DELAYED - {age:.1f}s ago")
            else:
                st.error(f"🔴 STALE - {age:.1f}s ago")
        else:
            st.info("⏳ Waiting for data...")
    
    with col3:
        if st.session_state.paused:
            st.warning("⏸️ PAUSED")
        else:
            st.info("▶️ RUNNING")
    
    st.markdown("---")


def render_main_dashboard():
    """Render main dashboard content"""
    # Update data
    data_updated = update_data()
    
    # Render header
    render_header()
    
    # Get current data
    terrain_data = st.session_state.current_data
    processed_data = st.session_state.processed_data
    connector_stats = st.session_state.connector.get_statistics() if st.session_state.connector else None
    
    # Main content in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Camera",
        "🗺️ Map",
        "⚠️ Hazards",
        "📊 Metrics",
        "📝 Science"
    ])
    
    with tab1:
        # Camera view
        camera_controls = camera_view.render_camera_controls()
        st.session_state.camera_control_settings = camera_controls

        if (
            st.session_state.data_source == "Direct IP Camera"
            and st.session_state.direct_camera_error
        ):
            st.error(st.session_state.direct_camera_error)

        stream_url = st.session_state.camera_stream_url if st.session_state.camera_connected else None
        camera_view.render_camera_view(
            terrain_data,
            camera_stream_url=stream_url,
            raw_frame=st.session_state.latest_camera_frame
        )
    
    with tab2:
        # Map viewer
        col1, col2 = st.columns([2, 1])
        
        with col1:
            map_viewer.render_map_viewer(processed_data)
        
        with col2:
            trajectory_planner.render_trajectory_planner(processed_data)
    
    with tab3:
        # Hazard panel
        hazard_panel.render_hazard_panel(processed_data)
    
    with tab4:
        # Metrics
        metrics.render_metrics(processed_data, connector_stats)
        
        st.markdown("---")
        
        # Performance charts
        metrics.render_performance_charts(processed_data)
    
    with tab5:
        # Science notes
        science_notes.render_science_notes(processed_data)
    
    # Auto-refresh
    if not st.session_state.paused:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()


def main():
    """Main application entry point"""
    try:
        # Render sidebar
        render_sidebar()
        
        # Render main dashboard
        render_main_dashboard()
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}", exc_info=True)
        
        if st.button("Restart Dashboard"):
            st.rerun()


if __name__ == "__main__":
    main()
