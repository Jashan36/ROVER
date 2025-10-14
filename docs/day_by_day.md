Day-by-day plan (5 days)

Day 1 — Environment & Simulation stub
- Create Python venv and install dependencies
- Install ROS2 + Gazebo/Ignition (recommend Ubuntu/WSL2)
- Verify Streamlit app starts (run `streamlit run dashboard/app.py`)
- Create URDF stub and ensure ROS2 package layout is ready

Day 2 — Nav2 integration
- Create ROS2 Nav2 launch files and parameter configs
- Setup differential-drive controller and local planner
- Test simple navigation in Gazebo world

Day 3 — AI integration
- Download selected Perseverance images and AI4Mars dataset
- Run inference with UNet stub on sample images
- Measure CPU latency and optimize

Day 4 — Fusion & Dashboard
- Fuse DEM and segmentation into traversability scores
- Stream segmentation overlays and metrics to Streamlit
- Implement "Science Notes" generator

Day 5 — Evaluation & Demo
- Run evaluation scripts: navigation metrics, segmentation mIoU
- Record demo and finalize README and demo.mp4
