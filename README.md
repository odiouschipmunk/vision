# Squash Match Analysis System
![Squash Analysis Demo](https://github.com/odiouschipmunk/vision/annotated2.gif)
## Overview
An advanced machine learning tool for analyzing squash matches through computer vision and AI. The system tracks player movements, ball trajectories, and classifies shot types to provide detailed insights for players and coaches.

## Features
- Real-time player tracking and pose estimation
- Ball trajectory tracking and prediction
- Shot type classification
- 3D position mapping of court movements
- Performance analysis and statistics
- Video processing and frame analysis

## System Components

### 1. Core Analysis Modules
- **Player Tracking**: Detects and tracks players using pose estimation
- **Ball Tracking**: Monitors ball position and trajectory
- **Shot Classification**: Identifies different types of shots (e.g., straight drive, crosscourt)
- **3D Position Mapping**: Converts 2D coordinates to 3D court positions

### 2. Data Processing
The system processes video input through several stages:
- Frame extraction and analysis
- Pose detection and tracking
- Ball position detection
- Shot type classification
- Data aggregation and storage


### 3. Key Classes

#### Ball Class
Handles ball tracking and position management:
- Position tracking in 2D and 3D coordinates
- Historical position storage
- Coordinate conversion methods

#### Player Class
Manages player tracking and pose data:
- Pose history tracking
- Real-time position updates
- Movement analysis

### 4. Data Output
The system generates several types of output:
- CSV files with frame-by-frame analysis
- Text descriptions of player positions and movements
- 3D visualizations of court positions
- Performance statistics and insights


## Contact
For help and contributions, please contact:
- Email: odiouschipmunk@gmail.com
- Documentation: [Google Docs](https://docs.google.com/document/d/1egeolMCFvLH1VurDKju9ZjA_MYELRHi1MeEyeTkoAA0/)

