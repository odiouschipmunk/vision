# Enhanced Shot Detection System

## Overview
The shot detection system has been significantly improved with better shot type classification, enhanced player identification, and comprehensive shot phase tracking.

## Key Enhancements

### 1. Shot Classification Model (`ShotClassificationModel`)
- **Comprehensive Shot Types**: Detects 8 different shot types:
  - `straight_drive`: Low trajectory shots down the wall
  - `crosscourt`: Diagonal shots across the court
  - `drop_shot`: Soft shots to the front court
  - `lob`: High defensive shots to back court
  - `boast`: Three-wall shots with side wall contact
  - `kill_shot`: Low, fast shots for winners
  - `volley`: Intercepted shots before bounce
  - `defensive_lob`: Very high defensive shots

- **Advanced Feature Analysis**:
  - Horizontal/vertical movement ratios
  - Speed profiles (slow, medium, fast, very fast)
  - Height profiles (very low, low, medium, high, very high)
  - Trajectory curvature analysis
  - Wall interaction detection
  - Court zone analysis (9 zones)

### 2. Enhanced Player Hit Detection (`PlayerHitDetector`)
- **Multiple Detection Methods**:
  - Proximity-based detection (player distance to ball)
  - Trajectory analysis (velocity/direction changes)
  - Racket position analysis (wrist/elbow keypoints)
  - Movement pattern analysis (player moving toward ball)

- **Weighted Confidence System**:
  - Combines all detection methods with confidence weights
  - Returns player ID, confidence score, and hit type
  - Supports hit types: `strong_hit`, `probable_hit`, `possible_hit`

### 3. Shot Phase Detection (`ShotPhaseDetector`)
- **Three-Phase Shot Tracking**:
  - **START**: Ball leaves racket (detected on hit)
  - **MIDDLE**: Ball hits wall (detected by trajectory changes near walls)
  - **END**: Ball hits floor (detected by downward movement and velocity decrease)

- **Phase Transition Detection**:
  - Automatic detection of phase changes
  - Confidence scoring for each transition
  - Detailed logging of transition frames and positions

### 4. Enhanced Shot Tracker (`ShotTracker`)
- **Comprehensive Shot Management**:
  - Tracks active and completed shots
  - Maintains shot history with full trajectory data
  - Phase-based visualization with different colors

- **Shot Statistics**:
  - Total shots, active shots, shots by player
  - Shot type distribution
  - Phase completion statistics
  - Average shot duration
  - Wall hit distribution
  - Hit confidence averages

## Visual Enhancements

### 1. Phase-Based Trajectory Colors
- **Yellow**: START phase (ball leaves racket)
- **Orange**: MIDDLE phase (ball hits wall)
- **Magenta**: END phase (ball hits floor)

### 2. Shot Markers
- **START markers**: Large circles with player ID
- **WALL hit markers**: Yellow squares with "WALL" text
- **FLOOR hit markers**: Red circles with "FLOOR" text

### 3. Real-Time Information Display
- Current shot phase and player
- Hit confidence and type
- Shot type classification
- Phase transition history

## Usage in Main Function

### Enhanced Shot Detection Flow
1. **Ball Position Tracking**: Maintains `past_ball_pos` with trajectory history
2. **Enhanced Hit Detection**: Uses `determine_ball_hit_enhanced()` with multiple algorithms
3. **Shot Classification**: Uses `shot_type_enhanced()` with ML-like feature analysis
4. **Phase Tracking**: Automatically detects and tracks shot phases
5. **Visualization**: Real-time display of shot information and trajectory

### Key Function Calls
```python
# Enhanced hit detection
who_hit, hit_confidence, hit_type = determine_ball_hit_enhanced(players, past_ball_pos)

# Enhanced shot classification
type_of_shot = shot_type_enhanced(past_ball_pos, frame_width, frame_height)

# Shot start detection
shot_started = shot_tracker.detect_shot_start(
    ball_hit, who_hit, frame_count, current_ball_pos, type_of_shot, hit_confidence, hit_type
)

# Active shot updates with phase tracking
shot_tracker.update_active_shots(current_ball_pos, frame_count, shot_started, type_of_shot)
```

## Technical Improvements

### 1. Better Accuracy
- Multi-algorithm hit detection with confidence weighting
- Physics-based trajectory analysis
- Multiple keypoint player position detection

### 2. Comprehensive Analysis
- 20+ trajectory features for shot classification
- Court zone mapping and coverage analysis
- Velocity profile analysis

### 3. Real-Time Performance
- Optimized algorithms for video processing
- Efficient trajectory management
- GPU-optimized operations where applicable

## Configuration Options

### Detection Thresholds
- Hit confidence threshold: 0.3 (adjustable)
- Proximity threshold: 100 pixels (adjustable)
- Wall hit threshold: 30 pixels (adjustable)
- Phase transition confidence: 0.6 (adjustable)

### Visualization Settings
- Active shot trajectory thickness: 2-3 pixels
- Completed shot transparency: 50%
- Phase marker sizes: 6-15 pixels
- Text overlay font sizes: 0.3-0.5

## Benefits

1. **More Accurate Shot Detection**: Multi-algorithm approach reduces false positives
2. **Better Player Identification**: Enhanced keypoint analysis and movement tracking
3. **Comprehensive Shot Analysis**: Detailed phase tracking and classification
4. **Rich Visual Feedback**: Real-time shot information and trajectory visualization
5. **Performance Analytics**: Detailed statistics for coaching analysis

## Future Enhancements

1. **Machine Learning Integration**: Train models on shot data for better classification
2. **Advanced Court Mapping**: 3D court reconstruction for better positioning
3. **Predictive Analysis**: Predict shot outcomes based on trajectory
4. **Player Style Analysis**: Learn individual player patterns and preferences
