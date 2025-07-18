"""
ReID System Testing and Visualization Utility
============================================

This script provides utilities to test and visualize the enhanced player 
re-identification system.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from enhanced_player_reid import EnhancedPlayerReID
import argparse
import os

def test_reid_system(video_path, output_dir="reid_test_output"):
    """
    Test the ReID system on a video file
    
    Args:
        video_path: Path to test video
        output_dir: Directory to save test results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ReID system
    reid_system = EnhancedPlayerReID()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    detection_results = []
    
    # For testing, we'll create mock detections
    # In real usage, these would come from your pose detection system
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for faster testing
        if frame_count % 5 != 0:
            continue
        
        # Create mock detections for testing
        # In real usage, these would come from pose detection
        mock_detections = create_mock_detections(frame, frame_count)
        
        if mock_detections:
            # Process through ReID system
            result = reid_system.process_frame(mock_detections, frame_count)
            detection_results.append(result)
            
            # Visualize results
            visualize_reid_result(frame, mock_detections, result, frame_count)
        
        # Display frame
        cv2.imshow('ReID Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Stop after 200 frames for testing
        if frame_count > 200:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save results
    save_reid_test_results(detection_results, reid_system, output_dir)

def create_mock_detections(frame, frame_count):
    """
    Create mock detections for testing
    This simulates what would come from pose detection
    """
    height, width = frame.shape[:2]
    
    # Create two mock players moving around
    detections = []
    
    # Player 1 (moves left to right)
    p1_x = int(width * 0.2 + (width * 0.3) * (frame_count % 100) / 100)
    p1_y = int(height * 0.5)
    p1_crop = frame[max(0, p1_y-50):min(height, p1_y+50), 
                   max(0, p1_x-30):min(width, p1_x+30)]
    
    if p1_crop.size > 0:
        detections.append({
            'track_id': 1,
            'crop': p1_crop,
            'position': (p1_x, p1_y),
            'bbox': (max(0, p1_x-30), max(0, p1_y-50), 
                    min(width, p1_x+30), min(height, p1_y+50)),
            'keypoints': None,
            'confidence': 0.9
        })
    
    # Player 2 (moves right to left)
    p2_x = int(width * 0.8 - (width * 0.3) * (frame_count % 100) / 100)
    p2_y = int(height * 0.6)
    p2_crop = frame[max(0, p2_y-50):min(height, p2_y+50), 
                   max(0, p2_x-30):min(width, p2_x+30)]
    
    if p2_crop.size > 0:
        detections.append({
            'track_id': 2,
            'crop': p2_crop,
            'position': (p2_x, p2_y),
            'bbox': (max(0, p2_x-30), max(0, p2_y-50), 
                    min(width, p2_x+30), min(height, p2_y+50)),
            'keypoints': None,
            'confidence': 0.9
        })
    
    # Simulate track ID swap around frame 100
    if 95 <= frame_count <= 105:
        # Swap track IDs to test detection
        for det in detections:
            if det['track_id'] == 1:
                det['track_id'] = 2
            elif det['track_id'] == 2:
                det['track_id'] = 1
    
    return detections

def visualize_reid_result(frame, detections, result, frame_count):
    """
    Visualize ReID results on frame
    """
    # Draw detections
    for det in detections:
        bbox = det['bbox']
        track_id = det['track_id']
        
        # Get player assignment
        assigned_player = result['player_assignments'].get(track_id, '?')
        
        # Choose color based on assignment
        color = (0, 255, 0) if assigned_player == 1 else (255, 0, 0) if assigned_player == 2 else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw labels
        cv2.putText(frame, f'T{track_id}->P{assigned_player}', 
                   (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display swap detection
    if result['swap_detection'] and result['swap_detection']['swap_detected']:
        cv2.putText(frame, 'SWAP DETECTED!', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display initialization status
    init_status = result['initialization_status']
    status_text = f"Init: P1={init_status[1]}, P2={init_status[2]}"
    cv2.putText(frame, status_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display frame count
    cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def save_reid_test_results(results, reid_system, output_dir):
    """
    Save test results and generate report
    """
    # Save detailed results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Get final statistics
    stats = reid_system.get_statistics()
    
    # Save references
    reid_system.save_references(os.path.join(output_dir, 'test_references.json'))
    
    # Generate report
    report = f"""
ReID System Test Report
=======================

Test Results:
- Total frames processed: {len(results)}
- Track ID swaps detected: {stats['total_swaps_detected']}
- Player 1 initialization: {'Complete' if stats['initialization_status'][1] else 'Incomplete'}
- Player 2 initialization: {'Complete' if stats['initialization_status'][2] else 'Incomplete'}

Reference counts:
- Player 1: {stats['reference_counts'][1]} features
- Player 2: {stats['reference_counts'][2]} features

Final mappings: {stats['current_mappings']}

Test Configuration:
- Initialization frames: 100-150
- Proximity threshold: 100 pixels
- Confidence threshold: 0.6
- Mock swap introduced: frames 95-105

Files generated:
- test_results.json: Detailed frame-by-frame results
- test_references.json: Final player references
- test_report.txt: This report
"""
    
    with open(os.path.join(output_dir, 'test_report.txt'), 'w') as f:
        f.write(report)
    
    print(f"Test results saved to {output_dir}")
    print(f"Swaps detected: {stats['total_swaps_detected']}")

def plot_reid_performance(results_file):
    """
    Plot ReID system performance metrics
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data for plotting
    frames = [r['frame_count'] for r in results]
    swaps = [1 if r['swap_detection'] and r['swap_detection']['swap_detected'] else 0 for r in results]
    
    # Plot swap detections
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(frames, swaps, 'ro-', markersize=8)
    plt.title('Track ID Swap Detections')
    plt.xlabel('Frame')
    plt.ylabel('Swap Detected')
    plt.grid(True)
    
    # Plot initialization status
    plt.subplot(2, 1, 2)
    p1_init = [r['initialization_status'][1] for r in results]
    p2_init = [r['initialization_status'][2] for r in results]
    
    plt.plot(frames, p1_init, 'b-', label='Player 1', linewidth=2)
    plt.plot(frames, p2_init, 'r-', label='Player 2', linewidth=2)
    plt.title('Player Initialization Status')
    plt.xlabel('Frame')
    plt.ylabel('Initialized')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reid_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test ReID System')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--output', type=str, default='reid_test_output', 
                       help='Output directory for results')
    parser.add_argument('--plot', type=str, help='Plot results from JSON file')
    
    args = parser.parse_args()
    
    if args.plot:
        plot_reid_performance(args.plot)
    elif args.video:
        test_reid_system(args.video, args.output)
    else:
        print("Please provide either --video for testing or --plot for visualization")

if __name__ == "__main__":
    main()
