import cv2
import numpy as np
import time
import sys

def main():
    print("PPE Safety Monitor - Python Version")
    print("===================================")
    
    # Check OpenCV
    try:
        print(f"OpenCV version: {cv2.__version__}")
    except:
        print("❌ OpenCV not available")
        print("Install with: conda install -c conda-forge opencv")
        return
    
    # Test camera
    print("\n🔍 Testing cameras...")
    camera_index = -1
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
                camera_index = i
                cap.release()
                break
            cap.release()
    
    if camera_index == -1:
        print("❌ No working camera found")
        return
    
    print(f"\n🎥 Starting monitoring with camera {camera_index}...")
    print("Press 'q' to quit, 'p' to pause")
    
    cap = cv2.VideoCapture(camera_index)
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
            
        frame_count += 1
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Add overlay information
        cv2.putText(frame, "PPE SAFETY MONITOR", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Simulate PPE detection
        if frame_count % 100 == 0:
            cv2.putText(frame, "SAFETY CHECK: HARD HAT DETECTED", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif frame_count % 50 == 0:
            cv2.putText(frame, "SAFETY CHECK: VEST DETECTED", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('PPE Safety Monitor - Python', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("⏸️  Paused. Press any key to continue...")
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Monitoring stopped. Processed {frame_count} frames.")

if __name__ == "__main__":
    main()
