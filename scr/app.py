import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import time
from PIL import Image

def init_webcam():
    """Initialize webcam with multiple attempts and methods"""
    # Try with cv2.CAP_DSHOW (DirectShow) on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap

    # Try without cv2.CAP_DSHOW
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap

    # Try other camera indices
    for idx in [1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap

    return None

def process_frame(frame, model, confidence_threshold, drowsy_counter):
    """Process a single frame and return predictions"""
    # Preprocess frame
    processed_frame = cv2.resize(frame, (128, 128))
    processed_frame = processed_frame / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)
    
    # Make prediction
    prediction = model.predict(processed_frame, verbose=0)
    
    # Get prediction and confidence
    is_drowsy = prediction[0][0] > confidence_threshold
    confidence = prediction[0][0] if is_drowsy else 1 - prediction[0][0]
    
    # Update drowsy counter
    if is_drowsy:
        drowsy_counter += 1
    else:
        drowsy_counter = max(0, drowsy_counter - 1)
    
    return is_drowsy, confidence, drowsy_counter

def add_prediction_to_frame(frame, status, confidence, drowsy_counter, alert_frames, fps=None):
    """Add prediction visualization to frame"""
    if drowsy_counter > alert_frames:
        color = (0, 0, 255)  # Red
        status_text = "ALERT: Drowsiness Detected!"
    elif status:
        color = (0, 165, 255)  # Orange
        status_text = "Warning: Signs of Drowsiness"
    else:
        color = (0, 255, 0)  # Green
        status_text = "Alert"
    
    cv2.putText(frame, 
                f"{status_text} ({confidence:.2%})", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                color, 
                2)
    
    if fps is not None:
        cv2.putText(frame,
                    f"FPS: {fps:.1f}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)
    
    return frame, status_text

def process_webcam(model, confidence_threshold, alert_frames, show_fps):
    """Handle webcam input"""
    st.info("Initializing webcam... Please wait.")
    
    # Initialize webcam
    cap = init_webcam()
    if cap is None:
        st.error("Could not initialize any camera. Please check your webcam connection.")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_placeholder = st.empty()
    metrics_text = st.empty()
    drowsy_counter = 0
    frame_count = 0
    start_time = time.time()
    stop_button = st.button("Stop")
    
    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to get frame. Trying to reinitialize camera...")
                cap.release()
                cap = init_webcam()
                if cap is None:
                    st.error("Failed to recover camera connection.")
                    break
                continue
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            fps = frame_count / (time.time() - start_time) if show_fps else None
            
            is_drowsy, confidence, drowsy_counter = process_frame(
                frame, model, confidence_threshold, drowsy_counter)
            
            frame, status = add_prediction_to_frame(
                frame, is_drowsy, confidence, drowsy_counter, alert_frames, fps)
            
            frame_placeholder.image(frame, channels="BGR")
            metrics_text.text(
                f"""
                Status: {status}
                Confidence: {confidence:.2%}
                Drowsy Frames: {drowsy_counter}
                {f'FPS: {fps:.1f}' if show_fps else ''}
                """
            )
            
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Error during webcam processing: {str(e)}")
    
    finally:
        cap.release()

def process_video(video_file, model, confidence_threshold, alert_frames, show_fps):
    """Handle video file input"""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Could not open video file")
        return
    
    frame_placeholder = st.empty()
    metrics_text = st.empty()
    drowsy_counter = 0
    frame_count = 0
    start_time = time.time()
    stop_button = st.button("Stop")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        fps = frame_count / (time.time() - start_time) if show_fps else None
        
        is_drowsy, confidence, drowsy_counter = process_frame(
            frame, model, confidence_threshold, drowsy_counter)
        
        frame, status = add_prediction_to_frame(
            frame, is_drowsy, confidence, drowsy_counter, alert_frames, fps)
        
        frame_placeholder.image(frame, channels="BGR")
        metrics_text.text(
            f"""
            Status: {status}
            Confidence: {confidence:.2%}
            Drowsy Frames: {drowsy_counter}
            {f'FPS: {fps:.1f}' if show_fps else ''}
            """
        )
        
        time.sleep(0.03)
    
    cap.release()

def process_image(image_file, model, confidence_threshold):
    """Handle image file input"""
    image = Image.open(image_file)
    image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    is_drowsy, confidence, _ = process_frame(image, model, confidence_threshold, 0)
    
    image, status = add_prediction_to_frame(
        image, is_drowsy, confidence, 0, 1)
    
    st.image(image, channels="BGR")
    st.text(f"""
    Status: {status}
    Confidence: {confidence:.2%}
    """)

def main():
    st.title("Drowsiness Detection System")
    
    st.sidebar.header("Settings")
    input_type = st.sidebar.radio("Select Input Type", 
                                 ["Webcam", "Video Upload", "Image Upload"])
    
    confidence_threshold = st.sidebar.slider(
        "Detection Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5
    )
    
    alert_frames = st.sidebar.slider(
        "Alert Frames", 
        min_value=1, 
        max_value=30, 
        value=10
    )
    
    show_fps = st.sidebar.checkbox("Show FPS", value=True)
    
    try:
        model = load_model('notebook/hybrid_model_best.h5')
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    if input_type == "Webcam":
        if st.button("Start Webcam"):
            process_webcam(model, confidence_threshold, alert_frames, show_fps)
            
    elif input_type == "Video Upload":
        video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        if video_file:
            process_video(video_file, model, confidence_threshold, alert_frames, show_fps)
            
    else:  # Image Upload
        image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        if image_file:
            process_image(image_file, model, confidence_threshold)

if __name__ == "__main__":
    main()