import asyncio
import cv2
import json
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import tempfile

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from hume import AsyncHumeClient
from hume.expression_measurement.stream.stream.types.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

HUME_API_KEY = os.getenv("HUME_API_KEY")
RTSP_URL = os.getenv("RTSP_URL")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "facs/results")
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# Analysis configuration
ANALYSIS_INTERVAL = float(os.getenv("ANALYSIS_INTERVAL", "2.0"))  # seconds between analyses
FRAME_BUFFER_SIZE = int(os.getenv("FRAME_BUFFER_SIZE", "10"))  # number of frames to keep

# Global MQTT Client
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    """Callback for when the MQTT client connects to the broker."""
    if rc == 0:
        logger.info("Connected to MQTT Broker")
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code {rc}")

mqtt_client.on_connect = on_connect

if MQTT_USERNAME and MQTT_PASSWORD:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)


class RTSPStream:
    """Manages RTSP video stream capture."""
    
    def __init__(self, url):
        """Initialize the RTSP stream.
        
        Args:
            url: The RTSP stream URL
        """
        self.url = url
        self.cap = None
        self.running = False
        
    def start(self):
        """Initialize and start the RTSP stream capture.
        
        Returns:
            bool: True if stream started successfully, False otherwise
        """
        self.running = True
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {self.url}")
            return False
        logger.info("RTSP Stream started")
        return True

    def stop(self):
        """Stop and release the RTSP stream capture."""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("RTSP Stream stopped")

    def read_frame(self):
        """Read a single frame from the stream.
        
        Returns:
            numpy.ndarray or None: The frame if successful, None otherwise
        """
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None


class FrameBuffer:
    """Manages a buffer of recent frames for batch analysis."""
    
    def __init__(self, max_size=10):
        """Initialize the frame buffer.
        
        Args:
            max_size: Maximum number of frames to keep in buffer
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_frame(self, frame):
        """Add a frame to the buffer.
        
        Args:
            frame: The frame to add
        """
        self.buffer.append({
            'frame': frame,
            'timestamp': time.time()
        })
    
    def get_frames(self):
        """Get all frames from the buffer.
        
        Returns:
            list: List of frame dictionaries
        """
        return list(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def is_full(self):
        """Check if buffer is full.
        
        Returns:
            bool: True if buffer is at max capacity
        """
        return len(self.buffer) >= self.max_size


def extract_facs_data(face_data):
    """Extract comprehensive F.A.C.S. (Facial Action Coding System) data.
    
    Args:
        face_data: Face prediction data from Hume API
        
    Returns:
        dict: Comprehensive F.A.C.S. and emotion data
    """
    result = {}
    
    # Extract basic identification
    result['face_id'] = face_data.face_id if hasattr(face_data, 'face_id') else "unknown"
    
    # Extract bounding box if available
    if hasattr(face_data, 'bbox'):
        bbox = face_data.bbox
        result['bounding_box'] = {
            'x': bbox.x if hasattr(bbox, 'x') else None,
            'y': bbox.y if hasattr(bbox, 'y') else None,
            'width': bbox.w if hasattr(bbox, 'w') else None,
            'height': bbox.h if hasattr(bbox, 'h') else None
        }
    
    # Extract emotions with full details
    if hasattr(face_data, 'emotions'):
        emotions = face_data.emotions
        sorted_emotions = sorted(
            emotions,
            key=lambda x: x.score if hasattr(x, 'score') else 0,
            reverse=True
        )
        
        result['emotions'] = {
            'top_3': [
                {
                    'name': e.name if hasattr(e, 'name') else str(e),
                    'score': round(e.score, 4) if hasattr(e, 'score') else 0
                }
                for e in sorted_emotions[:3]
            ],
            'all_emotions': [
                {
                    'name': e.name if hasattr(e, 'name') else str(e),
                    'score': round(e.score, 4) if hasattr(e, 'score') else 0
                }
                for e in sorted_emotions
            ],
            'count': len(emotions)
        }
    
    # Extract F.A.C.S. Action Units if available
    if hasattr(face_data, 'facs'):
        facs = face_data.facs
        result['facs'] = [
            {
                'action_unit': au.name if hasattr(au, 'name') else str(au),
                'evidence': round(au.score, 4) if hasattr(au, 'score') else 0
            }
            for au in facs
        ]
        result['facs_count'] = len(facs)
    
    # Extract facial landmarks/descriptions if available
    if hasattr(face_data, 'descriptions'):
        descriptions = face_data.descriptions
        result['descriptions'] = [
            {
                'name': desc.name if hasattr(desc, 'name') else str(desc),
                'score': round(desc.score, 4) if hasattr(desc, 'score') else 0
            }
            for desc in descriptions
        ]
    
    # Extract frame information
    if hasattr(face_data, 'frame'):
        result['frame_number'] = face_data.frame
    
    if hasattr(face_data, 'time'):
        result['frame_time'] = face_data.time
    
    # Extract confidence/probability
    if hasattr(face_data, 'prob'):
        result['detection_confidence'] = round(face_data.prob, 4)
    
    return result


async def analyze_frame(hume_socket, frame, frame_timestamp):
    """Analyze a single frame and return detailed F.A.C.S. data.
    
    Args:
        hume_socket: Connected Hume websocket
        frame: The frame to analyze
        frame_timestamp: Timestamp when frame was captured
        
    Returns:
        dict or None: Analysis results or None if analysis failed
    """
    try:
        # Resize frame for bandwidth optimization (640px width)
        height, width = frame.shape[:2]
        new_width = 640
        new_height = int(new_width * height / width)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Encode to JPEG with 70% quality for bandwidth efficiency
        success, buffer = cv2.imencode('.jpg', resized_frame, 
                                      [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        
        if not success:
            logger.warning("Failed to encode frame")
            return None
        
        # Create temporary file for the frame
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(buffer.tobytes())
            tmp_filepath = tmp_file.name
        
        try:
            # Create config for face detection with F.A.C.S. enabled
            config = Config(face={
                "identify_faces": True,
                "facs": {},  # Enable F.A.C.S. Action Units
                "descriptions": {}  # Enable facial descriptions
            })
            
            # Send file to Hume streaming API with face config
            result = await hume_socket.send_file(tmp_filepath, config=config)
            
            # Process results
            if result and hasattr(result, 'face') and result.face:
                face_predictions = result.face.predictions if hasattr(result.face, 'predictions') else []
                
                if face_predictions:
                    # Extract comprehensive F.A.C.S. data from first detected face
                    facs_data = extract_facs_data(face_predictions[0])
                    facs_data['capture_timestamp'] = frame_timestamp
                    facs_data['analysis_timestamp'] = time.time()
                    
                    return facs_data
            
            elif isinstance(result, dict):
                # Handle dictionary response format (backward compatibility)
                predictions = result.get("face", {}).get("predictions", [])
                if predictions:
                    facs_data = extract_facs_data(predictions[0])
                    facs_data['capture_timestamp'] = frame_timestamp
                    facs_data['analysis_timestamp'] = time.time()
                    return facs_data
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_filepath)
            except Exception as cleanup_error:
                logger.debug(f"Error cleaning up temp file: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}", exc_info=True)
        return None


async def process_video_stream():
    """Main processing loop for video stream analysis using Hume SDK v0.13.5."""
    if not HUME_API_KEY:
        logger.error("HUME_API_KEY is missing")
        return

    # Initialize RTSP stream
    stream = RTSPStream(RTSP_URL)
    if not stream.start():
        return

    # Initialize frame buffer
    frame_buffer = FrameBuffer(max_size=FRAME_BUFFER_SIZE)
    
    # Thread pool for blocking OpenCV operations
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Track last analysis time
    last_analysis_time = 0

    try:
        # Initialize Hume async client (v0.13.5 API)
        client = AsyncHumeClient(api_key=HUME_API_KEY)
        
        logger.info(f"Connecting to Hume Expression Measurement Stream...")
        logger.info(f"Analysis interval: {ANALYSIS_INTERVAL}s, Frame buffer size: {FRAME_BUFFER_SIZE}")
        
        # Connect to Hume streaming API (v0.13.5)
        async with client.expression_measurement.stream.connect() as hume_socket:
            logger.info("Connected to Hume Stream")
            
            loop = asyncio.get_running_loop()
            
            while True:
                start_time = time.time()
                
                # Read frame in separate thread to avoid blocking event loop
                frame = await loop.run_in_executor(executor, stream.read_frame)
                
                if frame is None:
                    logger.warning("Failed to read frame, retrying connection...")
                    stream.stop()
                    await asyncio.sleep(5)
                    if not stream.start():
                        logger.error("Failed to restart RTSP stream")
                        break
                    continue

                # Add frame to buffer
                frame_buffer.add_frame(frame)
                
                # Check if it's time to analyze
                time_since_last_analysis = start_time - last_analysis_time
                
                if time_since_last_analysis >= ANALYSIS_INTERVAL:
                    logger.info(f"Starting analysis batch ({len(frame_buffer.get_frames())} frames buffered)")
                    
                    # Analyze all buffered frames
                    frames_to_analyze = frame_buffer.get_frames()
                    analysis_results = []
                    
                    for idx, frame_data in enumerate(frames_to_analyze):
                        logger.debug(f"Analyzing frame {idx + 1}/{len(frames_to_analyze)}")
                        result = await analyze_frame(
                            hume_socket,
                            frame_data['frame'],
                            frame_data['timestamp']
                        )
                        
                        if result:
                            analysis_results.append(result)
                    
                    # Publish aggregated results
                    if analysis_results:
                        # Create comprehensive payload
                        payload = {
                            "batch_timestamp": time.time(),
                            "interval_seconds": ANALYSIS_INTERVAL,
                            "frames_analyzed": len(analysis_results),
                            "frames_buffered": len(frames_to_analyze),
                            "results": analysis_results
                        }
                        
                        # Publish to MQTT
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload, indent=2))
                        
                        # Log summary
                        if analysis_results:
                            first_result = analysis_results[0]
                            if 'emotions' in first_result and 'top_3' in first_result['emotions']:
                                top_emotion = first_result['emotions']['top_3'][0]
                                logger.info(
                                    f"Published batch analysis: {len(analysis_results)} faces detected, "
                                    f"Top emotion: {top_emotion['name']} ({top_emotion['score']:.2f})"
                                )
                                
                                # Log F.A.C.S. info if available
                                if 'facs' in first_result:
                                    logger.info(f"F.A.C.S. Action Units detected: {first_result['facs_count']}")
                    
                    # Clear buffer and update last analysis time
                    frame_buffer.clear()
                    last_analysis_time = start_time
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"Fatal error in processing loop: {e}", exc_info=True)
    finally:
        stream.stop()
        executor.shutdown(wait=True)
        logger.info("Video processing stopped")


def main():
    """Main entry point for the application."""
    # Start MQTT Client
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        logger.info("MQTT client started")
    except Exception as e:
        logger.error(f"Could not connect to MQTT Broker: {e}")
        return

    # Start async processing loop
    try:
        asyncio.run(process_video_stream())
    except KeyboardInterrupt:
        logger.info("Stopping application...")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        logger.info("Application stopped")


if __name__ == "__main__":
    main()
