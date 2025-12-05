#!/usr/bin/env python3
"""
MQTT Subscriber for F.A.C.S. Recognition Results
Subscribes to the MQTT topic and displays formatted results.
"""

import json
import paho.mqtt.client as mqtt
from datetime import datetime
import sys

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "facs/results"

def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker."""
    if rc == 0:
        print(f"✓ Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
        print(f"✓ Subscribed to topic: {MQTT_TOPIC}")
        print("=" * 80)
        print("Waiting for messages... (Press Ctrl+C to exit)")
        print("=" * 80)
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"✗ Failed to connect, return code {rc}")
        sys.exit(1)

def format_emotion_data(emotions):
    """Format emotion data for display."""
    if not emotions:
        return "No emotions detected"
    
    output = []
    if 'top_3' in emotions:
        output.append("  Top 3 Emotions:")
        for i, emotion in enumerate(emotions['top_3'], 1):
            output.append(f"    {i}. {emotion['name']}: {emotion['score']:.4f}")
    
    if 'count' in emotions:
        output.append(f"  Total emotions detected: {emotions['count']}")
    
    return "\n".join(output)

def format_facs_data(facs):
    """Format F.A.C.S. Action Units for display."""
    if not facs:
        return "No F.A.C.S. data available"
    
    output = ["  F.A.C.S. Action Units:"]
    for au in facs[:10]:  # Show first 10 AUs
        output.append(f"    {au['action_unit']}: {au['evidence']:.4f}")
    
    if len(facs) > 10:
        output.append(f"    ... and {len(facs) - 10} more")
    
    return "\n".join(output)

def on_message(client, userdata, msg):
    """Callback when a message is received."""
    try:
        # Parse JSON payload
        payload = json.loads(msg.payload.decode())
        
        # Print header
        print("\n" + "=" * 80)
        timestamp = datetime.fromtimestamp(payload.get('batch_timestamp', 0))
        print(f"📊 New Analysis Batch - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Print batch info
        print(f"Interval: {payload.get('interval_seconds', 0)}s")
        print(f"Frames analyzed: {payload.get('frames_analyzed', 0)}")
        print(f"Frames buffered: {payload.get('frames_buffered', 0)}")
        
        # Process each result
        results = payload.get('results', [])
        print(f"\n📸 Processing {len(results)} face detection(s):")
        
        for idx, result in enumerate(results, 1):
            print(f"\n--- Face #{idx} ---")
            print(f"Face ID: {result.get('face_id', 'unknown')}")
            
            if 'detection_confidence' in result:
                print(f"Detection Confidence: {result['detection_confidence']:.4f}")
            
            if 'bounding_box' in result:
                bbox = result['bounding_box']
                print(f"Bounding Box: x={bbox.get('x')}, y={bbox.get('y')}, "
                      f"w={bbox.get('width')}, h={bbox.get('height')}")
            
            # Display emotions
            if 'emotions' in result:
                print(f"\n😊 Emotions:")
                print(format_emotion_data(result['emotions']))
            
            # Display F.A.C.S. data
            if 'facs' in result:
                print(f"\n🎭 F.A.C.S. Analysis ({result.get('facs_count', 0)} Action Units):")
                print(format_facs_data(result['facs']))
            
            # Display descriptions
            if 'descriptions' in result:
                print(f"\n📝 Descriptions:")
                for desc in result['descriptions'][:5]:  # Show first 5
                    print(f"  - {desc['name']}: {desc['score']:.4f}")
            
            # Timestamps
            if 'capture_timestamp' in result:
                cap_time = datetime.fromtimestamp(result['capture_timestamp'])
                print(f"\n⏱️  Captured: {cap_time.strftime('%H:%M:%S.%f')[:-3]}")
            
            if 'analysis_timestamp' in result:
                ana_time = datetime.fromtimestamp(result['analysis_timestamp'])
                print(f"   Analyzed: {ana_time.strftime('%H:%M:%S.%f')[:-3]}")
        
        print("\n" + "=" * 80)
        
    except json.JSONDecodeError as e:
        print(f"✗ Error decoding JSON: {e}")
        print(f"Raw message: {msg.payload.decode()}")
    except Exception as e:
        print(f"✗ Error processing message: {e}")

def on_disconnect(client, userdata, rc):
    """Callback when disconnected from MQTT broker."""
    if rc != 0:
        print(f"\n✗ Unexpected disconnection. Return code: {rc}")

def main():
    """Main function to run the MQTT subscriber."""
    print("🚀 Starting F.A.C.S. Recognition MQTT Subscriber")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Topic: {MQTT_TOPIC}\n")
    
    # Create MQTT client
    client = mqtt.Client()
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    try:
        # Connect to broker
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # Start loop
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down subscriber...")
        client.disconnect()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

