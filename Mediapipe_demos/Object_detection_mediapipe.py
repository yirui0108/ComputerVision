import cv2
import mediapipe as mp

# Initialize MediaPipe Objectron and Drawing tools
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Define the Objectron solution with the desired object type
model_name = 'Cup'  # Change to 'Shoe', 'Chair', or 'Camera' as needed
objectron = mp_objectron.Objectron(
    static_image_mode=False,  # For live video, use False
    max_num_objects=5,        # Maximum objects to detect
    min_detection_confidence=0.3,  # Confidence threshold for detection
    min_tracking_confidence=0.5,   # Confidence threshold for tracking
    model_name=model_name
)

# Open laptop webcam (camera index 0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Objectron
    results = objectron.process(frame_rgb)

    # Draw detected objects and annotations on the frame
    if results.detected_objects:
        for detected_object in results.detected_objects:
            # Draw the 3D bounding box and landmarks
            mp_drawing.draw_landmarks(
                frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS
            )
            mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

            # Display the object type
            bbox = detected_object.landmarks_2d.landmark[0]
            cv2.putText(
                frame,
                f'{model_name}',  # Just display the model name
                (int(bbox.x * frame.shape[1]), int(bbox.y * frame.shape[0]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    # Display the annotated frame
    cv2.imshow("MediaPipe Objectron", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
