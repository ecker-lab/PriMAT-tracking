import cv2
import numpy as np

# Function to read bounding boxes from the results.txt file
def read_bounding_boxes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    bounding_boxes = []
    for line in lines:
        values = line.strip().split(',')
        bounding_boxes.append(list(map(float, values)))
    
    return bounding_boxes

# Function to draw bounding boxes on frames
def draw_boxes_on_frames(video_path, bounding_boxes, names, output_path='/usr/users/vogg/monkey-tracking-in-the-wild/videos/lemur_ids/alpha_ind1/output.mp4'):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define colors for individuals
    colors = np.random.randint(0, 200, size=(len(names), 3), dtype=np.uint8)

    # Create VideoWriter object for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    frame_number = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get bounding boxes for the current frame
        frame_boxes = [box for box in bounding_boxes if box[0] == frame_number]

        # Draw bounding boxes on the frame
        for box in frame_boxes:
            x, y, w, h = map(int, box[2:6])
            individual_id = int(box[-1])
            color = tuple(map(int, colors[individual_id]))

            # Draw bounding box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 5)

            text_background_height = 25
            cv2.rectangle(frame, (int(x), int(y)),
              (int(x + 50), int(y + 25)), (255, 255, 255), -1)
            # Put the name in the top left corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, names[individual_id], (int(x), int(y) + 15), font, 0.8, color, 1, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

        # Increment frame number
        frame_number += 1

    # Release VideoCapture and VideoWriter objects
    cap.release()
    out.release()

if __name__ == "__main__":
    # Path to video file
    video_path = '/usr/users/agecker/datasets/lemur_videos_eval/Videos/alpha_ind1.mp4'

    # Path to bounding boxes file
    bounding_boxes_path = '/usr/users/vogg/monkey-tracking-in-the-wild/videos/lemur_ids/alpha_ind1/results-out.txt'

    # Names of individuals
    names = ['Cha', 'Flo', 'Gen', 'Geo', 'Her', 'Rab', 'Red', 'Uns']

    # Read bounding boxes from file
    bounding_boxes = read_bounding_boxes(bounding_boxes_path)

    # Draw bounding boxes on frames and save the output video
    draw_boxes_on_frames(video_path, bounding_boxes, names)
