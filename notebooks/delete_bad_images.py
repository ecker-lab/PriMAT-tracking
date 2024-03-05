import os
import cv2
import keyboard
import pandas as pd



def draw_on_image(image, label_path, ending = ".jpg"):
    """
    img_id: name of the image file (without ending)
    img_path: path to the images
    label_path: path to the labels
    ending: type of image .jpg, .png, .JPG
    """

    class_names = ['Cha', 'Flo', 'Gen', 'Geo', 'Her', 'Rab', 'Red', 'Uns']

    coords = pd.read_csv(label_path, header = None, sep = " ")
    
    im_h, im_w = image.shape[:2]

    for i in range(len(coords)):
        center = tuple(map(int, (im_w * coords.iloc[i, 2], im_h * coords.iloc[i, 3])))
        low_left = tuple(map(int, (im_w * (coords.iloc[i, 2] - coords.iloc[i, 4]/2),
                                   im_h * (coords.iloc[i, 3] - coords.iloc[i, 5]/2))))
        up_right = tuple(map(int, (im_w * (coords.iloc[i, 2] + coords.iloc[i, 4]/2), 
                                 im_h * (coords.iloc[i, 3] + coords.iloc[i, 5]/2))))

        color = (i * 100 % 255, i * 75 % 255, i * 50 % 255)
        image = cv2.circle(image, center, radius = 2, color = color, thickness = 5)
        image = cv2.rectangle(image, low_left, up_right, color = color, thickness = 5)

        text = class_names[coords.iloc[i, 6]]  # Assuming the text is in the 7th column
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255)  # White text color
        text_position = (low_left[0], low_left[1] - 5)  # Slightly above the top-left corner
        image = cv2.putText(image, text, text_position, font, font_scale, text_color, font_thickness)

        
    return(image)

def main():
    # Specify the path to the folder containing images and labels
    folder_path = "/usr/users/vogg/Labelling/Lemurs/Individual_imgs/cleaned_labels1/"

    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path+"images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, "images", image_file)
        label_path = os.path.join(folder_path, "labels_with_ids", image_file.replace(os.path.splitext(image_file)[-1], '.txt'))

        # Check if the label file exists
        if not os.path.exists(label_path):
            print(f"Label file not found for {image_file}")
            continue

        # Read the image
        image = cv2.imread(image_path)

        # Draw bounding boxes on the image
        image = draw_on_image(image, label_path)


        cv2.putText(image, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Show the image
        cv2.imshow("Image", image)

        # Wait for user input
        key = cv2.waitKey(0)

        # Check if the user pressed the 'd' key to delete the image and label
        if keyboard.is_pressed('d'):
            os.remove(image_path)
            os.remove(label_path)
            print(f"Deleted {image_file}")

        # Check if the user pressed any other key to move to the next image
        else:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
