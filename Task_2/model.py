import cv2
import pickle

# Load images
img1_path = "c:/Users/kybav/OneDrive/Desktop/Home_Work/Home_work_5/Intership_test/Intership_test/Task_2/2024_august_png/august.png"
img2_path = "c:/Users/kybav/OneDrive/Desktop/Home_Work/Home_work_5/Intership_test/Intership_test/Task_2/2024_november_png/november.png"

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Verify that images loaded correctly
if img1 is None:
    print(f"Error: Could not load image at {img1_path}")
if img2 is None:
    print(f"Error: Could not load image at {img2_path}")

# Proceed only if images are loaded
if img1 is not None and img2 is not None:
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Save the model (keypoints and descriptors)
    keypoints1 = [
        (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp1
    ]
    keypoints2 = [
        (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2
    ]
    model_data = {
        "keypoints1": keypoints1,
        "descriptors1": des1,
        "keypoints2": keypoints2,
        "descriptors2": des2,
    }

    with open("sift_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Model saved as sift_model.pkl")
