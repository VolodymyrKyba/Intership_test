import cv2
import pickle
import matplotlib.pyplot as plt

img1_path = "2024_august_png/august.png"  # Load images
img2_path = "2024_november_png/november.png"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None:
    print(f"Error loading {img1_path}")  # Check if images loaded correctly
if img2 is None:
    print(f"Error loading {img2_path}")

if img1 is not None and img2 is not None:
    with open("sift_model.pkl", "rb") as f:  # Load the model
        model_data = pickle.load(f)

    kp1 = [
        cv2.KeyPoint(
            x=pt[0][0],
            y=pt[0][1],
            _size=pt[1],
            _angle=pt[2],  # Reconstruct the keypoints
            _response=pt[3],
            _octave=pt[4],
            _class_id=pt[5],
        )
        for pt in model_data["keypoints1"]
    ]
    kp2 = [
        cv2.KeyPoint(
            x=pt[0][0],
            y=pt[0][1],
            _size=pt[1],
            _angle=pt[2],
            _response=pt[3],
            _octave=pt[4],
            _class_id=pt[5],
        )
        for pt in model_data["keypoints2"]
    ]

    des1 = model_data["descriptors1"]  # Load descriptors
    des2 = model_data["descriptors2"]

    bf = cv2.BFMatcher()  # Matching descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # Draw matches on the images
    img3 = cv2.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    output_path = "output_matches.png"  # Save the output image with matches to the specified directory
    cv2.imwrite(output_path, img3)  # Save the image in BGR format (OpenCV's default)

    plt.figure(
        figsize=(12, 6)
    )  # Optionally, display the image using Matplotlib, converting BGR to RGB for correct display
    plt.imshow(
        cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    )  # Convert BGR to RGB for correct colors
    plt.axis("off")  # Hide axis
    plt.show()

    print(f"Image saved to {output_path}")
