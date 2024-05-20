import cv2

image_paths = ['data/IMG1.jpg', 'data/IMG2.jpg', 'data/IMG3.jpg', 'data/IMG4.jpg', 'data/IMG5.jpg']
video_path = 'data/MOVIE.mp4'


def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Unable to load image {path}")
        else:
            images.append(img)
    return images


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to load video {video_path}. Check the file path and format.")
        return None
    print(f"Successfully loaded video: {video_path}")
    return cap


def detect_orb_features(images):
    orb = cv2.ORB_create()
    features = []
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        features.append((keypoints, descriptors))
        # Visualize keypoints on the original image
        img_with_keypoints = cv2.drawKeypoints(
            img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imshow(f'ORB Features - Image {idx + 1}', img_with_keypoints)
        cv2.waitKey(500)  # Display each image for 500 milliseconds
    cv2.destroyAllWindows()
    return features


def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return sorted(good_matches, key=lambda x: x.distance)


def find_best_match(frame_descriptors, image_features):
    best_image_idx = None
    best_matches = []
    for idx, (_, descriptors) in enumerate(image_features):
        if descriptors is not None:
            matches = match_features(frame_descriptors, descriptors)
            if len(matches) > len(best_matches):
                best_matches = matches
                best_image_idx = idx
    return best_image_idx, best_matches


def process_video(video, images, image_features):
    orb = cv2.ORB_create()
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

        best_image_idx, best_matches = find_best_match(frame_descriptors, image_features)
        if best_image_idx is not None:
            img_keypoints, _ = image_features[best_image_idx]
            best_image = images[best_image_idx]

            # Draw only the filtered matches
            img_with_keypoints = cv2.drawMatches(
                frame, frame_keypoints, best_image, img_keypoints,
                best_matches, None,
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
            )

            cv2.imshow('Best Match', img_with_keypoints)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def main():
    images = load_images(image_paths)
    if not images:
        print("No images loaded. Check the file paths.")
        return

    video = load_video(video_path)
    if video is None:
        print("No video loaded. Check the file path.")
        return

    orb_features = detect_orb_features(images)
    print("Features extracted from the images using ORB.")

    process_video(video, images, orb_features)


if __name__ == '__main__':
    main()
