from flask import Flask, request, render_template, send_from_directory
import cv2
import os

app = Flask(__name__)

# Kaze detector code is here...
def match_images(image1, image2):
    # Initialize KAZE detector
    kaze = cv2.KAZE_create(threshold=0.001,  # Adjust the threshold
                       nOctaves=4,       # Number of octaves
                       nOctaveLayers=4,  # Number of layers per octave
                       diffusivity=cv2.KAZE_DIFF_PM_G2)  # Type of diffusivity
    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = kaze.detectAndCompute(image1, None)
    keypoints2, descriptors2 = kaze.detectAndCompute(image2, None)

    # Check if descriptors are not empty
    if descriptors1 is None or descriptors2 is None:
        return []

    # Initialize BFMatcher object with default params
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance: 
                good_matches.append(m)

    return good_matches

def find_best_match(fragment_path, images_folder):
    fragment = cv2.imread(fragment_path, 0)  # Read the fragment image in grayscale

    best_match = None
    max_matches = 0

    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path, 0)  # Read the image in grayscale

        matches = match_images(fragment, image)

        if len(matches) > max_matches:
            max_matches = len(matches)
            best_match = filename

    return best_match, max_matches

# app.py code from here....
@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        fragment_path = os.path.join('data/uploads', f.filename)
        f.save(fragment_path)

        images_folder = 'data/full_images'
        best_match, match_count = find_best_match(fragment_path, images_folder)
       
       # Check if best_match is None
        if best_match is None:
            best_match_path = "data/full_images/No Matching Found.png"  # Path to a default image or message
            match_count = 0
            # You might also want to add a message to inform the user that no match was found
            print("No Image Found")
        else:     
            best_match_path = os.path.join(images_folder, best_match)
        
        fragment_path = fragment_path.replace('\\', '/')
        best_match_path = best_match_path.replace('\\', '/')

        return render_template('results.html', fragment=fragment_path, best_match=best_match_path, match_count=match_count)
@app.route('/data/uploads/<filename>')
def send_upload(filename):
    return send_from_directory('data/uploads', filename)

@app.route('/data/full_images/<filename>')
def send_full_image(filename):
    return send_from_directory('data/full_images', filename)

if __name__ == '__main__':
    app.run(debug=True)
