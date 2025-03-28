#%% Imports packages

import numpy as np
import sys
import os
import pims
from tqdm import tqdm
import cv2

#%% add directories

sys.path.insert(0, 'Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Analysis\\movement_control_analysis')
import facemap_utils

dataPath='Z:\\home\\shared\\Gaia\\Coliseum\\Delays\\paper_code\\Datasets\\movement_control_datasets\\raw_data'
videoFolders = os.listdir(dataPath)   

#%% Step 1 - downsample the videos 

i=int(list(sys.argv)[1])
print(i)

this_path = os.path.join(dataPath,videoFolders[i])
print(this_path)
sub_folder = os.listdir(this_path)
number_item = next((item for item in sub_folder if item.isdigit()), None)

this_path2 = os.path.join(dataPath,videoFolders[i],number_item)

# List files in the subfolder
files_in_subfolder = os.listdir(this_path2)

# Find the .avi video file in the folder
video_files = [f for f in files_in_subfolder if f.endswith('.avi')]

my_file = video_files[0]
input_video_path = os.path.join(this_path2, my_file)
output_video_path = os.path.join(this_path,'downsample_video.avi') # in the main path

# Downsampling factor
downsample_factor_x = 2  # Downsample in the X dimension
downsample_factor_y = 2 # Downsample in the Y dimension

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video was loaded successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate new dimensions after downsampling
new_width = frame_width // downsample_factor_x
new_height = frame_height // downsample_factor_y


print(f"Original dimensions: {frame_width}x{frame_height}")
print(f"Downsampled dimensions: {new_width}x{new_height}")
print(f"FPS: {fps}, Total frames: {frame_count}")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG for better compatibility
out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height), isColor=False)

# Process each frame
for i in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i} could not be read. Exiting.")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame (downsample)
    resized_frame = cv2.resize(gray_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Write the frame to the output video
    out.write(resized_frame)

# Release the video capture and writer objects
cap.release()
out.release()

#%% Step 2 - perform SVD

ncomps = 100
# Load videos

i = int(list(sys.argv)[1])
print(f"Processing video index: {i}")

this_path = os.path.join(dataPath, videoFolders[i])
sub_folder = os.listdir(this_path)

# Find the .avi video file in the folder
video_files = [f for f in sub_folder if f.endswith('.avi')]
video_path = os.path.join(this_path, video_files[0])

print(f"Loading video: {video_path}")

# Open video using cv2
cap = cv2.VideoCapture(video_path)

# Get video properties
Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video dimensions: {Ly}x{Lx}, Total frames: {nframes}")

# Early exit for short videos
if nframes < 200:
    print("Video is too short, skipping.")
    sys.exit(1)

# Compute avgframe and avgmotion
nf = min(2000000, nframes)
nt0 = min(2000, nframes)
nsegs = int(np.floor(nf / nt0))
tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

avgframe = np.zeros((Ly, Lx), np.float32)
avgmotion = np.zeros((Ly, Lx), np.float32)
ns = 0

print("Calculating average frame and motion...")

for n in tqdm(range(nsegs)):
    t = tf[n]

    # Set the video position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    frames = []

    # Read nt0 frames
    for _ in range(nt0):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    # Stack frames into a single array (Time x Ly x Lx)
    im = np.stack(frames, axis=0)

    # Convert to float and transpose to (Ly, Lx, Time)
    im = np.transpose(im, (1, 2, 0)).astype(np.float32)

    # Add to averages
    avgframe += im.mean(axis=-1)
    immotion = np.abs(np.diff(im, axis=-1))
    avgmotion += immotion.mean(axis=-1)
    ns += 1

avgframe /= float(ns)
avgmotion /= float(ns)

# Perform chunked SVD and save U0

print('Performing SVD...')
nc = 50  # Number of components

U = np.zeros((Ly * Lx, nsegs * nc), np.float32)

for n in tqdm(range(nsegs)):
    t = tf[n]

    # Set the video position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    frames = []

    # Read nt0 frames
    for frame_idx in range(nt0):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {t + frame_idx}")
            break
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    # Ensure frames were read
    if len(frames) == 0:
        print(f"No frames read for segment {n + 1}/{nsegs} starting at frame {t}. Skipping.")
        continue

    # Stack frames into a single array (Time x Ly x Lx)
    im = np.stack(frames, axis=0)

    # Convert to float and transpose to (Ly, Lx, Time)
    im = np.transpose(im, (1, 2, 0)).astype(np.float32)

    # Reshape to (Ly * Lx, Time)
    im = np.reshape(im, (Ly * Lx, -1))

    # Compute motion by taking frame-to-frame differences
    im = np.abs(np.diff(im, axis=-1))

    # Subtract average motion (flattened)
    im -= avgmotion.flatten()[:, np.newaxis]

    # Perform SVD decomposition (facemap_utils.svdecon)
    usv = facemap_utils.svdecon(im, k=nc)

    # Store the U matrix in U0
    U[:, n * nc : (n + 1) * nc] = usv[0]


# save U as well
output_file = os.path.join(this_path, 'U_video.npy')
np.save(output_file, U)

print(f"Saved final U matrix to {output_file}")

#%% Step 3 - project spatial PCs onto movies

# Load U - the masks

U = np.load(os.path.join(this_path, 'U_video.npy'))
ncomps = 100
# Perform a second SVD on U to reduce to ncomps components
print("Performing second SVD to reduce to ncomps...")
USV = facemap_utils.svdecon(U, k=ncomps)
U_reduced = USV[0]  # Take the U matrix after reduction

# Save the reduced U matrix
output_file = os.path.join(this_path, 'U_video_reduced.npy')
np.save(output_file, U_reduced)

print(f"Saved reduced U matrix to {output_file}")

# Find the .avi video file in the folder
video_files = [f for f in sub_folder if f.endswith('.avi')]
video_path = os.path.join(this_path, video_files[0])

print(f"Loading video: {video_path}")

# Open video using cv2
cap = cv2.VideoCapture(video_path)

# Get video properties
Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video dimensions: {Ly}x{Lx}, Total frames: {nframes}")

# Early exit for short videos
if nframes < 200:
    print("Video is too short, skipping.")
    sys.exit(1)

# Compute avgframe and avgmotion
nf = min(2000000, nframes)
nt0 = min(2000, nframes)
nsegs = int(np.floor(nf / nt0))
tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

avgframe = np.zeros((Ly, Lx), np.float32)
avgmotion = np.zeros((Ly, Lx), np.float32)
ns = 0

print("Calculating average frame and motion...")

for n in tqdm(range(nsegs)):
    t = tf[n]

    # Set the video position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    frames = []

    # Read nt0 frames
    for _ in range(nt0):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    # Stack frames into a single array (Time x Ly x Lx)
    im = np.stack(frames, axis=0)

    # Convert to float and transpose to (Ly, Lx, Time)
    im = np.transpose(im, (1, 2, 0)).astype(np.float32)

    # Add to averages
    avgframe += im.mean(axis=-1)
    immotion = np.abs(np.diff(im, axis=-1))
    avgmotion += immotion.mean(axis=-1)
    ns += 1

avgframe /= float(ns)
avgmotion /= float(ns)

# project spatial PCs onto movies (in chunks, for each video, and save)

print('projecting onto movies')

motSVD = np.zeros((nframes, ncomps), np.float32)
nt0 = min(1000, nframes)
nsegs = int(np.ceil(nframes / nt0))

# time ranges
itimes = np.floor(np.linspace(0, nframes, nsegs + 1)).astype(int)

for n in tqdm(range(nsegs)):
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, itimes[n])  # Set the starting frame
    for _ in range(itimes[n], itimes[n + 1]):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(frame_gray)

    if len(frames) == 0:
        continue

    im = np.stack(frames, axis=-1).astype(np.float32)  # Ly x Lx x TIME
    im = np.reshape(im, (Ly * Lx, -1))

    # we need to keep around the last frame for the next chunk
    if n > 0:
        im = np.concatenate((imend[:, np.newaxis], im), axis=-1)
    imend = im[:, -1]
    im = np.abs(np.diff(im, axis=-1))

    # subtract off average motion
    im -= avgmotion.flatten()[:, np.newaxis]

    # project U onto immotion
    vproj = im.T @ U_reduced
    if n == 0:
        vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)

    motSVD[itimes[n] : itimes[n + 1], :] = vproj

np.save(os.path.join(dataPath, videoFolders[i], 'motSVD.npy'), motSVD)
    
print('done')

# Reshape U to generate motion masks

motMask = np.reshape(U_reduced, (Ly, Lx, ncomps))

# Normalize each motion mask
motMask0 = np.array([(motMask[:, :, j] / motMask[:, :, j].std()) for j in range(ncomps)])

# Save the final body motion masks
output_file = os.path.join(this_path, 'MotionSVD_masks.npy')
np.save(output_file, motMask0)

print(f"Saved final motion masks to {output_file}")