# pygetwindow: Used for getting information about windows titles and dimensions.
# mss: Used for capturing vision inputs.
# io: Used to store the captured vision input in RAM without saving it as an image file on disk.
# win32gui: Provides a way to interact with native Windows GUI, used here for getting window dimensions and making a window active.
# win32con: Constants used for native Windows operations, used here to show a window.
# time: Required to fix the issue with taking a vision input in the middle of maximizing the window, introduces a short delay before taking a vision input.
# Pillow: Imports a vision input image file (vision input cache).

# You'll need the following libraries installed:
# pip install transformers
# pip install torch
# pip install pygetwindow
# pip install mss
# pip install pywin32
# pip install Pillow


# Things I might consider adding/fixing:
# - need to add support for Salesforce BLIP-2 models; 
# - need to add 'psutils' to specify the process and executable for filtering input solely from VRChat.exe, currently captures all windows titled exclusively with 'VRChat' in the title;


import io
import pygetwindow as gw
import mss
import win32gui
import win32con
import time
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Function to make a window active
def make_window_active(hwnd):
    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    win32gui.SetForegroundWindow(hwnd)

# Function to get the client area of a window
def get_client_area(hwnd):
    rect = win32gui.GetClientRect(hwnd)
    point = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
    return (point[0], point[1], rect[2] - rect[0], rect[3] - rect[1])

# Function to capture a vision input
def capture_vision_input(auto_detect=False):
    if auto_detect:
        for title in gw.getAllTitles():
            if "VRChat" in title:
                window_title = title
                break
        else:
            print("No window with 'VRChat' found.")
            return
    else:
        window_title = input("Enter the title of the window: ")

    window = gw.getWindowsWithTitle(window_title)

    if len(window) == 0:
        print(f"No window with title {window_title} found.")
        return

    window = window[0]
    hwnd = window._hWnd #handle to a window

    # Check if the window is maximized
    placement = win32gui.GetWindowPlacement(hwnd)
    if placement[1] != win32con.SW_SHOWMAXIMIZED:
        # Maximize the window
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        # Introduce a delay of 3 seconds
        time.sleep(3)

    # Make the window active
    make_window_active(hwnd)

    x, y, width, height = get_client_area(hwnd)

    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": width, "height": height}
        vision_input = sct.grab(monitor)
        img_bytes = mss.tools.to_png(vision_input.rgb, vision_input.size)
        vision_feed = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    print(f"Vision input of {window_title} captured and stored in RAM.")
    
    # Activate to minimize the window after vision input is captured (single monitor mode)
#    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)

    return vision_feed

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(MODEL_ID)
# by default `from_pretrained` loads the weights in float32
# we load in float16 instead to save memory
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
model.to(device)

while True:  # Vision Encoding Loop
    # Automatically detect 'VRChat' in window titles and captures a vision input
    vision_feed = capture_vision_input(auto_detect=True)

    if vision_feed is not None:
        print("Cache decoding. \nPlease wait...")
        
        # Image captioning
        inputs = processor(vision_feed, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"Encoded result: {generated_text}")

    # Wait 10 seconds before running again
    time.sleep(10)
