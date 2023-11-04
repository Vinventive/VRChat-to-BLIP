import io
import pygetwindow as gw
import mss
import win32gui
import win32con
import time
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, Blip2ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

# Function to make a window active
def make_window_active(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        print(f"Error: Application window isn't accessible or active. {e}")

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
            return None
    else:
        window_title = input("Enter the title of the window: ")

    window = gw.getWindowsWithTitle(window_title)
    if len(window) == 0:
        print(f"No window with title {window_title} found.")
        return None

    window = window[0]
    hwnd = window._hWnd  # handle to a window

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

    # print(f"Vision input of {window_title} captured and stored in RAM.")
    return vision_feed

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID, torch_dtype=torch.float16)

# Load BLIP-2 model and processor (24GB VRAM HIGH-END GPU recommended)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# BLIP_MODEL_ID = "Salesforce/blip2-opt-2.7b"
# blip_processor = AutoProcessor.from_pretrained(BLIP_MODEL_ID)
# blip_model = Blip2ForConditionalGeneration.from_pretrained(BLIP_MODEL_ID, torch_dtype=torch.float16)

blip_model.to(device)

# Load quantized LLAMA-2 based GPTQ model and tokenizer
LLM_MODEL_ID = "TheBloke/Llama-2-7B-Chat-GPTQ"
# LLM_MODEL_ID = "TheBloke/Llama-2-13B-Chat-GPTQ" (24GB VRAM HIGH-END GPU recommended)
# LLM_MODEL_ID = "TheBloke/vicuna-7B-v1.5-GPTQ"

llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, device_map="auto", trust_remote_code=True)
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, use_fast=True)

def embed_blip_into_llm(image):
    # Process the image through BLIP
    blip_inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
    blip_generated_ids = blip_model.generate(**blip_inputs, max_new_tokens=20)
    blip_generated_text = blip_processor.batch_decode(blip_generated_ids, skip_special_tokens=True)[0].strip()
    
    # Prepare the prompt for LLM with instructions to add context and factual descriptions
    contextual_instruction = (
        "You are an AI capable of providing rich factual descriptions and background information. "
        "Using your comprehensive knowledge, describe the objects in the following image, "
        "adding any relevant historical, scientific, or cultural context. Provide facts and well-informed insights "
        "where applicable. Do not fabricate information or speculate beyond what can be reasonably inferred. "
        "If there isn't enough information, say 'That's all I can tell about it.'"
    )
    prompt_template = f"{contextual_instruction}\n\n\"I can see {blip_generated_text}.\"\n\n"
    
    # Process the BLIP output through LLM with a very low temperature to stick to the facts
    input_ids = llm_tokenizer(prompt_template, return_tensors='pt').input_ids.to(device)
    llm_output = llm_model.generate(input_ids=input_ids, temperature=0.1, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512, repetition_penalty=1.1)
    llm_generated_text = llm_tokenizer.decode(llm_output[0], skip_special_tokens=True)
    
    # Extract only the "I can see" part from the LLM output
    start_phrase = '"I can see'
    end_phrase = '."\n\n'
    start_index = llm_generated_text.find(start_phrase)
    end_index = llm_generated_text.find(end_phrase, start_index)
    output_embedded = llm_generated_text[start_index:end_index+len(end_phrase)].strip()
    
    # Print only the "I can see" part, including any context or assumptions
    print(output_embedded)

    # Return the full processed text in case it's needed for something else
    return llm_generated_text


# Main loop
while True:  # Vision Encoding Loop
    vision_feed = capture_vision_input(auto_detect=True)
    if vision_feed is not None:
        embed_blip_into_llm(vision_feed)
    # Wait 5 seconds before running again
    time.sleep(5)
