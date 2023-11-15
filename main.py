import base64
import requests
import os
from PIL import Image, ImageDraw
import json
import xml.etree.ElementTree as ET

# OpenAI API Key
api_key = "YOUR_API_KEY"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to perform object detection and labeling

def detect_objects(image_path, labels):
    label_list = ', '.join([f"'{label}'" for label in labels])
    prompt_text = (
        f"Identify and outline the entire body of any {label_list} in the image. "
        "Provide bounding boxes that encompass the full extent of each object, "
        "ensuring that all parts, such as the face, legs, and tail, are included within. "
        "Minimize the inclusion of background space as much as possible. "
        "Return the bounding box coordinates in a JSON array named 'results', "
        "with 'label' and 'coordinates' for each object. "
        "The 'coordinates' should detail 'x1', 'y1' (top-left corner), and 'x2', 'y2' "
        "(bottom-right corner) following the PIL draw method's requirements."
    )
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "system",
            "content": prompt_text
        },
        {
            "role": "user",
            "content": [

            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 700
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        res_json = response.json()["choices"][0]["message"]["content"].split("```json")[1].split("```")[0]
        print(res_json)
        result = json.loads(res_json)
        
        image = Image.open(image_path)
       
        original_width, original_height= image.size
        
        
        scaled_side = 768
        if (min(original_height, original_width) > scaled_side):
            combined_scale_factor = min(original_height, original_width) / scaled_side
        else:
            combined_scale_factor = 1

        for item in result["results"]:
            coords = item["coordinates"]
            item["coordinates"]["x1"] = int(coords["x1"] * combined_scale_factor)
            item["coordinates"]["y1"] = int(coords["y1"] * combined_scale_factor)
            item["coordinates"]["x2"] = int(coords["x2"] * combined_scale_factor)
            item["coordinates"]["y2"] = int(coords["y2"] * combined_scale_factor)
        image.save(image_path+"_annotated.jpg")

        print(result["results"])
        return result["results"]
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return []



# Function to save annotations in TensorFlow Object Detection format
def save_tf_annotations(image_path, objects, output_folder, image_width, image_height):
    image_filename = os.path.basename(image_path)
    annotation_filename = os.path.splitext(image_filename)[0] + ".xml"
    annotation_full_path = os.path.join(output_folder, annotation_filename)  # Full path including output_folder

    # Create an XML annotation file
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_filename

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")

    width.text = str(image_width)
    height.text = str(image_height)
    depth.text = "3"

    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj_elem, "name")
        name.text = obj['label']

        bbox = ET.SubElement(obj_elem, "bndbox")
        xmin = ET.SubElement(bbox, "xmin")
        ymin = ET.SubElement(bbox, "ymin")
        xmax = ET.SubElement(bbox, "xmax")
        ymax = ET.SubElement(bbox, "ymax")

        xmin.text = str(obj['coordinates']['x1'])
        ymin.text = str(obj['coordinates']['y1'])
        xmax.text = str(obj['coordinates']['x2'])
        ymax.text = str(obj['coordinates']['y2'])

    # Create and write the XML annotation file
    tree = ET.ElementTree(annotation)
    tree.write(annotation_full_path)

# Function to save annotations in JSON format
def save_json_annotations(image_path, objects, output_folder):
    image_filename = os.path.basename(image_path)
    annotation_filename = os.path.splitext(image_filename)[0] + ".json"
    annotation_path = os.path.join(output_folder, annotation_filename)

    # Create a dictionary to store JSON annotations
    json_annotations = {
        "filename": image_filename,
        "objects": objects
    }

    with open(annotation_path, "w") as annotation_file:
        json.dump(json_annotations, annotation_file, indent=4)

def process_images_in_folder(input_folder, output_format, output_folder,labels):
    # Updated with error handling
    labelMap = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):  # Supporting multiple image formats
            image_path = os.path.join(input_folder, filename)
            try:
                objects = detect_objects(image_path,labels)
            
                image = Image.open(image_path)
                
                if output_format == "tf":
                    #Extract the label from the object and add it to the labelMap if it is not already there with the id
                    for obj in objects:
                        label = obj["label"]
                        if label not in labelMap:
                            labelMap.append(label)
                    save_tf_annotations(image_path, objects, output_folder, image.width, image.height)

                elif output_format == "json":
                    save_json_annotations(image_path, objects, output_folder)
                else:
                    print(f"Invalid output format: {output_format}")
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
    if output_format == "tf":
        out= os.path.join(output_folder, "labelmap.pbtxt")
        labelMapFile = open(out, "w")
        
        for i in range(len(labelMap)):
            labelMapFile.write("item {\n")
            labelMapFile.write("  id: " + str(i+1) + "\n")
            labelMapFile.write("  name: '" + labelMap[i] + "'\n")
            labelMapFile.write("}\n")
            
            
# Main function 
def main():
    input_folder = "./test"
    output_format = "tf"  # Choose "tf" or "json"
    output_folder = "./test"
    labels = ['cat', 'dog', 'bird']

    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return
    process_images_in_folder(input_folder, output_format, output_folder,labels)

if __name__ == "__main__":
    main()