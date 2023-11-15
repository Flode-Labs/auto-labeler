
![Auto Labeler](./Auto%20Labeler.png)

# Auto Labeler

Auto Labeler is an automated image annotation tool that leverages the power of OpenAI's GPT-4 Vision API to detect objects in images and provide bounding box annotations.

## Features

- Object Detection: Automatically identifies objects in images.
- Bounding Box Annotations: Generates bounding boxes around detected objects.
- Multiple Formats: Supports saving annotations in both TensorFlow Object Detection and JSON formats.
- Custom Labels: Allows for specifying custom object labels for detection.

## Getting Started

### Prerequisites

- Python 3.6 or higher

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/Auto-Labeler.git
```


### Usage

1. Set your OpenAI API key:

```python
api_key = "YOUR_API_KEY"
```

2. Specify the path of the folder with the images that you want to label:

```python
input_folder = '/path/to/input/folder'
```

3. Specify the path of the folder where you want to save the labels:

```python
output_folder = '/path/to/output/folder'
```

4. Specify the format in which you want to save the labels:

```python
output_format = 'json' # or 'tf'
```

5. Define the list of labels you want to detect:

```python
labels = ['cat', 'dog', 'bird']
```

6. Run the main script:

```bash
python main.py
```




### Functions

- `encode_image(image_path)`: Encodes the image to base64 format.
- `detect_objects(image_path, labels)`: Detects objects in the image as specified by labels.
- `save_tf_annotations(...)`: Saves annotations in TensorFlow Object Detection XML format.
- `save_json_annotations(...)`: Saves annotations in JSON format.
- `process_images_in_folder(...)`: Processes multiple images in a specified folder.


## Acknowledgments

- OpenAI for providing the GPT-4 Vision API.
- The Python community for maintaining and improving the libraries used in this project.

---

Ensure you replace `YOUR_API_KEY` with your actual OpenAI API key.
