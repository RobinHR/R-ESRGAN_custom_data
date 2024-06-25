# Real-ESRGAN Training with Custom Dataset
Welcome to this comprehensive guide on using the Real-ESRGAN technique with your own dataset! If you're ready to enhance your images using this powerful method, you've come to the right place. Here's everything you need to get started:

Prerequisites
Before we dive into the steps, make sure you have the following:

- GitHub Account
- PyCharm
- Python version 3.9
- CUDA 11.1.0
- Torch 1.9.0
- Torchvision 0.10.0
- Torchaudio 0.9.0

# Step by step guide
Step 1: Clone the Repository
First, clone the Real-ESRGAN repository from GitHub:
```
git clone https://github.com/xinntao/Real-ESRGAN.git
```

Step 2: Navigate to the Directory
Change your directory to the cloned repository:
```
cd Real-ESRGAN
```

Step 3: Install Required Packages
Install the necessary packages by running:
```
pip install -r requirements.txt
pip install --user torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python setup.py develop
```

Step 4: Prepare Your Data
Collect your data. For instance, I used a script to save episodes of a Star Trek Episode as JPG files. Below is the script I used to extract frames from a video and save them as images. You can create a new script in the directory of Real-ESRGAN named extract_images:
```
import cv2
import os
def video_to_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame {frame_number}.")
            break

        frame_filename = os.path.join(output_path, f"image{frame_number}.jpg")
        cv2.imwrite(frame_filename, frame)

        print(f"Frame {frame_number} saved as {frame_filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = r"FILL THIS WITH YOUR VIDEOFILE AND CORRECT PATH"
    output_directory = os.path.dirname(video_file)
    os.makedirs(output_directory, exist_ok=True)
    video_to_frames(video_file, output_directory)
```
Ensure your images are named image0.jpg, image1.jpg, etc.

Step 5: Create Folders for Your Dataset
Create new directories for your dataset within the project:
1. Right-click on the Real-ESRGAN folder -> New -> Directory.
2. Name the directory datasets.
3. Inside datasets, create another directory named own_dataset_root.
4. Within own_dataset_root, create a directory named own_images.

Step 6: Generate Lower Resolution Images (Multiscale)
Use the following command to create lower resolution versions of your images:
```
python scripts/generate_multiscale_DF2K.py --input datasets/own_dataset_root/own_images --output datasets/own_dataset_root/own_images_multiscale
```

Step 7: Create Meta Information Text File
Run the following command to generate a text file containing a list of image file paths:
```
python scripts/generate_meta_info.py --input datasets/own_dataset_root/own_images datasets/own_dataset_root/own_images_multiscale --root datasets/own_dataset_root datasets/own_dataset_root --meta_info datasets/own_dataset_root/meta_info/meta_info_own_imagesmultiscale.txt
```

Step 8: Download Pre-trained Models
Download the pre-trained models using these commands:
```
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
```

Step 9: Update Configuration File
Edit the finetune_realesrgan_x4plus.yml file located in the Options folder. Update the datasets section with your dataset paths:
```
datasets:
  train:
    name: own_dataset
    type: RealESRGANDataset
    dataroot_gt: datasets/own_dataset_root
    meta_info: datasets/own_dataset_root/meta_info/meta_info_own_imagesmultiscale.txt
    io_backend:
      type: disk
```

Step 10: Update Pre-trained Model Path
In the same finetune_realesrgan_x4plus.yml file, update the path section from 'pretrain_network_g' to RealESRGAN_x4plus.pth. 
First you should have this:
```
path:
  # use the pre-trained Real-ESRNet model
  pretrain_network_g: experiments/pretrained_models/RealESRNet_x4plus.pth
```
You have to change that to this:
```
path:
# use the pre-trained Real-ESRNet model
  RealESRGAN_x4plus.pth: experiments/pretrained_models/RealESRNet_x4plus.pth
```

Step 11: Train the Model
Run the training script with the following command:
```
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --auto_resume
```

Your model will now begin training! The duration of this process depends on the number of images in your dataset.

Should you have any issues when you train to train the training script like 'The paging file is too small' or 'cuDNN error: CUDNN_STATUS_INTERNAL_ERROR' you should try lower the num_worker
This should also be in the finetune_realesrgan_x4plus.yml file. Try changing it to this:
```
    # data loader
    use_shuffle: true
    num_worker_per_gpu: 1
    batch_size_per_gpu: 4
    dataset_enlarge_ratio: 1
    prefetch_mode: ~
```

And there you have it! Follow these steps to successfully train Real-ESRGAN with your custom dataset. Happy enhancing!


