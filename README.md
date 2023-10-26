# []
This project uses []. The derived information is then used to [] from the NTU-RGBD dataset.

## Classifications
Hello! I have removed lots of private infomation from this repository in order to make this public. Anything listed as "[]" is redacted infomation.

## Prerequisites
The project requires Python and specific Python libraries. Clone the repository and navigate to the project's root directory. Install the necessary Python libraries using the requirements.txt file included in the repository:

## Copy code
```pip install -r requirements.txt```
## Dataset
This project utilizes the NTU-RGBD dataset. Please download this dataset separately.

## Instructions
1. Clone the Repository: Start by cloning this repository to your local machine.

2. Dataset Preparation: Replace the txt2npy.py file in the NTU-RGBD dataset repository with the txt2npy.py file from this repository.

3. Data Path Modification: In the pre_process.py file of this repository, update the data_path variable to point to the raw_npy folder inside your local NTU-RGBD dataset repository.

4. Pre-processing: Run the pre_process.py script. This is used to standardize the distribution of video frames across the dataset, ensuring a stable token count. Videos with too few or too many frames will be removed. The script will generate a filtered.csv file and a summary.csv file. These files hold the paths to each video and record the number of frames each video has.

5. []: Run the [].py script. This will create the [] used to encode the [].

6. Transformer Training: Run the [].py script. This will retrieve data from the [] file, load the [], and use the []. The [] includes a [] that are designed to predict the []. After training, the script will extract the [].
