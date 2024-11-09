import subprocess
import os

# Define the path to your Python executable and the script
python_executable = r"C:/Users/kalya/Desktop/Desktop/ML/ML_Env/Scripts/python.exe"
script_path = "C:/Users/kalya/Desktop/Desktop/ML/lama/bin/predict.py"

# Define the arguments
model_path = os.path.join(os.getcwd(), "big-lama").replace("\\","/")
indir = os.path.join(os.getcwd(), "no_bbox_ILSVRC2012_val_00000002.jpeg").replace("\\","/")
outdir = os.path.join(os.getcwd(), "output").replace("\\","/")

# Create the command to run
command = [
    python_executable,
    script_path,
    f"model.path={model_path}",
    f"indir={indir}",
    f"outdir={outdir}"
]

# Execute the command
try:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:", result.stdout)  # Print standard output
    print("Error:", result.stderr)    # Print standard error (if any)
except subprocess.CalledProcessError as e:
    print("An error occurred while executing the command.")
    print("Return Code:", e.returncode)
    print("Output:", e.stdout)
    print("Error:", e.stderr)



# c:/Users/kalya/Desktop/Desktop/ML/ML_Env/Scripts/python.exe c:/Users/kalya/Desktop/Desktop/ML/lama/lama/bin/predict.py model.path=C:/Users/kalya/Desktop/Desktop/ML/lama/lama/bin/models/big-lama/ indir=C:/Users/kalya/Desktop/Desktop/ML/lama/lama/masks outdir=C:/Users/kalya/Desktop/Desktop/ML/lama/lama/output


# all should be in .png
# dont remove the objects from original image
# after selecting bounding box just create the mask with bounding box coordinates as a separate image and dont touch the original image

# image augumentation before creating masks and create the new version of the same image
# automatic bounding box on the augumented image and remove how much even objects and create all these masks in the same mask image file
# then give this mask and image as input to the model
# 
# automatic bounding box ``
# 
#  


