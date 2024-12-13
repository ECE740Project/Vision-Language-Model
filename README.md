# Vision-Language-Model
## Pre-requisited 

1. Install [docker](https://docs.docker.com/engine/install/) in the host system.

## Setup

1. Clone this repository
2. Change your current directory to this repository
3. Run Docker Engine.
4. Then run `docker-compose up -d`
5. Download the [model](https://drive.google.com/file/d/1VlTykaTBh2ETfy7tQPK5AA77fOMLKJZ5/view?usp=sharing) in the repository directory

Then navigate to the jupyter notebook in the docker container and go to `./notebooks/Tranformer Model.ipynb` ,then change the image directories that suits your paths

```
object_dir = '<Path to target directory>'
background_dir = '<Path to background>'
original_image_dir = '<Coco 2017 training dataset path>'

// Change the model path as well
model.load_weights("./weights-improvement-48-0.00.keras")
```
Run all the cells except `model.fit()`