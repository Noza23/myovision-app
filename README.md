# myovision-app

[![LMU: Munich](https://img.shields.io/badge/LMU-Munich-009440.svg)](https://www.en.statistik.uni-muenchen.de/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This is a sub-project of the main project [myovision](https://github.com/Noza23/myovision).
It provides a Backend for the Application developed for the convenient use and further development of the myovision project.
It is a RESTful API developed with the FastAPI framework and has a designated Frontend for the user interaction [myovision-app-front](https://github.com/davitchanturia/myovision-app-front)

With a slight modification the Backend can be generalized and used for any computer vision project concerning instance segmentation.

## Visualizations

- Annotation Tool for collecting labeled data
  [Video](https://drive.google.com/file/d/1JFWEre71lWuu_wAtUcogsJ7cMXfV57Or/view?usp=sharing)

- Inference Tool for observing model's predictions and relevant metrics
  [Video](https://drive.google.com/file/d/1JFWEre71lWuu_wAtUcogsJ7cMXfV57Or/view?usp=sharing)

# Setup

To setup the Application backend locally follow the steps:

## 1. Get Model Checkpoint
  - Get the model checkpoint from [link](https://drive.google.com/file/d/1wAlAgqo_NCNnrE8zjQFIkHXpLhTjg3fs/view)
  - Place it in the *./checkpoints/* directory

## 2. Install Dependencies
  - Install python dependencies using: ```pip install -r requirements.txt```
  - Install redis: follow the simple instructions based on your OS [link](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/)


## 3. Set Environment variables
  - Copy the example .env file: ```cp .env-example .env```
  - Adjust the default variables in *.env* file if desired


# Starting Application

1. **Start Redis**: follow the simple instructions based on your OS [link](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/)

2. **Start API**:
    ```shell
    uvicorn backend.main.app
    ```

3. **Start Frontend**: Follow the [Instructions](https://github.com/davitchanturia/myovision-app-front?tab=readme-ov-file#setup)

Finally, open the browser go to the url the frontend is running.

## Using Container Images

Both [Backend](https://github.com/Noza23/myovision-app/blob/main/Dockerfile) and [Frontend] contian Dockerfile to build docker images using: ```docker build``` ([docs](https://docs.docker.com/reference/cli/docker/image/build/)).


it can then be conveniently managed with docker-compose: ```docker-compose up``` ([docs](https://docs.docker.com/reference/cli/docker/compose/up/)).


## Contact Information

```json
{
    name = "Giorgi Nozadze",
    email = "giorginozadze23@yahoo.com"
}
```
