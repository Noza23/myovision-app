# myovision-app

[![LMU: Munich](https://img.shields.io/badge/LMU-Munich-009440.svg)](https://www.en.statistik.uni-muenchen.de/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This is a sub-project of the main project [myovision](https://github.com/Noza23/myovision).
It provides a Backend for the Application developed for the convenient use and further development of the myovision project.
It is a RESTful API developed with the FastAPI framework and has a designated Frontend for the user interaction [myovision-app-front](https://github.com/davitchanturia/myovision-app-front)

With a slight modification the Backend can be generalized and used for any computer vision project concerning instance segmentation.

## Contact Information

The Backend as well as the Frontend won't be published open-source for now, Contact the maintainer for the further information or if you are interested in the project.

```python
{
    name = "Giorgi Nozadze",
    email = "giorginozadze23@yahoo.com"
}
```

## Visualizations

- Annotation Tool for collecting labeled data
  [Video](https://drive.google.com/file/d/1JFWEre71lWuu_wAtUcogsJ7cMXfV57Or/view?usp=sharing)

- Inference Tool for observing model's predictions and relevant metrics
  [Video](https://drive.google.com/file/d/1JFWEre71lWuu_wAtUcogsJ7cMXfV57Or/view?usp=sharing)

# Installation

- Local Setup

```bash
mv .env.example .env
uvicorn backend.main:app --reload
```

- Docker Build

```bash
docker build -t myovision-app .
docker run -d --name myovision-app -p 8000:8000 myovision-app
```

- Docker Compose

```bash
docker-compose up -d
```
