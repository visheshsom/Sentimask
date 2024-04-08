# Sentimask

## Overview
Sentimask is a Django-based web application designed for real-time face mask detection. Utilizing the YOLOv5 model, it provides fast and accurate detection capabilities. This project integrates with a SQLite database for data management, making it suitable for deployment in environments where face mask compliance needs to be monitored.

## Features
- **Real-time Mask Detection**: Leveraging the YOLOv5 object detection model for high accuracy and performance.
- **Data Management**: Integrated with SQLite for efficient data storage and retrieval.
- **Web Application**: Built with Django, offering a robust framework for web applications.

## Installation

### Prerequisites
- Python 3.8+
- Django 3.2+
- YOLOv5 dependencies

### Setup
1. Clone the repository and its submodules using the following command:

    ```shell
    git clone <repository-url> --recurse-submodules
    ```

2. Navigate to the project directory:

    ```shell
    cd Sentimask
    ```

3. Install dependencies:

    ```shell
    pip install -r requirements.txt
    ```

4. Migrate the database:

    ```shell
    python manage.py migrate
    ```

5. Run the server:

    ```shell
    python manage.py runserver
    ```

Access the web application at `http://localhost:8000`.

## Usage
Explain how to use the application, including any web interfaces and how to perform detections.

## Contributing
Contributions to Sentimask are welcome! Please read the contributing guidelines before submitting pull requests.

## License
Specify the license under which the project is made available.

## Acknowledgements
- YOLOv5 for the object detection model.
- Django for the web application framework.
