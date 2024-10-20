# Gender Classifier and Age Detector

## Project Overview

This project is a real-time **Gender Classifier and Age Detector** that operates on a live webcam feed. The system uses computer vision techniques to detect faces, classify gender (male or female), and estimate the age of the individual. When a face is detected, a square is drawn around it, along with the estimated gender and age displayed above the bounding box. This project is designed to run smoothly in real-time, making it suitable for live demonstrations and practical applications.

## Key Features

- **Real-time Face Detection**: Detects human faces using live webcam input.
- **Gender Classification**: Classifies the detected face as either male or female.
- **Age Estimation**: Provides an estimate of the individual’s age based on the detected facial features.
- **Face Bounding Box**: Draws a square around the detected face in real-time for easy identification.
- **Accurate and Efficient**: Designed to process live video input efficiently without major delays.

## Technologies Used

The following technologies and libraries were used to implement the project:

- **OpenCV**: For real-time face detection and video stream handling.
- **Dlib**: For face detection and alignment.
- **TensorFlow/Keras**: Used for training and running the gender classifier and age detector models.
- **Pre-Trained Models**: Leveraged pre-trained deep learning models for age estimation and gender classification.
- **NumPy**: For data manipulation and numerical processing.
- **Python**: The core programming language for integrating the components and running the application.

## Model Implementation

### Face Detection
- **Haar Cascades** from OpenCV were used to detect faces in each frame of the video stream. This method allows the software to identify human faces by recognizing specific patterns and features.

### Gender Classification
- A **Convolutional Neural Network (CNN)** was trained to classify gender based on facial features. The model outputs either "Male" or "Female" based on the live input.

### Age Estimation
- A separate **CNN model** was trained on age datasets to predict the approximate age range of the individual. The output is an integer representing the predicted age.

### Live Webcam Feed
- The system captures frames from the webcam, processes each frame for face detection, and classifies the detected face for gender and age. The results are displayed directly on the video feed with a square around the face, including a label for gender and age.

## Installation

To run this project on your local machine, follow the instructions below:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repository/gender-age-detector.git
    ```

2. **Install the necessary dependencies**

3. **Run the application**:
    ```bash
    python gender_age_detector.py
    ```

## Usage

1. **Start the webcam feed**: After running the script, the system will access your webcam.
2. **Real-time detection**: The software will detect faces in real-time, drawing a square around each detected face.
3. **Gender and age output**: Above the face bounding box, the estimated gender and age will be displayed.

## System Architecture

1. **Webcam Input**: Captures frames from a live webcam feed.
2. **Face Detection**: Uses Haar cascades to identify faces in each frame.
3. **Gender and Age Classification**: The detected faces are fed into pre-trained CNN models for gender classification and age detection.
4. **Bounding Box & Labels**: A square is drawn around the face, with the gender and age label displayed on top.

## Example Output

When running the program, the output will display a live webcam feed with the following:
- A square around each detected face.
- Gender and age estimations for each detected face, e.g., "Male, Age: 25".

## Future Improvements

- **Multiple Faces**: Extend functionality to detect and classify multiple faces simultaneously in the same frame.
- **Mobile Integration**: Deploy the system on mobile devices for real-time classification on the go.
- **Model Optimization**: Further train the models for more precise age estimations and gender classifications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- **Ziad Amer** - Lead Developer and Machine Learning Engineer

## Acknowledgments

Thanks to the developers of OpenCV, Dlib, and TensorFlow for providing the necessary tools and libraries to build this project.
