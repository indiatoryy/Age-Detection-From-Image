# Race-Specific Convolutional Neural Networks for Accurate and Fair Juvenile Face Recognition

# Introduction

Facial age estimation plays a pivotal role in various applications, yet achieving accurate results, especially with juvenile subjects and diverse racial backgrounds, remains a challenge. Existing algorithms struggle when only facial images are available, and incorporating race-specific information has been overlooked in many models. This project presents an innovative approach utilizing race-specific Convolutional Neural Networks (CNNs) to improve the accuracy, fairness, and inclusivity of facial age estimation algorithms.

## 1. Model Architecture and Training

The proposed model architecture consists of three convolutional layers with increasing channels, enabling the network to learn intricate facial features critical for precise age estimation. Max pooling layers efficiently extract features, and four fully connected layers capture abstract representations of data. The model outputs age estimates in 17 bins, each corresponding to different age groups. Hyperparameters, including 40 epochs, a batch size of 120, learning rate of 0.001, and Cross Entropy Loss with Rectified Linear Unit (ReLU) activation, were tuned for optimal performance.

<img width="981" alt="Screenshot 2023-10-22 at 12 02 12 PM" src="https://github.com/indiatoryy/Age-Detection-From-Image/assets/105636722/4f9f1134-4a5e-41ab-9a9f-457ee386f91e">

## 2. Datasets
In this project, two primary datasets were employed to create a comprehensive and diverse dataset for training and evaluating the facial age estimation model: UTKFace and FaceARG.

**1. UTKFace:**
UTKFace is a widely-used database containing 20,000 facial images spanning a vast age range from 0 to 116 years. Each image in the UTKFace dataset is meticulously labeled with age, gender, race, and the time the photograph was collected. This dataset serves as a foundational component, providing a substantial number of images for training and validating the age estimation model. 

**2. FaceARG:**
FaceARG is a supplementary dataset comprising 175,000 labeled facial images. Similar to UTKFace, FaceARG includes annotations for age, race, and gender, making it a valuable addition to the project. The dataset's substantial size enriches the diversity of the training data, ensuring that the model encounters a wide array of facial features, expressions, and ages. 

**Dataset Processing and Balance:**
To enhance the consistency and balance within the dataset, the project contributors carefully curated the images from both UTKFace and FaceARG. Efforts were made to address disparities in the number of datapoints across different racial categories. By strategically combining images from both datasets, the final dataset achieved improved equilibrium in terms of race representation. Additionally, the dataset focused on emphasizing images within the 10-25 age range, aligning with the project's primary objective of juvenile age estimation.

## 3. Discussion and Evaluation

The study rigorously compares the performance of the Combined Race Model (considering all races) with race-specific models. Results demonstrate that race-specific CNNs outperform the combined approach, showcasing improved accuracy, reduced loss, and error. Testing accuracies exhibit higher consistency within race-specific models, enhancing fairness. Despite challenges, such as the Black category's lower performance, the race-specific approach outperforms existing algorithms. Mean Absolute Error (MAE) evaluation places the model's accuracy favorably among state-of-the-art models, especially for ages 10-25, even with a smaller dataset size.

## 4. Conclusion and Future Research

This research underscores the importance of race-specific CNNs in facial age estimation, highlighting the need to account for unique facial aging patterns across different races. The project challenges the traditional one-size-fits-all approach and advocates for tailored models based on race and gender. Future research avenues include refining the proposed architecture, exploring combined gender and race models, and investigating the causes of performance differences among racial categories. This approach represents a significant step towards creating inclusive, accurate, and fair facial recognition technology, with applications in crucial areas such as child protection.

## 5. Usage

Researchers and developers keen on advancing facial age estimation technology can seamlessly integrate the proposed race-specific CNNs into their projects. By implementing this approach, users can achieve enhanced accuracy and fairness, thereby contributing to the development of more equitable facial recognition systems.

## Acknowledgments

We express our gratitude to the contributors and researchers whose work in the fields of machine learning and facial recognition paved the way for this project. Their efforts continue to inspire and drive advancements in technology and inclusivity.

## Contributors
This report was completed for the U of T Engineering Science course ECE353 (Machine Intelligence, Software & Neural Network). This report was co-written by India Tory, Nida Copty, and Thomas Nguyen.

## Further Reading
Please explore the attached files in this repository for access to a complete research report and the complete codebase and datasets.
