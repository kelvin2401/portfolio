# Kelvin's Data Science Portfolio

Welcome to my data science portfolio. I'm passionate about exploring the world of data and leveraging it to gain insights and make predictions. Below, you'll find a collection of projects that demonstrate my skills in data scraping, exploratory data analysis, data preprocessing, and machine learning. I'm also excited to share my work in the fields of natural language processing, computer vision, and deep learning, which allows me to tackle more complex and diverse data challenges.

## Projects

### Home Credit Data Analysis and Risk Prediction

- Processed large Home Credit data that consists of 7 tables which needs to be aggregated and merged.
- Used Tableau to generate business insights and suggested to create marketing campaigns for accountants to take loans.
- Conducted experiments, such as:
  - Comparing models using only application table vs application + bureau + past application tables vs all tables.
  - Comparing models using all columns vs removing columns with >= 80% missing values.
  - Comparing models using all features vs using feature selection (removing features with high correlation and using variance thresholding).
  - Comparing models based on machine learning methods: Decision Tree, K Nearest Neighbors, Gaussian Naive Bayes, Random Forest, Logistic Regression and Light Gradient-Boosting Machine.
- Used ROC AUC score to evaluate models and chose the model which uses all tables, all features and LGBM Classifier as the best model.

### Telco Customer Segmentation and Churn Prediction

- Implemented K-means clustering for customer segmentation and visualized results in Tableau.
- Logistic regression identified as the most effective classifier for churn prediction, emphasizing the importance of Tenure Months.
- Advocated the promotion of call center services based on their demonstrated effectiveness in reducing churn.
- Highlighted students as a strategic target demographic, citing their high monthly consumption, extended tenure, and low churn rate.

### Chronic Disease Prediction using BPJS Dataset

- Processed a large BPJS dataset, which includes FKRTL, Non-Capitation FKTP, and Membership data.
- Developed predictive models using Decision Tree, Random Forest, and Multilayer Perceptron (Artificial Neural Network).
- Utilized web scraping techniques (Selenium and Scrapy) to extract hospital data from the BPJS website, enabling the ranking of provinces by their need for new hospitals.

### Clean vs Messy Room Image Classification with Deep Learning

- Employed Convolutional Neural Networks and transfer learning with VGG16, ResNet50, and Xception.
- Created a model for classifying room images as either clean or messy, showcasing the power of deep learning in image classification tasks.

### Architecture Image Classification

- Employed Feature Extraction Algorithms like SIFT, BRISK, and KAZE.
- Utilized Support Vector Machine (SVM) for classifying architectural images based on their features.

### Sports Ball Object Detection

- Scraped sports ball image data using Octoparse.
- Conducted image annotation using Roboflow to train an object detection model.

### Twitter Dangerous Speech Clustering, Classification, and Topic Modeling

- Extracted tweets using the snscrape library.
- Visualized data through word clouds.
- Employed text feature extraction techniques, including Word2Vec, TF-IDF, and CountVectorizer.

### Sentiment Analysis of Amazon Food Reviews

- Conducted sentiment analysis on Amazon Food Reviews.
- Explored various supervised learning methods, as well as VADER and RoBERTa models.

### All-NBA Team Prediction

- Employed SMOTE Oversampling and Grid Search Hyperparameter Tuning for predicting All-NBA Teams.

These projects represent my journey through the exciting world of data science, and I'm thrilled to continue learning and exploring new challenges. Feel free to explore each project for more details and insights into my data science journey.

Thank you for visiting my portfolio!

