# Predicting Bitcoin Price Movements Using Sentiment Analysis

### **Capstone Project - Fall 2024**

**Authors:**
- Alibi Nauanov
- Douaa Zouhir
- Jalal Haider

**Supervisor:**
- Xianbin Gu

---

## Overview

This project integrates social media sentiment analysis and financial metrics to predict Bitcoin price movements. By leveraging advanced machine learning models, we developed a predictive framework that combines sentiment scores with financial indicators for enhanced market trend forecasting.

---

## Key Features
- **Sentiment Analysis Models**: Utilized VADER, TextBlob, and BERT for robust sentiment scoring.
- **Ensemble Learning**: Combined sentiment models using Random Forest for improved accuracy.
- **Time-Series Modeling**: Implemented LSTM to capture temporal patterns in Bitcoin price trends.
- **Comprehensive Dataset**: Analyzed over 285,000 tweets alongside Bitcoin price data spanning January 31, 2023, to June 6, 2023.

---

## Technologies Used
- **Python**
- **TensorFlow**
- **Scikit-learn**
- **BERT**
- **VADER**
- **TextBlob**
- **Pandas**
- **Matplotlib**
- **LSTM (Long Short-Term Memory)**
- **Random Forest**

---

## Methodology
1. **Data Collection**  
   - Gathered datasets from Hugging Face and Kaggle.
   - Integrated Bitcoin price data (open, close, high, low prices) with tweet sentiment data.
2. **Sentiment Analysis**  
   - Analyzed tweet sentiment using VADER, TextBlob, and BERT.
   - Generated metrics such as polarity, subjectivity, and compound scores.
3. **Feature Engineering**  
   - Merged sentiment scores with Bitcoin price metrics.
   - Sequenced data for LSTM model input, capturing 10-day patterns.
4. **Model Training**  
   - Trained LSTM model on 80% of the dataset; tested on 20%.
   - Incorporated ensemble learning to combine predictions.

---

## Results
- **Performance**: Achieved 78% accuracy with the ensemble model, outperforming individual sentiment models.
- **Visualization**: Generated graphs comparing predicted vs. actual Bitcoin prices, highlighting model effectiveness.
- **Key Insights**: BERT-based sentiment scores demonstrated superior accuracy among individual tools.

---

## Limitations
- Limited adaptability to rapid market shifts or ambiguous language in social media posts.
- Dependency on historical price data for predictions.

---


## Access the Full Paper
[View the PDF](https://github.com/alibinauanov/Bitcoin-Price-Prediction-Using-Sentiment-Analysis-and-Ensemble-Methods/blob/main/Capstone%20Report%20Fall%202024%20CSDSE.pdf)
