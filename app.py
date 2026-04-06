import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import tensorflow as tf  # 导入 tensorflow 用于加载 1D CNN 模型

# 初始化 Flask 应用
app = Flask(__name__)

# 加载训练好的模型
STAND_SVM_MODEL_PATH = 'D:/Desktop/cq_html_2026/predictor/optimized_stand_svm_pipeline.pkl'
MINMAX_SVM_MODEL_PATH = 'D:/Desktop/cq_html_2026/predictor/optimized_minmax_svm_pipeline.pkl'
STAND_RF_MODEL_PATH = 'D:/Desktop/cq_html_2026/predictor/optimized_stand_rf_pipeline.pkl'
MINMAX_RF_MODEL_PATH = 'D:/Desktop/cq_html_2026/predictor/optimized_minmax_rf_pipeline.pkl'
STAND_LSTM_MODEL_PATH = 'D:/Desktop/cq_html_2026/predictor/optimized_stand_lstm_model.h5'
MINMAX_LSTM_MODEL_PATH = 'D:/Desktop/cq_html_2026/predictor/optimized_minmax_lstm_model.h5'
STAND_CNN_MODEL_PATH = 'D:/Desktop/cq_html_2026/predictor/optimized_stand_cnn_model.h5'
MINMAX_CNN_MODEL_PATH = 'D:/Desktop/cq_html_2026/predictor/optimized_minmax_cnn_model.h5'

stand_svm_model = joblib.load(STAND_SVM_MODEL_PATH)
minmax_svm_model = joblib.load(MINMAX_SVM_MODEL_PATH)
stand_rf_model = joblib.load(STAND_RF_MODEL_PATH)
minmax_rf_model = joblib.load(MINMAX_RF_MODEL_PATH)
stand_lstm_model = tf.keras.models.load_model(STAND_LSTM_MODEL_PATH)
minmax_lstm_model = tf.keras.models.load_model(MINMAX_LSTM_MODEL_PATH)
stand_cnn_model = tf.keras.models.load_model(STAND_CNN_MODEL_PATH)
minmax_cnn_model = tf.keras.models.load_model(MINMAX_CNN_MODEL_PATH)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    return render_template('index.html')


# 提取特征函数
def extract_features(current_data, time_data):
    features = {
        'mean': np.mean(current_data),
        'std': np.std(current_data),
        'min': np.min(current_data),
        'median': np.median(current_data),
        'skewness': skew(current_data),
        'kurtosis': kurtosis(current_data),
        'rms': np.sqrt(np.mean(current_data ** 2)),
        'trend_slope': np.polyfit(np.arange(len(current_data)), current_data, 1)[0],
        'dominant_frequency': np.argmax(np.abs(fft(current_data))),
        'spectral_centroid': np.sum(np.arange(len(current_data)) * np.abs(fft(current_data))) / np.sum(np.abs(fft(current_data))),
        'peak_value': np.max(current_data)
    }

    return features


# 主函数（处理上传文件）
@app.route('/upload', methods=['POST'])
def upload_file():
    # 获取上传的文件、归一化方法和模型选择
    files = request.files.getlist('file')
    normalization_method = request.form.get('normalization')
    model_method = request.form.get('model')  # 用户选择的模型方法

    results = []

    for file in files:
        if file:
            # 读取 CSV 数据
            data = pd.read_csv(file, usecols=[3, 4])  # 假设时间在第4列，电流在第5列
            time_data = data.iloc[:, 0].values
            current_data = data.iloc[:, 1].values

            # 提取特征
            features = extract_features(current_data, time_data)

            # 将特征转化为特征向量
            feature_vector = np.array([
                features['mean'], features['std'], features['min'], features['median'],
                features['skewness'], features['kurtosis'], features['rms'], features['trend_slope'],
                features['dominant_frequency'], features['spectral_centroid']
            ]).reshape(1, -1)

            # 根据用户选择加载对应的模型
            if model_method == 'svm':
                if normalization_method == 'standard':
                    model = stand_svm_model
                elif normalization_method == 'minmax':
                    model = minmax_svm_model
            elif model_method == 'rf':
                if normalization_method == 'standard':
                    model = stand_rf_model
                elif normalization_method == 'minmax':
                    model = minmax_rf_model
            elif model_method == 'lstm':
                # LSTM 模型输入需要 reshape (1, 10, 1)
                feature_vector_reshaped = feature_vector.reshape((feature_vector.shape[0], feature_vector.shape[1], 1))
                if normalization_method == 'standard':
                    model = stand_lstm_model
                elif normalization_method == 'minmax':
                    model = minmax_lstm_model
            elif model_method == 'cnn':
                # CNN 模型输入需要 reshape (1, 10, 1)
                feature_vector_reshaped = feature_vector.reshape((feature_vector.shape[0], feature_vector.shape[1], 1))
                if normalization_method == 'standard':
                    model = stand_cnn_model
                elif normalization_method == 'minmax':
                    model = minmax_cnn_model
            else:
                return jsonify({"error": "Invalid model or normalization method selected."})

            # 使用训练好的模型进行预测
            if model_method == 'lstm' or model_method == 'cnn':
                predicted_values = model.predict(feature_vector_reshaped)
            else:
                predicted_values = model.predict(feature_vector)

            # 获取预测结果并确保转换为原生类型
            result = {
                'file_name': file.filename,
                'mean': round(float(features['mean']), 4),
                'std': round(float(features['std']), 4),
                'min': round(float(features['min']), 4),
                'median': round(float(features['median']), 4),
                'skewness': round(float(features['skewness']), 4),
                'kurtosis': round(float(features['kurtosis']), 4),
                'rms': round(float(features['rms']), 4),
                'trend_slope': round(float(features['trend_slope']), 4),
                'dominant_frequency': int(features['dominant_frequency']),
                'spectral_centroid': round(float(features['spectral_centroid']), 4),
                'peak_value': round(float(features['peak_value']), 4),
                'predicted_peak_value': round(float(predicted_values[0][0]), 4),
                'predicted_rise_rate': round(float(predicted_values[0][1]), 4)
            }

            results.append(result)

    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True)
