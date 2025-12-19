
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime
from scipy import stats
import statsmodels.api as sm

st.set_page_config(
    page_title="Анализ температурных данных",
    layout="wide"
)

st.title("Анализ климатических данных и мониторинг температуры")

st.sidebar.header("Данные и настройки")

uploaded_file = st.sidebar.file_uploader("Загрузите temperature_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
else:
    st.warning("Загрузите файл с данными")
    st.stop()

city_list = sorted(data['city'].unique())
selected_city = st.sidebar.selectbox("Выберите город", city_list)

api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")

enable_parallel = st.sidebar.checkbox("Включить параллельную обработку", value=False)
window_size = st.sidebar.slider("Размер окна для скользящего среднего", 7, 90, 30)

def calculate_moving_statistics(city_data, window=30):
    city_data = city_data.sort_index().copy()
    
    city_data['rolling_mean'] = city_data['temperature'].rolling(
        window=window, center=True, min_periods=1
    ).mean()
    
    city_data['rolling_std'] = city_data['temperature'].rolling(
        window=window, center=True, min_periods=1
    ).std()
    
    city_data['upper_bound'] = city_data['rolling_mean'] + 2 * city_data['rolling_std']
    city_data['lower_bound'] = city_data['rolling_mean'] - 2 * city_data['rolling_std']
    
    return city_data

def detect_anomalies(city_data):
    city_data = city_data.copy()
    city_data['anomaly'] = (
        (city_data['temperature'] > city_data['upper_bound']) |
        (city_data['temperature'] < city_data['lower_bound'])
    )
    return city_data

def get_current_weather_sync(city_name, api_key):
    if not api_key:
        return None, "API ключ не указан"
    
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric',
        'lang': 'ru'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            weather_data = response.json()
            return {
                'temperature': weather_data['main']['temp'],
                'description': weather_data['weather'][0]['description'],
                'city': weather_data['name'],
                'country': weather_data['sys']['country']
            }, None
        elif response.status_code == 401:
            return None, "Неверный API ключ"
        else:
            return None, f"Ошибка: {response.status_code}"
    except:
        return None, "Ошибка подключения"

st.header(f"Анализ данных для {selected_city}")

city_data = data[data['city'] == selected_city].copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "Временной ряд", 
    "Аномалии", 
    "Текущая погода", 
    "Статистика"
])

with tab1:
    st.subheader("Временной ряд температуры")
    
    city_data_with_stats = calculate_moving_statistics(city_data, window_size)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=city_data_with_stats.index,
        y=city_data_with_stats['temperature'],
        mode='lines',
        name='Температура',
        line=dict(color='lightblue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=city_data_with_stats.index,
        y=city_data_with_stats['rolling_mean'],
        mode='lines',
        name=f'Скользящее среднее ({window_size} дней)',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=city_data_with_stats.index.tolist() + city_data_with_stats.index.tolist()[::-1],
        y=city_data_with_stats['upper_bound'].tolist() + city_data_with_stats['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Диапазон ±2σ'
    ))
    
    fig.update_layout(
        title=f'Температурный ряд: {selected_city}',
        xaxis_title='Дата',
        yaxis_title='Температура (°C)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Средняя температура", f"{city_data['temperature'].mean():.1f}°C")
    with col2:
        st.metric("Максимальная", f"{city_data['temperature'].max():.1f}°C")
    with col3:
        st.metric("Минимальная", f"{city_data['temperature'].min():.1f}°C")
    with col4:
        st.metric("Стандартное отклонение", f"{city_data['temperature'].std():.1f}°C")

with tab2:
    st.subheader("Температурные аномалии")
    
    city_data_with_anomalies = detect_anomalies(city_data_with_stats)
    anomaly_count = city_data_with_anomalies['anomaly'].sum()
    
    fig2 = go.Figure()
    
    normal_data = city_data_with_anomalies[~city_data_with_anomalies['anomaly']]
    fig2.add_trace(go.Scatter(
        x=normal_data.index,
        y=normal_data['temperature'],
        mode='markers',
        name='Нормальная температура',
        marker=dict(color='blue', size=3, opacity=0.3)
    ))
    
    anomaly_data = city_data_with_anomalies[city_data_with_anomalies['anomaly']]
    fig2.add_trace(go.Scatter(
        x=anomaly_data.index,
        y=anomaly_data['temperature'],
        mode='markers',
        name='Аномалии',
        marker=dict(color='red', size=6, symbol='diamond')
    ))
    
    fig2.update_layout(
        title=f'Температурные аномалии: {selected_city}',
        xaxis_title='Дата',
        yaxis_title='Температура (°C)',
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего аномалий", anomaly_count)
    with col2:
        st.metric("Процент аномалий", f"{(anomaly_count/len(city_data)*100):.2f}%")
    with col3:
        st.metric("Ожидаемый процент", "~4.55%")

with tab3:
    st.subheader("Текущая погода")
    
    if api_key:
        if st.button("Получить текущую погоду"):
            with st.spinner("Запрос данных..."):
                weather_data, error = get_current_weather_sync(selected_city, api_key)
                
                if error:
                    st.error(f"Ошибка: {error}")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Текущая температура", f"{weather_data['temperature']}°C")
                        st.write(f"Описание: {weather_data['description']}")
                        st.write(f"Город: {weather_data['city']}, {weather_data['country']}")
                    
                    with col2:
                        current_month = datetime.now().month
                        month_to_season = {
                            12: "winter", 1: "winter", 2: "winter",
                            3: "spring", 4: "spring", 5: "spring",
                            6: "summer", 7: "summer", 8: "summer",
                            9: "autumn", 10: "autumn", 11: "autumn"
                        }
                        current_season = month_to_season.get(current_month, "unknown")
                        
                        season_data = city_data[city_data['season'] == current_season]
                        
                        if len(season_data) > 0:
                            mean_temp = season_data['temperature'].mean()
                            std_temp = season_data['temperature'].std()
                            lower_bound = mean_temp - 2 * std_temp
                            upper_bound = mean_temp + 2 * std_temp
                            
                            is_normal = lower_bound <= weather_data['temperature'] <= upper_bound
                            
                            st.write(f"Текущий сезон: {current_season}")
                            st.write(f"Средняя за сезон: {mean_temp:.1f}°C")
                            st.write(f"Нормальный диапазон: {lower_bound:.1f}°C - {upper_bound:.1f}°C")
                            
                            if is_normal:
                                st.success("Температура в пределах нормы")
                            else:
                                st.warning("Аномальная температура")
    else:
        st.warning("Введите API ключ для получения текущей погоды")

with tab4:
    st.subheader("Статистика по сезонам")
    
    seasonal_stats = city_data.groupby('season').agg({
        'temperature': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    seasonal_stats.columns = ['_'.join(col).strip() for col in seasonal_stats.columns.values]
    seasonal_stats = seasonal_stats.reset_index()
    
    st.dataframe(seasonal_stats)
    
    fig3 = go.Figure()
    
    season_order = ['winter', 'spring', 'summer', 'autumn']
    seasonal_stats_ordered = seasonal_stats.set_index('season').reindex(season_order).reset_index()
    
    fig3.add_trace(go.Bar(
        x=seasonal_stats_ordered['season'],
        y=seasonal_stats_ordered['temperature_mean'],
        error_y=dict(
            type='data',
            array=seasonal_stats_ordered['temperature_std'],
            visible=True
        ),
        name='Средняя температура ± σ',
        marker_color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    ))
    
    fig3.update_layout(
        title=f'Сезонный температурный профиль: {selected_city}',
        xaxis_title='Сезон',
        yaxis_title='Температура (°C)',
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.header("Информация")
