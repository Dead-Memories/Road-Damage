import os
import pandas as pd
import folium
from folium import IFrame
import base64

def visualize_defects(csv_path: str,
                      photo_dir: str = 'outputs',
                      output_html: str = 'Report_test.html') -> None:

    # 1) Читаем и сразу парсим timestamp как datetime
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    # 2) Группируем по кадру (filename):
    #    - берём первую по времени точку для lat/lon/timestamp
    #    - собираем все дефекты в строку через запятую
    df_grouped = (
        df
        .sort_values('timestamp')  
        .groupby('filename', as_index=False)
        .agg({
            'lat':       'first',
            'lon':       'first',
            'timestamp': 'first',
            'class':      lambda x: ', '.join(sorted(set(x)))
        })
        .rename(columns={'class': 'defects'})
    )

    # 3) Cортируем уже сгруппированный DF по timestamp
    df_grouped = df_grouped.sort_values('timestamp').reset_index(drop=True)

    # 4) Центр карты
    center_lat = df_grouped['lat'].mean()
    center_lon = df_grouped['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # 5) Рисуем маршрут 
    coords = df_grouped[['lat', 'lon']].values.tolist()
    folium.PolyLine(coords, weight=3, opacity=0.7).add_to(m)

    # 6) И добавляем маркеры
    for _, row in df_grouped.iterrows():
        photo_path = os.path.join(photo_dir, row['filename'])
        if not os.path.exists(photo_path):
            print(f"Warning: File not found: {photo_path}")
            continue

        with open(photo_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        ext = os.path.splitext(photo_path)[1].lstrip('.')
        data_uri = f"data:image/{ext};base64,{img_data}"

        frame_name = os.path.splitext(row['filename'])[0]

        html = f"""
        <strong>Frame:</strong> {frame_name}<br>
        <strong>Time:</strong> {row['timestamp']}<br>
        <strong>Defects:</strong> {row['defects']}<br>
        <img src="{data_uri}" alt="defect" width="200">
        """
        iframe = IFrame(html, width=260, height=330)
        popup = folium.Popup(iframe, max_width=2650)

        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=popup,
            icon=folium.Icon(icon='exclamation-triangle', prefix='fa')
        ).add_to(m)

    # 7) Сохраняем
    m.save(output_html)
    print(f'Map saved to {output_html}')

visualize_defects(
    '/Users/olya/Desktop/Road Detection/reports/detections_copy.csv',
    '/Users/olya/Desktop/Road Detection/outputs',
    'Report_test.html'
)
