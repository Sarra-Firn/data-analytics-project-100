#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from dotenv import load_dotenv


# In[2]:


visits_df = pd.read_csv("visits.csv")
registrations_df = pd.read_csv("registrations.csv")


# In[3]:


print("Визиты:")
print(visits_df.head())
print(visits_df.describe(include='all'))

print("\nРегистрации:")
print(registrations_df.head())
print(registrations_df.describe(include='all'))


# In[4]:


print(visits_df.isnull().sum())
print(registrations_df.isnull().sum())


# In[5]:


import pandas as pd
import requests
from datetime import datetime


# In[6]:


# Задаем даты
start_date = "2023-03-01"
end_date = "2023-09-01"

# Базовый URL
base_url = "https://data-charts-api.hexlet.app"


# In[7]:


visits_url = f"{base_url}/visits?begin={start_date}&end={end_date}"
visits_response = requests.get(visits_url)

# Проверка статуса
if visits_response.status_code == 200:
    visits_data = visits_response.json()
    visits_df = pd.DataFrame(visits_data)
    print("Визиты загружены успешно!")
else:
    print(f"Ошибка при запросе визитов: {visits_response.status_code}")


# In[8]:


registrations_url = f"{base_url}/registrations?begin={start_date}&end={end_date}"
registrations_response = requests.get(registrations_url)

if registrations_response.status_code == 200:
    registrations_data = registrations_response.json()
    registrations_df = pd.DataFrame(registrations_data)
    print("Регистрации загружены успешно!")
else:
    print(f"Ошибка при запросе регистраций: {registrations_response.status_code}")


# In[9]:


print("Визиты:")
display(visits_df.head())

print("\nРегистрации:")
display(registrations_df.head())


# In[10]:


visits_df.to_csv("api_visits.csv", index=False)
registrations_df.to_csv("api_registrations.csv", index=False)


# In[11]:


# Преобразуем datetime в дату
visits_df['date_group'] = pd.to_datetime(visits_df['datetime']).dt.date
registrations_df['date_group'] = pd.to_datetime(registrations_df['datetime']).dt.date


# In[12]:


# Исключаем ботов
filtered_visits_df = visits_df[~visits_df['user_agent'].str.contains('bot', case=False)]


# In[13]:


# Считаем количество визитов по дате и платформе
visits_grouped = (
    filtered_visits_df
    .groupby(['date_group', 'platform'])
    .agg(visits=('visit_id', 'nunique'))
    .reset_index()
)


# In[14]:


# Считаем регистрации по дате и платформе
registrations_grouped = (
    registrations_df
    .groupby(['date_group', 'platform'])
    .agg(registrations=('user_id', 'nunique'))
    .reset_index()
)


# In[15]:


# Объединяем по дате и платформе
conversion_df = pd.merge(visits_grouped, registrations_grouped,
                         on=['date_group', 'platform'],
                         how='outer').fillna(0)

# Приведение типов
conversion_df['visits'] = conversion_df['visits'].astype(int)
conversion_df['registrations'] = conversion_df['registrations'].astype(int)

# Расчет конверсии (%)
conversion_df['conversion'] = (conversion_df['registrations'] / conversion_df['visits']) * 100
conversion_df['conversion'] = conversion_df['conversion'].round(6)


# In[16]:


conversion_df = conversion_df.sort_values(by='date_group').reset_index(drop=True)


# In[17]:


conversion_df.to_json("conversion.json")


# In[18]:


ads_df = pd.read_csv("ads.csv")


# In[19]:


# Преобразуем дату к типу datetime
ads_df['date_group'] = pd.to_datetime(ads_df['date']).dt.date

# Группировка по дате и кампании с суммой затрат
ads_grouped = (
    ads_df
    .groupby(['date_group', 'utm_campaign'], dropna=False)
    .agg(cost=('cost', 'sum'))
    .reset_index()
)


# In[20]:


# Группируем conversion_df по дате (без платформ)
agg_conversion = (
    conversion_df
    .groupby('date_group')
    .agg(visits=('visits', 'sum'),
         registrations=('registrations', 'sum'))
    .reset_index()
)


# In[21]:


# Объединяем по дате
ads_merged_df = pd.merge(agg_conversion, ads_grouped, on='date_group', how='outer')

# Подставляем значения по умолчанию
ads_merged_df['utm_campaign'] = ads_merged_df['utm_campaign'].fillna('none')
ads_merged_df['cost'] = ads_merged_df['cost'].fillna(0).astype(int)
ads_merged_df['visits'] = ads_merged_df['visits'].fillna(0).astype(int)
ads_merged_df['registrations'] = ads_merged_df['registrations'].fillna(0).astype(int)


# In[22]:


ads_merged_df = ads_merged_df.sort_values(by='date_group').reset_index(drop=True)

# Сохраняем в JSON
ads_merged_df.to_json("ads.json", orient='columns')


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
import os

# Создадим директорию для графиков, если её нет
os.makedirs("./charts", exist_ok=True)

# Стиль графиков
sns.set(style="whitegrid")


# In[24]:


total_visits = conversion_df.groupby("date_group")["visits"].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=total_visits, x="date_group", y="visits")
plt.title("Итоговые визиты по всем платформам")
plt.xlabel("Дата")
plt.ylabel("Количество визитов")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./charts/total_visits.png")
plt.close()


# In[25]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=conversion_df, x="date_group", y="visits", hue="platform")
plt.title("Визиты с разбивкой по платформам")
plt.xlabel("Дата")
plt.ylabel("Визиты")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./charts/visits_by_platform.png")
plt.close()


# In[26]:


total_regs = conversion_df.groupby("date_group")["registrations"].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=total_regs, x="date_group", y="registrations")
plt.title("Итоговые регистрации по всем платформам")
plt.xlabel("Дата")
plt.ylabel("Регистрации")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./charts/total_registrations.png")
plt.close()


# In[27]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=conversion_df, x="date_group", y="registrations", hue="platform")
plt.title("Регистрации с разбивкой по платформам")
plt.xlabel("Дата")
plt.ylabel("Регистрации")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./charts/registrations_by_platform.png")
plt.close()


# In[28]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=conversion_df, x="date_group", y="conversion", hue="platform")
plt.title("Конверсия по платформам")
plt.xlabel("Дата")
plt.ylabel("Конверсия (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./charts/conversion_by_platform.png")
plt.close()


# In[29]:


avg_conversion = conversion_df.groupby("date_group")["conversion"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_conversion, x="date_group", y="conversion")
plt.title("Средняя конверсия (все платформы)")
plt.xlabel("Дата")
plt.ylabel("Конверсия (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./charts/avg_conversion.png")
plt.close()


# In[30]:


plt.figure(figsize=(12, 6))
sns.barplot(data=ads_merged_df, x="date_group", y="cost", hue="utm_campaign", dodge=False)
plt.title("Затраты на рекламу по дням")
plt.xlabel("Дата")
plt.ylabel("Стоимость")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("./charts/ad_costs_by_campaign.png")
plt.close()


# In[31]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=ads_merged_df, x="date_group", y="visits", hue="utm_campaign")
plt.title("Визиты с выделением рекламных кампаний")
plt.xlabel("Дата")
plt.ylabel("Визиты")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("./charts/visits_with_ads.png")
plt.close()


# In[32]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=ads_merged_df, x="date_group", y="registrations", hue="utm_campaign")
plt.title("Регистрации с выделением рекламных кампаний")
plt.xlabel("Дата")
plt.ylabel("Регистрации")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("./charts/registrations_with_ads.png")
plt.close()


# In[33]:


# 1. Импорт библиотек
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Загрузка переменных окружения
load_dotenv()

API_URL = os.getenv('API_URL')
DATE_BEGIN = os.getenv('DATE_BEGIN')
DATE_END = os.getenv('DATE_END')

# 3. Создание папки для графиков
os.makedirs('./charts', exist_ok=True)

# 4. Загрузка данных с API
visits = requests.get(f"{API_URL}/visits", params={'begin': DATE_BEGIN, 'end': DATE_END}).json()
registrations = requests.get(f"{API_URL}/registrations", params={'begin': DATE_BEGIN, 'end': DATE_END}).json()

# 5. Преобразование в датафреймы
visits_df = pd.DataFrame(visits)
registrations_df = pd.DataFrame(registrations)

# 6. Предобработка и фильтрация визитов
visits_df['datetime'] = pd.to_datetime(visits_df['datetime'])
visits_df['date_group'] = visits_df['datetime'].dt.date
visits_df = visits_df[~visits_df['user_agent'].str.contains('bot', case=False)]

# 7. Группировка визитов
visits_grouped = visits_df.groupby(['date_group', 'platform']).agg(visits=('visit_id', 'nunique')).reset_index()

# 8. Предобработка регистраций
registrations_df['datetime'] = pd.to_datetime(registrations_df['datetime'])
registrations_df['date_group'] = registrations_df['datetime'].dt.date

# 9. Группировка регистраций
registrations_grouped = registrations_df.groupby(['date_group', 'platform']).agg(registrations=('user_id', 'nunique')).reset_index()

# 10. Объединение и расчет конверсии
merged_df = pd.merge(visits_grouped, registrations_grouped, on=['date_group', 'platform'], how='outer').fillna(0)
merged_df['conversion'] = (merged_df['registrations'] / merged_df['visits'].replace({0: np.nan})) * 100
merged_df['conversion'] = merged_df['conversion'].fillna(0)

# 11. Сохранение в JSON
merged_df.to_json('./conversion.json')

# 12. Загрузка и обработка рекламы
ads_df = pd.read_csv('./ads.csv', parse_dates=['date'])
ads_df['date_group'] = ads_df['date'].dt.date
ads_grouped = ads_df.groupby('date_group').agg(
    cost=('cost', 'sum'),
    utm_campaign=('utm_campaign', lambda x: ', '.join(set(x)))
).reset_index()

# 13. Объединение с основной таблицей
ads_merged = pd.merge(
    merged_df.groupby('date_group').agg({
        'visits': 'sum',
        'registrations': 'sum'
    }).reset_index(),
    ads_grouped,
    on='date_group',
    how='left'
)

ads_merged['utm_campaign'] = ads_merged['utm_campaign'].fillna('none')
ads_merged['cost'] = ads_merged['cost'].fillna(0)

# 14. Сохранение в JSON
ads_merged.to_json('./ads.json')

# 15. Визуализация
sns.set(style="whitegrid")

# Итоговые визиты
plt.figure(figsize=(10, 5))
merged_df.groupby('date_group')['visits'].sum().plot()
plt.title('Итоговые визиты')
plt.xlabel('Дата')
plt.ylabel('Визиты')
plt.tight_layout()
plt.savefig('./charts/total_visits.png')
plt.close()

# Итоговые визиты по платформам
plt.figure(figsize=(10, 5))
sns.lineplot(data=merged_df, x='date_group', y='visits', hue='platform')
plt.title('Визиты по платформам')
plt.xlabel('Дата')
plt.ylabel('Визиты')
plt.tight_layout()
plt.savefig('./charts/visits_by_platform.png')
plt.close()

# Итоговые регистрации
plt.figure(figsize=(10, 5))
merged_df.groupby('date_group')['registrations'].sum().plot()
plt.title('Итоговые регистрации')
plt.xlabel('Дата')
plt.ylabel('Регистрации')
plt.tight_layout()
plt.savefig('./charts/total_registrations.png')
plt.close()

# Итоговые регистрации по платформам
plt.figure(figsize=(10, 5))
sns.lineplot(data=merged_df, x='date_group', y='registrations', hue='platform')
plt.title('Регистрации по платформам')
plt.xlabel('Дата')
plt.ylabel('Регистрации')
plt.tight_layout()
plt.savefig('./charts/registrations_by_platform.png')
plt.close()

# Конверсия по платформам
plt.figure(figsize=(10, 5))
sns.lineplot(data=merged_df, x='date_group', y='conversion', hue='platform')
plt.title('Конверсия по платформам')
plt.xlabel('Дата')
plt.ylabel('Конверсия (%)')
plt.tight_layout()
plt.savefig('./charts/conversion_by_platform.png')
plt.close()

# Средняя конверсия
plt.figure(figsize=(10, 5))
merged_df.groupby('date_group')['conversion'].mean().plot()
plt.title('Средняя конверсия')
plt.xlabel('Дата')
plt.ylabel('Конверсия (%)')
plt.tight_layout()
plt.savefig('./charts/average_conversion.png')
plt.close()

# Стоимости рекламы
plt.figure(figsize=(10, 5))
ads_merged.set_index('date_group')['cost'].plot()
plt.title('Затраты на рекламу')
plt.xlabel('Дата')
plt.ylabel('Стоимость')
plt.tight_layout()
plt.savefig('./charts/ads_costs.png')
plt.close()

# Визиты с выделением рекламной кампании
plt.figure(figsize=(10, 5))
sns.scatterplot(data=ads_merged, x='date_group', y='visits', hue='utm_campaign')
plt.title('Визиты с рекламой')
plt.tight_layout()
plt.savefig('./charts/visits_with_ads.png')
plt.close()

# Регистрации с выделением рекламной кампании
plt.figure(figsize=(10, 5))
sns.scatterplot(data=ads_merged, x='date_group', y='registrations', hue='utm_campaign')
plt.title('Регистрации с рекламой')
plt.tight_layout()
plt.savefig('./charts/registrations_with_ads.png')
plt.close()


# In[3]:


import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()
API_URL = os.getenv('API_URL')
DATE_BEGIN = os.getenv('DATE_BEGIN')
DATE_END = os.getenv('DATE_END')

# Создание папки charts, если нет
os.makedirs('./charts', exist_ok=True)

# Загружаем данные по API
response = requests.get(f"{API_URL}/registrations", params={'begin': DATE_BEGIN, 'end': DATE_END})
registrations = pd.DataFrame(response.json())

# Сохраняем в JSON
registrations.to_json('./registrations.json', orient='records')

# Подготовка данных
registrations['datetime'] = pd.to_datetime(registrations['datetime'])
registrations['date_group'] = registrations['datetime'].dt.date

# Группировка по типу регистрации
grouped = registrations.groupby(['date_group', 'registration_type']).size().unstack(fill_value=0)

# Построение графика
plt.figure(figsize=(14, 7))
for column in grouped.columns:
    plt.plot(grouped.index, grouped[column], label=column)

plt.title('Регистрации по типу регистрации')
plt.xlabel('Дата')
plt.ylabel('Количество регистраций')
plt.legend(title='Тип регистрации')
plt.grid(True)
plt.tight_layout()
plt.savefig('./charts/registrations_by_type.png')
plt.close()


# In[ ]:




