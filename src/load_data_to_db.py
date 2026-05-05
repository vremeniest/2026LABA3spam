import psycopg2

# DSN строка подключения (host=postgres, т.к. в Docker Compose сервис называется postgres)
DSN = "host=postgres port=5432 dbname=spam_db user=spam_user password=spam_password"

def load_data():
    try:
        print("Подключаюсь к PostgreSQL...")
        conn = psycopg2.connect(DSN)
        cur = conn.cursor()
        print("✅ Подключено успешно!")
        
        # Создаём таблицу
        cur.execute('''
            CREATE TABLE IF NOT EXISTS raw_messages (
                id SERIAL PRIMARY KEY,
                label TEXT,
                text TEXT
            )
        ''')
        
        # Читаем файл построчно
        count = 0
        with open('data/raw/sms.tsv', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    label = parts[0]
                    text = parts[1]
                    cur.execute(
                        "INSERT INTO raw_messages (label, text) VALUES (%s, %s)",
                        (label, text)
                    )
                    count += 1
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Загружено {count} сообщений")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_data()