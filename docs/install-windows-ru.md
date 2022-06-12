# Установка для Windows

1. Скачайте [classibundler.zip](https://github.com/mikhail-matrosov/classibundler/releases/download/main/classibundler.zip) из последнего релиза и распакуйте в любую папку на вашем компьютере. Допустим, это будет `C:\Users\User\Desktop\classibundler`.

2. Установите Python. Чтобы проверить, что он уже установлен, в меню Пуск найдите `Anaconda PowerShell Prompt`. Если нет, то скачать Python проще всего с сайта [anaconda.com](https://www.anaconda.com/), в установщике можно следовать рекомендованным настройкам. Мы рекомендуем именно эту версию Python, потому что с ней проще всего начать работать.

3. В меню Пуск откройте `Anaconda PowerShell Prompt` и наберите команды:
```sh
cd .\Desktop\classibundler
pip install -r requirements.txt
notepad config.py
```

4. В файле `config.py` в строчке 23 укажите путь к папке с вашими пациентами.

5. Настройка закончена, можно пользоваться скриптами. Например, в новом `Anaconda PowerShell Prompt` наберите:
```sh
cd .\Desktop\classibundler
python draw_profiles.py path\to\your\patient\folder\IvanovIvan
```
