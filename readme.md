## Installation
1. Open folder in WSL
2. in root project
```cp .env.example .env```
3. in .env file, change the path value of SQLITE_PATH
4. ```cd back_django```
5. ```python3 -m venv .venv```
6. ```source .venv/bin/activate```
7. ```pip install -r requirements.txt```
8. ```make migrate```
9. ```yarn install```
10. ```make asset```
11. ```make start```

Server is lauched!


## Configuration
python3 src/manage.py startapp nameofffolder
create your entities
python3 src/manage.py makemigrations appname
