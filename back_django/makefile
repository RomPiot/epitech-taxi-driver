help:
	python3 src/manage.py --help

start_old:
	python3 src/manage.py runserver 8001

start:
	python3 src/manage.py runserver_plus 8001

migration:
	python3 src/manage.py makemigrations

migrate:
	python3 src/manage.py migrate

freeze:
	pip freeze > requirements.txt

install:
	pip install -r requirements.txt

source:
	source .venv/bin/activate

asset:
	yarn start

graph:
	python3 src/manage.py graph_models -a -o graph.dot
	
urls:
	python3 src/manage.py show_urls

shell: 
	python3 src/manage.py shell_plus

rename_app: 
	@python3 src/manage.py rename_app $(args)

admin: 
	@python3 src/manage.py admin_generator $(args)

superadmin: 
	python3 src/manage.py createsuperuser 

clear_cache: 
	sudo python3 src/manage.py clean_pyc


settings: 
	python3 src/manage.py print_settings
