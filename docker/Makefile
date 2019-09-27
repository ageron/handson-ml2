
help:
	cat Makefile
run:
	docker-compose up --build
exec:
	docker-compose exec handson-ml2 bash
build: stop .FORCE
	docker-compose build
rebuild: stop .FORCE
	docker-compose up --build
stop:
	docker-compose down || true;
.FORCE:
