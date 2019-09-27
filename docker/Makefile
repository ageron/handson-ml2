
help:
	cat Makefile
run:
	docker-compose up --build
exec:
	docker-compose exec handson-ml2 bash
build: stop .FORCE
	docker-compose build
rebuild: stop .FORCE
	docker-compose build --force-rm
stop:
	docker-compose down || true;
.FORCE:
