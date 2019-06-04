
help:
	cat Makefile
run:
	docker-compose up
exec:
	docker-compose exec handson-ml2 bash
build: stop .FORCE
	docker-compose build
rebuild: stop .FORCE
	docker-compose build --force-rm
stop:
	docker stop handson-ml2 || true; docker rm handson-ml2 || true;
.FORCE:
