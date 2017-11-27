
help:
	cat Makefile
run:
	docker-compose up
exec:
	docker-compose exec handson-ml /bin/bash
build: stop .FORCE
	docker-compose build
rebuild: stop .FORCE
	docker-compose build --force-rm
stop:
	docker stop handson-ml || true; docker rm handson-ml || true;
.FORCE:
