
help:
	cat Makefile
run:
	docker-compose up
exec:
	docker-compose exec -ti hondson-ml /bin/bash
build: stop .FORCE
	docker-compose build --force-rm 
stop:
	docker stop handson-ml || true
.FORCE:
