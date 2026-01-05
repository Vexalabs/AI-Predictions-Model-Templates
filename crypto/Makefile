.PHONY: install run test lint build docker-run

install:
	pip install -r requirements.txt

run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8080

test:
	pytest tests/

lint:
	black .

build:
	docker build -t crypto-model .

docker-run:
	docker run -p 8080:8080 crypto-model
