data:
	python data/generate_data.py
train:
	python src/train.py
test:
	python -m pytest -q tests
api:
	cd src && uvicorn api:app --reload
docker:
	docker build -t churn-api . && docker run -p 8000:8000 churn-api
