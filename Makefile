.PHONY: all test clean
# Test 
test:
	python -m unittest discover -v
# Install commands 
create_eviroment: 
	conda create --name NNInitialization --file requirements.txt
updatate_requirement:
	pip freeze > requirements.txt