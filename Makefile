# Aylien entities Makefile

# Container
#REGISTRY      := gcr.io/aylien-production
#CONTAINER     :=
#VERSION       := `cat VERSION`

# EVALUATION ARGS
EVALUATION_DATASET      := data/WCEP/test.jsonl
MODEL_ID                := bart-large-cnn
MAX_ARTICLES_IN_CLUSTER := 5

## Resources
#RESOURCES_ROOT    := gs://aylien-science-files/dynamic-ensembles
#RESOURCES_VERSION ?= 1
#
#TEST_RESOURCES_VERSION ?= test
#

# WCEP dataset location
# gs://aylien-science-datasets/summarization/MultiNews/
# gs://aylien-science-datasets/summarization/WCEP/
# gsutil cp -r gs://aylien-science-datasets/summarization/WCEP

RUN_FLAGS    ?=

###########
## TASKS ##
###########
resources/$(TEST_RESOURCES_VERSION):
	mkdir -p ./resources
	gsutil cp -r $(RESOURCES_ROOT)/$(TEST_RESOURCES_VERSION) ./resources

resources/$(RESOURCES_VERSION):
	mkdir -p ./resources
	gsutil cp -r $(RESOURCES_ROOT)/$(RESOURCES_VERSION) ./resources

.PHONY: test
test: resources/$(TEST_RESOURCES_VERSION)
	RESOURCES=resources/$(TEST_RESOURCES_VERSION) python -Wignore -m unittest discover
	pycodestyle aylien_entity_linking

#evaluate: resources/$(RESOURCES_VERSION)
.PHONY: evaluate
evaluate:
	python transformer_decoding/evaluate.py \
		--evaluation-dataset $(EVALUATION_DATASET) \
		--model-id $(MODEL_ID) \
		$(RUN_FLAGS)

.PHONY: clean
clean:
	rm -f *.pyc *.pkl *.npy
	rm -rf *.egg-info

#################
## DEVELOPMENT ##
#################

.PHONY: dev
dev:
	pip install -e .
	pip install -r requirements.txt
