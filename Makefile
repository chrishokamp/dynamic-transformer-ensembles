# Aylien entities Makefile

# Container
#REGISTRY      := gcr.io/aylien-production
#CONTAINER     :=
#VERSION       := `cat VERSION`

## Resources
#RESOURCES_ROOT    := gs://aylien-science-files/dynamic-ensembles
#RESOURCES_VERSION ?= 1

#TEST_RESOURCES_VERSION ?= test

# WCEP dataset location
# gs://aylien-science-datasets/summarization/MultiNews/
# gs://aylien-science-datasets/summarization/WCEP/
# gsutil cp -r gs://aylien-science-datasets/summarization/WCEP

# FINE-TUNING (Training BART with in-domain data)
# summarization datadir (TODO: format of this data?)
DATADIR                := 'data/test_dataset'
BASE_MODEL_NAME_OR_PATH := 'bart-large'
OUTPUT_DIR             := 'fine-tuned-model'

# EVALUATION ARGS
EVALUATION_DATASET      := data/WCEP/test.jsonl
MODEL_ID                := bart-large-cnn
MAX_ARTICLES_IN_CLUSTER := 5

# used for flags and additional script args
RUN_FLAGS    ?=


###########
## TASKS ##
###########
#evaluate: resources/$(RESOURCES_VERSION)
.PHONY: evaluate
evaluate:
	python transformer_decoding/evaluate.py \
		--evaluation-dataset $(EVALUATION_DATASET) \
		--model-id $(MODEL_ID) \
		$(RUN_FLAGS)

.PHONY: fine-tune-bart
fine-tune-bart:
	mkdir -p $(OUTPUT_DIR)
	python bin/run_bart_sum.py \
		--data_dir=$(DATADIR) \
		--model_type=bart \
		--model_name_or_path=$(BASE_MODEL_NAME_OR_PATH) \
		--learning_rate=3e-5 \
		--train_batch_size=4 \
		--eval_batch_size=4 \
		--output_dir=$(OUTPUT_DIR) \
		--n_gpu=0 \
		--do_train

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
