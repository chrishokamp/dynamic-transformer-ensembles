# Aylien entities Makefile

# Container
#REGISTRY      := gcr.io/aylien-production
#CONTAINER     :=
#VERSION       := `cat VERSION`

EVALUATION_DATASET ?= data/aida-conll-test/tiny-aida-conll-yago.json

# Resources
RESOURCES_ROOT    := gs://aylien-science-files/entities/resources
RESOURCES_VERSION ?= 1

TEST_RESOURCES_VERSION ?= test

SPOTTER ?= ner
LINKER  ?= FastTextEntityLinker

# Redirect Resolution args
NUM_PROCS        ?= 4
BATCH_SIZE       ?= 1000
FILES_TO_PROCESS ?= entity_defs.csv prior_prob.csv

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

.PHONY: evaluate
evaluate: resources/$(RESOURCES_VERSION)
	python aylien_entity_linking/evaluate.py \
		--resource-path resources/$(RESOURCES_VERSION) \
		--evaluation-dataset $(EVALUATION_DATASET) \
		--strict-boundary \
		--spotter $(SPOTTER) \
		--linker $(LINKER) \
		$(RUN_FLAGS)

# resolve redirects in entity_defs.csv and prior_prob.csv
# this task assumes mysql is setup correctly and tables are loaded,
#  see README
.PHONY: resolve-redirects
resolve-redirects:
	python bin/resolve_redirects.py \
		--resource-path resources/$(RESOURCES_VERSION) \
		--files-to-process $(FILES_TO_PROCESS) \
		--num-procs $(NUM_PROCS) \
		--batch-size $(BATCH_SIZE) \
		$(RUN_FLAGS)

.PHONY: candidates
candidates:
	python bin/candidates.py \
		--output resources/$(RESOURCES_VERSION)/candidates.json \
		--prior-probs resources/$(RESOURCES_VERSION)/resolved_prior_prob.csv \
		--entity-defs resources/$(RESOURCES_VERSION)/resolved_entity_defs.csv \
		$(RUN_FLAGS)

.PHONY: entity-vectors
entity-vectors:
	python bin/entity_vectors.py \
		--storage resources/$(RESOURCES_VERSION)/fasttext-vectorizer \
		--entity-descriptions $(ENTITY_DESCRIPTIONS) \
		--output resources/$(RESOURCES_VERSION)/entity-vectors.csv

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
	echo 'You currently need to manually install faiss: https://github.com/facebookresearch/faiss'
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
