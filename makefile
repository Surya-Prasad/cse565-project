# Makefile for running tests and scoring outputs

#  DO NOT CHANGE THESE VARIABLES 
PYTHON := python3
SCORE_SCRIPT:= score.py
TIME_LIMIT:= 10m
MEM_LIMIT:= 1048576
TEST_DIR:= test-instances
STUDENT_OUT_DIR:= student-test-outputs
TIMEOUT := gtimeout
#  YOU MAY CHANGE THE VARIABLES BELOW AS NEEDED
MAIN_SCRIPT:= main.py
STUDENT_TEST_DIR:= student-test-cases
OUT_DIR:= outputs

TEST_FILES := $(wildcard $(TEST_DIR)/*.graph)
STUDENT_TEST_FILES := $(wildcard $(STUDENT_TEST_DIR)/*.graph)

# Phony targets
.PHONY: all run score clean test

# Default target
all: score

# --- Target: make run ---
# Runs main.py on .graph files and saves output to .out files
run: clean
	@echo "--- Executing Student Tests ---"
	@for file in $(STUDENT_TEST_FILES); do \
		NAME=$$(basename $$file .graph); \
		echo "Running: $$file "; \
		$ $(TIME_LIMIT) $(PYTHON) $(MAIN_SCRIPT) "$$file"; \
		EXIT_CODE=$$?; \
		echo "TEST. EXIT CODE $$EXIT_CODE" >> $(OUT_DIR)/$$NAME.out; \
		if [ $$EXIT_CODE -eq 124 ]; then \
			echo "TIMED OUT" >> $(OUT_DIR)/$$NAME.out; \
			echo "  Warning: $$NAME timed out after $(TIME_LIMIT)."; \
		elif [ $$EXIT_CODE -ne 0 ]; then \
			echo "MEMORY ERROR" >> $(OUT_DIR)/$$NAME.out; \
			echo "  Warning: $$NAME failed (Exit Code: $$EXIT_CODE) MemoryError."; \
		fi \
	done
	-mv $(OUT_DIR)/*.out $(STUDENT_OUT_DIR)/
	@echo "Student Tests execution complete."
	@echo ""

	@echo "--- Executing Common Tests ---"
	@for file in $(TEST_FILES); do \
		NAME=$$(basename $$file .graph); \
		echo "Running: $$file "; \
		$(TIMEOUT) $(TIME_LIMIT) $(PYTHON) $(MAIN_SCRIPT) "$$file"; \
		EXIT_CODE=$$?; \
		echo "TEST. EXIT CODE $$EXIT_CODE" >> $(OUT_DIR)/$$NAME.out; \
		if [ $$EXIT_CODE -eq 124 ]; then \
			echo "  Warning: $$NAME timed out after $(TIME_LIMIT)."; \
		elif [ $$EXIT_CODE -ne 0 ]; then \
			echo "  Warning: $$NAME failed (Exit Code: $$EXIT_CODE) MemoryError."; \
		fi \
	done
	@echo "Common Tests execution complete."
	@echo ""

# --- Target: make score ---
# Scores output files in OUT_DIR against .truth files in TEST_DIR
score: run
	@echo "--- Scoring Student Test Results ---"
	$(PYTHON) $(SCORE_SCRIPT) $(STUDENT_TEST_DIR) $(STUDENT_OUT_DIR)
	
	@echo "--- Scoring Common Test Results ---"
	$(PYTHON) $(SCORE_SCRIPT) $(TEST_DIR) $(OUT_DIR)
	
	@echo "Scoring complete."

# --- Target: make clean ---
# Cleans up output directories
clean:
	@echo "Cleaning up..."
	-rm -rf $(STUDENT_OUT_DIR)
	@mkdir -p $(STUDENT_OUT_DIR)
	-rm -rf $(OUT_DIR)
	@mkdir -p $(OUT_DIR)
	@echo "Clean complete."