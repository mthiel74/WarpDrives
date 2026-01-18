.PHONY: setup test demo clean lint format help install-dev

PYTHON := python3
PIP := pip

help:
	@echo "WarpBubbleSim - GR Warp Bubble Spacetime Simulator"
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Install the package and dependencies"
	@echo "  install-dev - Install with development dependencies"
	@echo "  test        - Run test suite"
	@echo "  demo        - Generate demo outputs"
	@echo "  lint        - Run linters"
	@echo "  format      - Format code with black"
	@echo "  clean       - Remove generated files"
	@echo ""

setup:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,notebooks]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

demo: setup
	@echo "Generating demo outputs..."
	@mkdir -p out
	$(PYTHON) -m warpbubblesim.cli.main render --metric alcubierre \
		--scenario scenarios/alcubierre_demo.yaml --output out/
	$(PYTHON) -m warpbubblesim.cli.main geodesics --metric alcubierre \
		--output out/alcubierre_geodesics.mp4
	$(PYTHON) -m warpbubblesim.cli.main render --metric natario \
		--scenario scenarios/natario_demo.yaml --output out/
	$(PYTHON) -m warpbubblesim.cli.main render --metric bobrick_martire \
		--scenario scenarios/bobrick_martire_demo.yaml --output out/
	$(PYTHON) -m warpbubblesim.cli.main render --metric lentz \
		--scenario scenarios/lentz_demo.yaml --output out/
	@echo "Demo outputs generated in out/"

demo-quick: setup
	@echo "Generating quick demo outputs (low resolution)..."
	@mkdir -p out
	$(PYTHON) -m warpbubblesim.cli.main render --metric alcubierre \
		--scenario scenarios/alcubierre_demo.yaml --output out/ --resolution 64
	@echo "Quick demo output generated in out/"

lint:
	$(PYTHON) -m ruff check warpbubblesim/ tests/
	$(PYTHON) -m mypy warpbubblesim/ --ignore-missing-imports

format:
	$(PYTHON) -m black warpbubblesim/ tests/
	$(PYTHON) -m ruff check --fix warpbubblesim/ tests/

clean:
	rm -rf out/*.png out/*.mp4 out/*.gif
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build dist
	rm -rf .coverage htmlcov

ci-test:
	$(PYTHON) -m pytest tests/ -v --tb=short -x

ci-demo:
	@mkdir -p out
	$(PYTHON) -m warpbubblesim.cli.main render --metric alcubierre \
		--scenario scenarios/alcubierre_demo.yaml --output out/ --resolution 32
