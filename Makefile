.PHONY: init dev

init:
	source .env/bin/activate

dev:
	maturin develop