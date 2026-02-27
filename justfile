# FlashSR-Go justfile
# Run `just` (https://github.com/casey/just) for convenience targets.

default: ci

# Run all tests
test:
    go test ./...

# Run tests with race detector
test-race:
    go test -race ./...

# Run golden tests (requires FLASHSR_ORT_LIB to be set)
test-golden:
    go test -tags golden ./...

# Run linter
lint:
    golangci-lint run ./...

# Format code
fmt:
    gofmt -w .
    goimports -w . || true

# Vet
vet:
    go vet ./...

# Run benchmarks
bench:
    go test -bench=. -benchmem -run=^$ ./...

# Build CLI binary
build:
    go build -o bin/flashsr ./cmd/flashsr

# Full CI: fmt check + vet + lint + test-race
ci: vet test-race

# Tidy module
tidy:
    go mod tidy

# Show help
help:
    @just --list
