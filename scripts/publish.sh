#!/bin/bash
# publish.sh - Build and publish all cosma packages to PyPI
# Usage:
#   ./scripts/publish.sh build          # Build all packages
#   ./scripts/publish.sh publish        # Publish all packages to PyPI
#   ./scripts/publish.sh test           # Publish to TestPyPI
#   ./scripts/publish.sh all            # Build and publish to PyPI
#   ./scripts/publish.sh clean          # Clean all build artifacts

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Package order (backend and tui first, then main package)
PACKAGES=(
    "packages/cosma-backend"
    "packages/cosma-tui"
    "."
)

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Build all packages
build_all() {
    log_info "Building all packages..."
    
    for pkg in "${PACKAGES[@]}"; do
        if [ ! -d "$pkg" ]; then
            log_error "Package directory $pkg does not exist"
            exit 1
        fi
        
        pkg_name=$(basename "$pkg")
        log_info "Building $pkg_name..."
        
        (cd "$pkg" && uv build)
        
        log_success "$pkg_name built successfully"
    done
    
    log_success "All packages built!"
}

# Publish all packages to PyPI
publish_all() {
    log_info "Publishing all packages to PyPI..."
    
    for pkg in "${PACKAGES[@]}"; do
        pkg_name=$(basename "$pkg")
        log_info "Publishing $pkg_name..."
        
        (cd "$pkg" && uv publish)
        
        log_success "$pkg_name published successfully"
    done
    
    log_success "All packages published to PyPI!"
}

# Publish all packages to TestPyPI
publish_test() {
    log_info "Publishing all packages to TestPyPI..."
    
    for pkg in "${PACKAGES[@]}"; do
        pkg_name=$(basename "$pkg")
        log_info "Publishing $pkg_name to TestPyPI..."
        
        (cd "$pkg" && uv publish --index testpypi)
        
        log_success "$pkg_name published to TestPyPI"
    done
    
    log_success "All packages published to TestPyPI!"
    log_warning "Test installation with: uvx --index https://test.pypi.org/simple/ cosma"
}

# Clean all build artifacts
clean_all() {
    log_info "Cleaning build artifacts..."
    
    for pkg in "${PACKAGES[@]}"; do
        pkg_name=$(basename "$pkg")
        
        if [ -d "$pkg/dist" ]; then
            log_info "Cleaning $pkg_name/dist..."
            rm -rf "$pkg/dist"
        fi
        
        # Also clean egg-info directories
        find "$pkg" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    done
    
    log_success "All build artifacts cleaned!"
}

# Show usage
show_usage() {
    echo "Usage: $0 {build|publish|test|all|clean}"
    echo ""
    echo "Commands:"
    echo "  build     Build all packages"
    echo "  publish   Publish all packages to PyPI"
    echo "  test      Publish all packages to TestPyPI"
    echo "  all       Build and publish to PyPI"
    echo "  clean     Clean all build artifacts"
    echo ""
    echo "Examples:"
    echo "  $0 build          # Just build packages"
    echo "  $0 test           # Test publish to TestPyPI"
    echo "  $0 all            # Build and publish to PyPI"
}

# Main script logic
case "${1:-}" in
    build)
        build_all
        ;;
    publish)
        publish_all
        ;;
    test)
        build_all
        publish_test
        ;;
    all)
        build_all
        publish_all
        ;;
    clean)
        clean_all
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
