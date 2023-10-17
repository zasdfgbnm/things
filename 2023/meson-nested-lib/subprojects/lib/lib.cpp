#include "lib.h"
#include <iostream>
#include <vector>

static std::vector<const Backend*> backends;

void hello() {
  for (auto b : backends) {
    std::cout << "Hello from " << b->name() << std::endl;
  }
}

void registerBackend(const Backend* b) {
  backends.push_back(b);
}
