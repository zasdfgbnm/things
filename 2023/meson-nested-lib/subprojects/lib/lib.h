#pragma once

#include <string>

void hello();

struct Backend {
  virtual std::string name() const = 0;
};

void registerBackend(const Backend* b);
